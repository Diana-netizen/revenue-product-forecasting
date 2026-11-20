"""Pipeline CLI:  python -m forecast.run {ingest|tune|backtest|test|production|all} --config configs/default.yaml

ingest      raw CSV -> data/processed/model_df.parquet (+ production calendar)
tune        5-fold forward-chaining CV tuning              -> artifacts/tuning/
backtest    tuned models on Oct-2023 / Oct-2024 origins    -> reports/validation_results.csv, artifacts/selection.json
test        refit on data before Oct-2025, score ONCE      -> reports/test_results.csv (+ TFT interpretation)
production  refit the selected model on all data, forecast the next year -> artifacts/production/, reports/
"""
import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from . import ingest
from .config import load_config
from .constants import TARGETS
from .folds import build_fold
from .metrics import evaluate, report_table, snaive_forecasts, summarise
from .models import get_registry
from .tune import load_tuning, save_tuning, tune_model


# ------------------------------------------------------------------ helpers
def _dirs(cfg):
    p = cfg["paths"]
    d = {k: Path(v) for k, v in p.items()}
    for k in ("processed", "artifacts", "reports"):
        d[k].mkdir(parents=True, exist_ok=True)
    return d


def _load_frame(cfg, production=False):
    name = "model_df_production.parquet" if production else "model_df.parquet"
    return pd.read_parquet(_dirs(cfg)["processed"] / name)


def _cv_folds(cfg, df):
    v = cfg["validation"]
    test_origin = pd.Timestamp(v["test_origin"])
    dev_df = df[df["Date"] < test_origin].copy()                  # development code never sees the test year
    folds = []
    for o in map(pd.Timestamp, v["cv_origins"]):
        assert o + pd.DateOffset(years=1) <= test_origin, f"fold {o.date()} overlaps the test period"
        try:
            folds.append(build_fold(dev_df, o, f"CV_{o:%Y_%m}"))
        except ValueError as e:
            print("Skipping fold:", e)
    val_origins = set(map(pd.Timestamp, v["validation_origins"]))
    val_names = [f.name for f in folds if f.origin in val_origins]
    print(f"{len(folds)} CV folds: {[f.name for f in folds]} | validation folds: {val_names}")
    return folds, val_names


def _backtest(folds, phase, registry, tuned, keep_artifacts=False):
    rows, arts = [], {}
    for fd in folds:
        rows.append(evaluate(fd, snaive_forecasts(fd), "Seasonal Naive", phase))
        for name, spec in registry.items():
            t = tuned[name]
            art, _ = spec["fit"](fd, t["params"], t["steps"], track=False, targets=TARGETS)
            rows.append(evaluate(fd, spec["predict"](art, fd, t["steps"]), name, phase))
            if keep_artifacts:
                arts[(fd.name, name)] = art
    return pd.concat(rows, ignore_index=True), arts


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


# ------------------------------------------------------------------ stages
def run_ingest(cfg):
    d = _dirs(cfg)
    raw = ingest.load_raw(d["raw_csv"], cfg)
    prod = ingest.build_model_frame(raw, cfg, extend_to=cfg["production"]["end"])
    hist = prod[prod["Date"] <= pd.Timestamp(cfg["study"]["end"])]
    hist.reset_index(drop=True).to_parquet(d["processed"] / "model_df.parquet", index=False)
    prod.to_parquet(d["processed"] / "model_df_production.parquet", index=False)
    print(f"ingest: {len(hist):,} historical rows, {len(prod):,} incl. production calendar")



def run_tune(cfg):
    d = _dirs(cfg)
    tuning_dir = d["artifacts"] / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    folds, val_names = _cv_folds(cfg, _load_frame(cfg))
    registry = get_registry(cfg)
    tuned = {}

    keep_fraction = cfg["tuning"]["keep_fraction"]

    for name, spec in registry.items():
        print(f"\n=== Tuning {name} ===")

        result_path = tuning_dir / f"{name.lower()}_result.json"
        table_path = tuning_dir / f"{name.lower()}_table.csv"

        signature = {
            "n_configs": spec["n_configs"],
            "max_steps": spec["max_steps"],
            "keep_fraction": keep_fraction,
            "folds": [f.name for f in folds],
            "validation_folds": val_names,
        }

        # Reuse a fully completed model if its tuning setup is unchanged.
        if result_path.exists() and table_path.exists():
            saved = json.loads(result_path.read_text(encoding="utf-8"))

            if saved.get("signature") != signature:
                raise RuntimeError(
                    f"{name}: saved tuning configuration differs from "
                    "the current configuration. Do not reuse this result. "
                    "Use a new tuning directory for a new search."
                )

            tuned[name] = {
                "params": saved["params"],
                "steps": saved["steps"],
                "cv_loss": saved["cv_loss"],
                "table": pd.read_csv(table_path),
            }

            print(f"{name}: completed result restored; skipping.")
            continue

        result = tune_model(
            name,
            spec,
            folds,
            val_names,
            keep_fraction=keep_fraction,
            checkpoint_dir=tuning_dir / f"{name.lower()}_folds",
        )

        tuned[name] = result

        # Save this model immediately, before moving to the next model.
        payload = {
            "signature": signature,
            "params": result["params"],
            "steps": result["steps"],
            "cv_loss": result["cv_loss"],
        }

        temp_json = result_path.with_suffix(".json.tmp")
        temp_json.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        temp_json.replace(result_path)

        temp_csv = table_path.with_suffix(".csv.tmp")
        result["table"].to_csv(temp_csv, index=False)
        temp_csv.replace(table_path)

        print(
            name,
            "->",
            result["params"],
            "steps:",
            result["steps"],
            "CV loss:",
            result["cv_loss"],
            "[model checkpoint saved]",
        )

    # Rebuild the aggregate files used by backtest and production.
    save_tuning(tuned, tuning_dir)
    print("\nTuning complete. Results saved to:", tuning_dir)



def run_backtest(cfg):
    d = _dirs(cfg)
    folds, val_names = _cv_folds(cfg, _load_frame(cfg))
    val_folds = [f for f in folds if f.name in val_names]
    registry, tuned = get_registry(cfg), load_tuning(d["artifacts"] / "tuning")
    results, _ = _backtest(val_folds, "validation", registry, tuned)
    results.to_csv(d["reports"] / "validation_results.csv", index=False)
    summarise(results).to_csv(d["reports"] / "validation_summary.csv", index=False)
    report_table(results).to_csv(d["reports"] / "report_validation.csv")
    sel = (results[(results["Horizon"] == "Full_Year") & (results["Model"] != "Seasonal Naive")]
           .groupby("Model")[["MASE", "Scaled_Pinball"]].mean().sort_values("MASE"))
    print(sel)
    best = sel.index[0]
    (d["artifacts"] / "selection.json").write_text(json.dumps(
        dict(best_model=best, criterion="full-year MASE, mean over validation folds/products/targets",
             scores=sel.to_dict()), indent=2))
    print("Selected on validation:", best)


def run_test(cfg):
    """Run once, after tuning and selection are frozen."""
    d = _dirs(cfg)
    test_fold = build_fold(_load_frame(cfg), cfg["validation"]["test_origin"], "Test", is_test=True)
    registry, tuned = get_registry(cfg), load_tuning(d["artifacts"] / "tuning")
    results, arts = _backtest([test_fold], "test", registry, tuned, keep_artifacts=True)
    results.to_csv(d["reports"] / "test_results.csv", index=False)
    summarise(results).to_csv(d["reports"] / "test_summary.csv", index=False)
    report_table(results).to_csv(d["reports"] / "report_test.csv")
    print(report_table(results))
    if "TFT" in registry:
        from .interpret import run_interpretation
        run_interpretation(arts[(test_fold.name, "TFT")], test_fold, TARGETS, d["reports"] / "interpretation")


def run_production(cfg):
    """Refit the selected model on everything up to the study end and forecast the next year."""
    d = _dirs(cfg)
    best = json.loads((d["artifacts"] / "selection.json").read_text())["best_model"]
    tuned = load_tuning(d["artifacts"] / "tuning")[best]
    spec = get_registry(cfg, [best])[best]
    fd = build_fold(_load_frame(cfg, production=True), cfg["production"]["origin"], "Production", is_test=True)
    out_dir = d["artifacts"] / "production"
    art, _ = spec["fit"](fd, tuned["params"], tuned["steps"], track=False, targets=TARGETS, save_dir=out_dir)
    spec["save"](art, out_dir, tuned["params"])
    fc = spec["predict"](art, fd, tuned["steps"])
    rows = []
    for p in fd.products:
        for k, t in enumerate(TARGETS):
            q = fc[p][t]
            rows.append(pd.DataFrame({"Date": fd.eval_dates[p], "Product": p, "Target": t,
                                      "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2],
                                      "seasonal_naive": fd.snaive[p][:, k]}))
    forecast = pd.concat(rows, ignore_index=True)
    forecast.to_csv(d["reports"] / "production_forecast.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps(dict(
        model=best, params=tuned["params"], steps=tuned["steps"], trained_through=cfg["study"]["end"],
        forecast_origin=cfg["production"]["origin"], forecast_end=cfg["production"]["end"],
        git_commit=_git_commit()), indent=2))
    print(f"production: {best} forecast {forecast['Date'].min().date()} -> {forecast['Date'].max().date()}")


STAGES = dict(ingest=run_ingest, tune=run_tune, backtest=run_backtest, test=run_test, production=run_production)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[*STAGES, "all"])
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for s in (STAGES if args.stage == "all" else [args.stage]):
        print(f"\n######## stage: {s}")
        STAGES[s](cfg)


if __name__ == "__main__":
    main()