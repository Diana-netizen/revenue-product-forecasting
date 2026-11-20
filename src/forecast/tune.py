
"""Resumable hyper-parameter tuning over forward-chaining folds."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import SEED


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, indent=2, default=_json_default),
        encoding="utf-8"
    )
    temp.replace(path)


def tune_model(
    name,
    spec,
    cv_folds,
    val_names,
    keep_fraction=1 / 3,
    seed=SEED,
    checkpoint_dir=None,
):
    """
    Successive-halving tuning with fold-level checkpoints.

    Completed fold curves are saved immediately. On restart, matching
    configurations and completed folds are loaded and skipped.
    """
    rng = np.random.default_rng(seed)

    configs = [
        spec["default"]
    ] + [
        spec["sample"](rng)
        for _ in range(spec["n_configs"] - 1)
    ]

    folds = {f.name: f for f in cv_folds}
    all_names = [f.name for f in cv_folds]

    checkpoint_path = None
    curves = {}

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint.json"

        config_signature = json.dumps(
            configs, sort_keys=True, default=_json_default
        )

        if checkpoint_path.exists():
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))

            if saved.get("config_signature") != config_signature:
                raise RuntimeError(
                    f"{name}: checkpoint configuration differs from the "
                    "current tuning configuration. Use a new checkpoint "
                    "directory rather than overwriting previous results."
                )

            for key, values in saved.get("curves", {}).items():
                cid, fname = key.split("|", 1)
                curves[(int(cid), fname)] = np.asarray(values, dtype=float)

            print(
                f"{name}: restored {len(curves)} completed "
                "configuration/fold runs."
            )

    def save_checkpoint():
        if checkpoint_path is None:
            return

        payload = {
            "model": name,
            "config_signature": json.dumps(
                configs, sort_keys=True, default=_json_default
            ),
            "configs": configs,
            "curves": {
                f"{cid}|{fname}": values.tolist()
                for (cid, fname), values in curves.items()
            },
        }
        _write_json_atomic(checkpoint_path, payload)

    def run(cid, fname):
        key = (cid, fname)

        if key in curves:
            print(f"  {name} cfg{cid:02d} {fname}: already saved; skipping")
            return

        t0 = time.time()
        _, curve = spec["fit"](
            folds[fname],
            configs[cid],
            spec["max_steps"],
            track=True,
            targets=spec["tune_targets"],
        )

        curve = np.asarray(curve, dtype=float)

        if curve.size == 0 or not np.isfinite(curve).any():
            raise RuntimeError(
                f"{name} cfg{cid:02d} {fname}: no valid tuning curve returned."
            )

        curves[key] = curve
        save_checkpoint()

        print(
            f"  {name} cfg{cid:02d} {fname}: "
            f"best={np.nanmin(curve):.4f} "
            f"({time.time() - t0:.0f}s) [checkpoint saved]"
        )

    def score(cid, fnames):
        missing = [
            fname for fname in fnames
            if (cid, fname) not in curves
        ]
        if missing:
            raise RuntimeError(
                f"Missing {name} cfg{cid:02d} folds: {missing}"
            )

        mean_curve = np.nanmean(
            [curves[(cid, fname)] for fname in fnames],
            axis=0,
        )
        best_index = int(np.nanargmin(mean_curve))
        return float(mean_curve[best_index]), best_index + 1

    # Rung 1: all configurations on validation folds.
    for cid in range(len(configs)):
        for fname in val_names:
            run(cid, fname)

    r1 = {
        cid: score(cid, val_names)[0]
        for cid in range(len(configs))
    }

    survivors = sorted(r1, key=r1.get)[
        :max(2, math.ceil(len(configs) * keep_fraction))
    ]

    # Rung 2: survivors on all cross-validation folds.
    for cid in survivors:
        for fname in all_names:
            run(cid, fname)

    final = {
        cid: score(cid, all_names)
        for cid in survivors
    }

    best = min(final, key=lambda cid: final[cid][0])

    table = pd.DataFrame([
        dict(
            Model=name,
            Config=cid,
            Rung1_ValidationFolds=r1[cid],
            CV_Loss=final[cid][0] if cid in final else np.nan,
            Best_Step=final[cid][1] if cid in final else np.nan,
            Selected=cid == best,
            Params=json.dumps(configs[cid], default=_json_default),
        )
        for cid in range(len(configs))
    ])

    result = dict(
        params=configs[best],
        steps=final[best][1],
        cv_loss=final[best][0],
        table=table,
    )

    save_checkpoint()
    return result


def save_tuning(tuned, directory):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)

    summary = {
        name: {
            "params": result["params"],
            "steps": result["steps"],
            "cv_loss": result["cv_loss"],
        }
        for name, result in tuned.items()
    }

    _write_json_atomic(d / "tuned.json", summary)

    tables = [
        result["table"]
        for result in tuned.values()
        if "table" in result
    ]

    if tables:
        combined = pd.concat(tables, ignore_index=True)
        temp = d / "tuning_table.csv.tmp"
        combined.to_csv(temp, index=False)
        temp.replace(d / "tuning_table.csv")


def load_tuning(directory):
    return json.loads(
        (Path(directory) / "tuned.json").read_text(encoding="utf-8")
    )
