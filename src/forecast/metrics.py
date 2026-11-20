"""Seasonal-naive benchmark and forecast metrics (MASE, sMAPE, pinball)."""
import numpy as np
import pandas as pd

from .constants import HORIZON_ORDER, QUANTILES, SEASONAL_LAG, TARGETS

METRICS = ["MASE", "MAE_vs_SNaive", "sMAPE", "Pinball", "Scaled_Pinball"]


def seasonal_naive(train_raw, eval_dates, origin):
    """Repeat the last 52 weeks. train_raw: DataFrame(index=date, cols=TARGETS), training rows only."""
    j = (eval_dates - origin).days.to_numpy()
    m = j // SEASONAL_LAG + 1                                   # 1 for days 0-363, 2 for day 364
    src = eval_dates - pd.to_timedelta(SEASONAL_LAG * m, unit="D")
    vals = train_raw.reindex(src).to_numpy(dtype=float, copy=True)
    miss = np.isnan(vals[:, 0])                                 # e.g. a removed 29 Feb -> same weekday, a week earlier
    if miss.any():
        vals[miss] = train_raw.reindex(src[miss] - pd.Timedelta(days=7)).to_numpy(dtype=float)
    return np.nan_to_num(vals)


def mase_scale(train_raw, pub_train):
    """In-sample MAE of the seasonal-naive forecast (publication days only): the MASE denominator."""
    lagged = train_raw.reindex(train_raw.index - pd.Timedelta(days=SEASONAL_LAG)).to_numpy(dtype=float)
    cur = train_raw.to_numpy(dtype=float)
    ok = np.isfinite(lagged).all(axis=1) & (pub_train == 1)
    return np.abs(cur[ok] - lagged[ok]).mean(axis=0)


def snaive_forecasts(fd):
    return {p: {t: np.repeat(fd.snaive[p][:, k][:, None], 3, axis=1) for k, t in enumerate(TARGETS)}
            for p in fd.products}


def evaluate(fd, forecasts, model, phase):
    """One row per product x target x horizon (first month / quarter / full year), publication days only."""
    rows = []
    for p in fd.products:
        pub = fd.eval_pub[p] == 1
        for k, t in enumerate(TARGETS):
            y, q, sn, scale = fd.eval_raw[p][:, k], forecasts[p][t], fd.snaive[p][:, k], fd.mase_scale[p][k]
            for h, hm in fd.horizon_masks.items():
                m = pub & hm
                if m.sum() == 0:
                    continue
                yy, qq, ss = y[m], q[m], sn[m]
                med = qq[:, 1]
                mae = np.abs(yy - med).mean()
                den = np.abs(yy) + np.abs(med)
                smape = np.where(den > 0, 200 * np.abs(yy - med) / np.where(den > 0, den, 1), 0.0).mean()
                err = yy[:, None] - qq
                pin = np.maximum(QUANTILES * err, (QUANTILES - 1) * err).mean()
                mae_sn = np.abs(yy - ss).mean()
                rows.append(dict(Phase=phase, Fold=fd.name, Model=model, Product=p, Target=t, Horizon=h,
                                 Days=int(m.sum()), MAE=mae,
                                 MASE=mae / scale if scale > 0 else np.nan,
                                 MAE_vs_SNaive=mae / mae_sn if mae_sn > 0 else np.nan,
                                 sMAPE=smape, Pinball=pin,
                                 Scaled_Pinball=pin / scale if scale > 0 else np.nan))
    return pd.DataFrame(rows)


def summarise(results, by=("Phase", "Model", "Target", "Horizon")):
    out = results.groupby(list(by))[METRICS].mean().reset_index()
    out["Horizon"] = pd.Categorical(out["Horizon"], HORIZON_ORDER, ordered=True)
    return out.sort_values([c for c in by if c != "Model"] + ["Model"]).reset_index(drop=True)


def report_table(results, metrics=("MASE", "sMAPE", "Pinball")):
    """Mean over products: rows = target x model, columns = metric x horizon."""
    t = results.groupby(["Target", "Model", "Horizon"])[list(metrics)].mean().unstack("Horizon")
    return t.reindex(columns=pd.MultiIndex.from_product([list(metrics), HORIZON_ORDER])).round(3)