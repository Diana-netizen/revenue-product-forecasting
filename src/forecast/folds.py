"""One forecast origin -> arrays every model consumes. Nothing after the origin leaks into training."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

from .constants import HORIZON, LOOKBACK, STRIDE, TARGETS
from .features import future_features, history_input_features, make_preprocessors
from .metrics import mase_scale, seasonal_naive


def to_dense(a):
    return (a.toarray() if sparse.issparse(a) else np.asarray(a)).astype(np.float32)


def build_fold(df, origin, name, is_test=False):
    """Preprocessors are fitted on rows BEFORE the origin only. With is_test=True the evaluation-period
    targets are never exposed (also used for production forecasts, where they do not exist yet)."""
    origin = pd.Timestamp(origin)
    span = (df[df["Date"] < origin + pd.DateOffset(years=1)]
            .sort_values(["Product", "Date"]).reset_index(drop=True))
    span[["Net Price", "Quantity"]] = span[["Net Price", "Quantity"]].fillna(0.0)
    is_tr = (span["Date"] < origin).to_numpy()

    hist_pre, fut_pre = make_preprocessors()
    hist_pre.fit(span.loc[is_tr, history_input_features])
    fut_pre.fit(span.loc[is_tr, future_features])
    H = to_dense(hist_pre.transform(span[history_input_features]))
    F = to_dense(fut_pre.transform(span[future_features]))
    hn = list(hist_pre.get_feature_names_out())
    lgb_cols = [hn.index(c) for c in ("revenue__Revenue_Lag_364", "revenue__Revenue_Same_Date_Last_Year",
                                      "numerical__Quantity_Lag_364", "numerical__Quantity_Same_Date_Last_Year")]
    lgb_cols += [H.shape[1], H.shape[1] + 1]                     # + the two appended scaled targets

    fd = SimpleNamespace(
        name=name, origin=origin, is_test=is_test, span_df=span, tft_frame=None,
        products=sorted(span["Product"].unique()), lgb_cols=lgb_cols,
        n_train={}, hist={}, fut={}, pub={}, y_train={}, y_eval={}, stats={},
        eval_raw={}, eval_dates={}, eval_pub={}, snaive={}, mase_scale={})

    for p in fd.products:
        g = span[span["Product"] == p]
        ix = g.index.to_numpy()
        dates = pd.DatetimeIndex(g["Date"])
        n = int((dates < origin).sum())
        if len(g) - n != HORIZON:
            raise ValueError(f"{name}/{p}: expected {HORIZON} evaluation days, found {len(g) - n}")
        if n < LOOKBACK + HORIZON:
            raise ValueError(f"{name}/{p}: only {n} training days (< {LOOKBACK + HORIZON})")
        raw = g[["Net Price", "Quantity"]].to_numpy(float)
        pub = g["Is_Publication_Day"].to_numpy(np.float32)

        z = np.log1p(np.maximum(raw, 0))
        trp = pub[:n] == 1
        mu, sd = z[:n][trp].mean(0), z[:n][trp].std(0)           # statistics from training publication days
        sd = np.where(sd < 1e-8, 1.0, sd)
        zs = np.where(pub[:, None] == 1, (z - mu) / sd, 0.0).astype(np.float32)

        h = np.concatenate([H[ix], zs], axis=1)
        h[n:, -2:] = 0.0                                         # evaluation-period targets are never visible

        fd.n_train[p], fd.hist[p], fd.fut[p], fd.pub[p] = n, h, F[ix], pub
        fd.y_train[p] = zs[:n]
        fd.y_eval[p] = None if is_test else zs[n:]               # tuning curves only, never for test/production
        fd.stats[p] = (mu, sd)
        fd.eval_raw[p], fd.eval_dates[p], fd.eval_pub[p] = raw[n:], dates[n:], pub[n:]
        tr = pd.DataFrame(raw[:n], index=dates[:n], columns=TARGETS)
        fd.snaive[p] = seasonal_naive(tr, dates[n:], origin)
        fd.mase_scale[p] = mase_scale(tr, pub[:n])

    d0 = fd.eval_dates[fd.products[0]]
    fd.horizon_masks = {
        "First_Month":   np.asarray(d0 < origin + pd.DateOffset(months=1)),
        "First_Quarter": np.asarray(d0 < origin + pd.DateOffset(months=3)),
        "Full_Year":     np.ones(len(d0), dtype=bool)}
    return fd


def window_origins(fd, stride=STRIDE):
    """(product, forecast-origin index) pairs; the last window ends exactly at the fold's origin."""
    return [(p, i) for p in fd.products
            for i in range(fd.n_train[p] - HORIZON, LOOKBACK - 1, -stride)]


def gather(fd, wins):
    Hh = np.stack([fd.hist[p][i - LOOKBACK:i] for p, i in wins])
    Ff = np.stack([fd.fut[p][i:i + HORIZON] for p, i in wins])
    Yy = np.stack([fd.y_train[p][i:i + HORIZON] for p, i in wins])
    Ww = np.stack([fd.pub[p][i:i + HORIZON] for p, i in wins])
    return Hh, Ff, Yy, Ww


def to_quantiles(q_scaled, fd, p):
    """(365, 2, 3) standardised-log quantiles -> {'Revenue': (365,3), 'Quantity': (365,3)} in original units."""
    mu, sd = fd.stats[p]
    y = np.expm1(q_scaled * sd[None, :, None] + mu[None, :, None])
    y = np.sort(np.maximum(y, 0), axis=-1) * fd.eval_pub[p][:, None, None]   # non-publication days are known zeros
    return {"Revenue": y[:, 0, :], "Quantity": y[:, 1, :]}