"""LightGBM quantile models: per-iteration validation curves, shared binning, publication-day rows only."""
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ..constants import HORIZON, LOOKBACK, QUANTILES, ROW_STRIDE, SEED, TARGETS
from ..folds import to_quantiles, window_origins

TUNE_TARGETS = TARGETS
_LEAD = (np.arange(HORIZON) / (HORIZON - 1)).astype(np.float32)[:, None]


def _block(fd, p, i):
    h = fd.hist[p][i - LOOKBACK:i][:, fd.lgb_cols]               # lags + scaled targets over the 90-day lookback
    s = np.concatenate([h[-1], h[-7:].mean(0), h[-28:].mean(0), h.mean(0), h.std(0)])
    return np.hstack([np.broadcast_to(s, (HORIZON, s.size)), fd.fut[p][i:i + HORIZON], _LEAD])


def _train_matrix(fd):
    X, Y = [], []
    for p, i in window_origins(fd):
        keep = np.flatnonzero(fd.pub[p][i:i + HORIZON] == 1)[::ROW_STRIDE]
        X.append(_block(fd, p, i)[keep])
        Y.append(fd.y_train[p][i:i + HORIZON][keep])
    return np.vstack(X).astype(np.float32), np.vstack(Y)


def _eval_matrix(fd, p):
    return _block(fd, p, fd.n_train[p]).astype(np.float32)


def fit(fd, params, steps, track=True, targets=None, save_dir=None):
    Xtr, Ytr = _train_matrix(fd)
    if track:
        keeps = {p: fd.eval_pub[p] == 1 for p in fd.products}
        Xv = np.vstack([_eval_matrix(fd, p)[keeps[p]] for p in fd.products])
        Yv = np.vstack([fd.y_eval[p][keeps[p]] for p in fd.products])
    boosters, curves = {}, []
    for k, t in enumerate(TARGETS):
        dtr = lgb.Dataset(Xtr, label=Ytr[:, k], params={"max_bin": 63}, free_raw_data=False)  # binned once, reused x3
        dva = lgb.Dataset(Xv, label=Yv[:, k], reference=dtr) if track else None
        for q in QUANTILES:
            rec = {}
            P = dict(objective="quantile", alpha=float(q), metric="quantile", verbosity=-1, seed=SEED,
                     num_threads=os.cpu_count(), bagging_fraction=0.8, bagging_freq=1, max_bin=63,
                     learning_rate=params["learning_rate"], num_leaves=params["num_leaves"],
                     min_data_in_leaf=params["min_data_in_leaf"], feature_fraction=params["feature_fraction"],
                     lambda_l2=params["lambda_l2"])
            boosters[(t, float(q))] = lgb.train(
                P, dtr, num_boost_round=steps,
                valid_sets=[dva] if track else None, valid_names=["val"] if track else None,
                callbacks=[lgb.record_evaluation(rec)] if track else None)
            if track:
                curves.append(np.asarray(rec["val"]["quantile"]))
    return boosters, (np.mean(curves, axis=0) if track else [])


def predict(boosters, fd, steps):
    out = {}
    for p in fd.products:
        X = _eval_matrix(fd, p)
        q = np.stack([np.column_stack([boosters[(t, float(a))].predict(X, num_iteration=steps) for a in QUANTILES])
                      for t in TARGETS], axis=1)                  # (365, 2, 3)
        out[p] = to_quantiles(q, fd, p)
    return out


def save(boosters, directory, params=None):
    d = Path(directory); d.mkdir(parents=True, exist_ok=True)
    for (t, q), b in boosters.items():
        b.save_model(str(d / f"lgbm_{t.lower()}_q{int(round(q * 100))}.txt"))


DEFAULT = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=20, feature_fraction=0.8, lambda_l2=0.0)


def sample(rng):
    return dict(learning_rate=float(np.exp(rng.uniform(np.log(0.02), np.log(0.10)))),
                num_leaves=int(rng.choice([15, 31, 63])),
                min_data_in_leaf=int(rng.choice([20, 50, 100, 200])),
                feature_fraction=float(rng.uniform(0.5, 1.0)),
                lambda_l2=float(np.exp(rng.uniform(np.log(0.01), np.log(10)))))