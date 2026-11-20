"""Temporal Fusion Transformer (pytorch-forecasting), one model per target."""
from pathlib import Path

import joblib
import lightning.pytorch as pl
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from ..constants import HORIZON, LOOKBACK, QUANTILES, SEED, STEPS_PER_EPOCH, TFT_BATCH
from ._torch import seed_all

TUNE_TARGETS = ("Revenue",)       # tune on revenue only; the same hyper-parameters are reused for quantity

KNOWN_CATS = ["Day_of_Week", "Month", "Quarter"]
KNOWN_REALS = ["time_idx", "Day_of_Month", "Year", "Week_of_Year", "Is_Weekend", "Is_Month_Start",
               "Is_Month_End", "Is_Publication_Day", "COVID_Intervention"]
LAG_COLS = ["Revenue_Lag_364", "Revenue_Same_Date_Last_Year", "Quantity_Lag_364", "Quantity_Same_Date_Last_Year"]


def ensure_tft_frame(fd):
    """One pandas frame per fold (train + evaluation rows) shared by both TFT targets."""
    if fd.tft_frame is not None:
        return fd.tft_frame
    d = fd.span_df.copy()
    d["Product"] = d["Product"].astype(str)
    d["time_idx"] = d.groupby("Product").cumcount()
    d["tft_weight"] = d["Is_Publication_Day"].astype("float32")      # loss weight: 0 on non-publication days
    for c in KNOWN_CATS:
        d[c] = d[c].astype(str)
    ev = (d["Date"] >= fd.origin).to_numpy()
    pub = d["Is_Publication_Day"].to_numpy() == 1
    for col, src, k in [("_log_revenue", "Net Price", 0), ("_log_quantity", "Quantity", 1)]:
        v = np.log1p(np.maximum(d[src].to_numpy(float), 0))
        fill = d["Product"].map({p: fd.stats[p][0][k] for p in fd.products}).to_numpy(float)
        v = np.where(pub, v, fill)                                   # neutral placeholder (weight 0)
        if fd.is_test:
            v = np.where(ev, fill, v)                                # test/production actuals never enter the dataset
        d[col] = v
    for c in LAG_COLS:
        v = np.log1p(np.maximum(d[c].to_numpy(float), 0))

        # Use training-period median for missing values
        training_median = np.nanmedian(v[~ev])

        v = np.where(np.isfinite(v), v, training_median)
        v = np.where(ev, training_median, v)      
        
        d[f"_log_{c}"] = v                 # neutral placeholder (weight 0)
    fd.tft_frame = d
    return d


class CurveCallback(pl.Callback):
    def __init__(self):
        self.curve = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking and "val_loss" in trainer.callback_metrics:
            self.curve.append(float(trainer.callback_metrics["val_loss"]))


def _fit_target(fd, target, params, epochs, track, save_dir=None):
    seed_all(); pl.seed_everything(SEED, workers=True)
    d = ensure_tft_frame(fd)
    tcol, ocol = ("_log_revenue", "_log_quantity") if target == "Revenue" else ("_log_quantity", "_log_revenue")
    ds = TimeSeriesDataSet(
        d[d["Date"] < fd.origin], time_idx="time_idx", target=tcol, group_ids=["Product"], weight="tft_weight",
        min_encoder_length=LOOKBACK, max_encoder_length=LOOKBACK,
        min_prediction_length=HORIZON, max_prediction_length=HORIZON,
        static_categoricals=["Product"], time_varying_known_categoricals=KNOWN_CATS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=[tcol, ocol] + [f"_log_{c}" for c in LAG_COLS],
        target_normalizer=GroupNormalizer(groups=["Product"]),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
        allow_missing_timesteps=False)
    train_loader = ds.to_dataloader(train=True, batch_size=TFT_BATCH, num_workers=0)
    val_loader = None
    if track:       # validation = the fold's forecast year (CV / validation folds only, never test or production)
        val_loader = TimeSeriesDataSet.from_dataset(ds, d, predict=True, stop_randomization=True)\
            .to_dataloader(train=False, batch_size=len(fd.products), num_workers=0)
    model = TemporalFusionTransformer.from_dataset(
        ds, learning_rate=params["lr"], hidden_size=params["hidden"], attention_head_size=params["heads"],
        dropout=params["dropout"], hidden_continuous_size=params["hidden_cont"], output_size=3,
        loss=QuantileLoss(quantiles=[float(q) for q in QUANTILES]), reduce_on_plateau_patience=3,
        log_interval=-1, log_val_interval=-1)
    cb = CurveCallback()
    trainer = pl.Trainer(
        max_epochs=epochs, accelerator="auto", devices=1, gradient_clip_val=0.1,
        limit_train_batches=min(STEPS_PER_EPOCH, len(train_loader)),    # fixed "epoch" length across folds
        num_sanity_val_steps=0, enable_checkpointing=False, enable_model_summary=False,
        enable_progress_bar=False, logger=False, callbacks=[cb])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if save_dir is not None:                                            # reload with TemporalFusionTransformer.load_from_checkpoint
        d_ = Path(save_dir); d_.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(d_ / f"tft_{target.lower()}.ckpt"))
        joblib.dump(ds.get_parameters(), d_ / f"tft_{target.lower()}_dataset.joblib")
    return model, ds, cb.curve


def fit(fd, params, steps, track=True, targets=("Revenue", "Quantity"), save_dir=None):
    art, curves = {}, []
    for t in targets:
        model, ds, curve = _fit_target(fd, t, params, steps, track, save_dir)
        art[t] = (model, ds)
        curves.append(curve)
    return art, (np.mean(curves, axis=0) if track else [])


def _unpack(res):
    return (res.output, res.index) if hasattr(res, "output") else (res[0], res[1])


def predict_loader(fd, ds):
    return TimeSeriesDataSet.from_dataset(ds, ensure_tft_frame(fd), predict=True, stop_randomization=True)\
        .to_dataloader(train=False, batch_size=len(fd.products), num_workers=0)


def predict(art, fd, steps=None):
    out = {p: {} for p in fd.products}
    for t, (model, ds) in art.items():
        q, index = _unpack(model.predict(predict_loader(fd, ds), mode="quantiles", return_index=True))
        q = q.detach().cpu().numpy()                                  # (products, 365, 3), log1p units
        for row, p in enumerate(index["Product"].astype(str)):
            out[p][t] = np.sort(np.maximum(np.expm1(q[row]), 0), axis=1) * fd.eval_pub[p][:, None]
    return out


def save(art, directory, params=None):
    """Checkpoints are written inside fit(save_dir=...) because Lightning needs the trainer."""
    return None


DEFAULT = dict(hidden=16, heads=4, dropout=0.1, hidden_cont=8, lr=1e-3)


def sample(rng):
    return dict(hidden=int(rng.choice([16, 32, 64])), heads=int(rng.choice([2, 4])),
                dropout=float(rng.choice([0.1, 0.2, 0.3])), hidden_cont=int(rng.choice([8, 16])),
                lr=float(np.exp(rng.uniform(np.log(3e-4), np.log(3e-3)))))