from pathlib import Path

import numpy as np
import pandas as pd

from forecast.config import load_config
from forecast.constants import LOOKBACK, HORIZON
from forecast.folds import build_fold, window_origins, gather

cfg = load_config("configs/default.yaml")

df = pd.read_parquet(
    Path(cfg["paths"]["processed"]) / "model_df.parquet"
)

# Use the first validation fold
origin = pd.Timestamp(cfg["validation"]["cv_origins"][0])

fd = build_fold(
    df,
    origin=origin,
    name=origin.strftime("%Y-%m-%d"),
)

# Generate training windows
wins = window_origins(fd)

print("Number of training windows:", len(wins))

# Gather model inputs
H, F, Y, W = gather(fd, wins)

arrays = {
    "Historical inputs": H,
    "Future inputs": F,
    "Training targets": Y,
    "Window weights": W,
}

for name, arr in arrays.items():
    arr = np.asarray(arr)

    print(f"\n{name}")
    print("Shape:", arr.shape)
    print("NaN values:", np.isnan(arr).sum())
    print("Infinite values:", np.isinf(arr).sum())

    assert np.isfinite(arr).all(), (
        f"{name} contains NaN or infinite values"
    )

print("\nAll gathered model inputs passed.")