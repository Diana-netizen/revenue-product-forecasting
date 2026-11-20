
from pathlib import Path

import pandas as pd

from forecast.config import load_config
from forecast.constants import LOOKBACK, HORIZON
from forecast.folds import build_fold

cfg = load_config("configs/default.yaml")

df = pd.read_parquet(
    Path(cfg["paths"]["processed"]) / "model_df.parquet"
)

origins = cfg["validation"]["cv_origins"]

print("Dataset shape:", df.shape)
print("Products:", sorted(df["Product"].unique()))
print("Lookback:", LOOKBACK)
print("Forecast horizon:", HORIZON)

print("\nFOLD DIAGNOSTICS")

for origin in origins:
    origin = pd.Timestamp(origin)

    try:
        fd = build_fold(
            df,
            origin=origin,
            name=origin.strftime("%Y-%m-%d"),
        )

        print(f"\nFold: {origin.date()}")
        print("Training observations:", fd.n_train)
        print("Evaluation dates per product:")

        for product, dates in fd.eval_dates.items():
            print(f"{product}: {len(dates)}")

            assert len(dates) == HORIZON, (
                f"{product}: expected {HORIZON} evaluation dates, "
                f"got {len(dates)}"
            )

        for product, n in fd.n_train.items():
            assert n >= LOOKBACK + HORIZON, (
                f"{product} has insufficient training observations"
            )

        print("Fold passed.")

    except Exception as e:
        print(f"Fold failed: {type(e).__name__}: {e}")
