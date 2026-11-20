
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.config import load_config
from forecast.constants import HORIZON, QUANTILES, TARGETS
from forecast.folds import build_fold
from forecast.metrics import snaive_forecasts
from forecast.models import get_registry


cfg = load_config("configs/default.yaml")

df = pd.read_parquet(
    Path(cfg["paths"]["processed"]) / "model_df.parquet"
)

origin = pd.Timestamp(cfg["validation"]["cv_origins"][0])

fd = build_fold(
    df,
    origin=origin,
    name=origin.strftime("%Y-%m-%d"),
)

registry = get_registry(cfg)

# Use default parameters and one training run for each model.
model_predictions = {}

for name, spec in registry.items():    
    print(f"\nTraining {name}")
    print("Parameters:", spec["default"])
    print("Maximum steps:", spec["max_steps"])

    art, _ = spec["fit"](
        fd,
        params=spec["default"],
        steps=spec["max_steps"],
        track=False,
        targets=TARGETS,
    )

    print(f"Generating {name} forecasts...")

    model_predictions[name] = spec["predict"](art, fd, steps=spec["max_steps"])

    print(f"{name} training and prediction completed.")


# Seasonal Naive baseline
model_predictions["Seasonal Naive"] = snaive_forecasts(fd)


# Validate all prediction outputs
for model, predictions in model_predictions.items():

    print(f"\n{'=' * 50}")
    print(model)

    for product in fd.products:
        for target in TARGETS:

            q = predictions[product][target]

            print(
                f"{product} | {target} | "
                f"Shape: {q.shape}"
            )

            assert q.shape == (HORIZON, len(QUANTILES)), (
                f"{model}: incorrect shape for {product}, {target}"
            )

            assert np.isfinite(q).all(), (
                f"{model}: NaN or infinite values for {product}, {target}"
            )

            assert (q >= 0).all(), (
                f"{model}: negative forecasts for {product}, {target}"
            )

            assert len(fd.eval_dates[product]) == HORIZON

print("\nAll model prediction checks passed.")

# Save forecasts for subsequent evaluation
output_dir = Path("artifacts/first_fold")
output_dir.mkdir(parents=True, exist_ok=True)

for model, predictions in model_predictions.items():

    forecast_data = {}

    for product in fd.products:
        for target in TARGETS:
            forecast_data[f"{product}__{target}"] = (
                predictions[product][target]
            )

    filename = f"{model.lower().replace(' ', '_')}_forecasts.npz"

    np.savez_compressed(
        output_dir / filename,
        **forecast_data
    )

    print(f"Saved {model} forecasts.")
