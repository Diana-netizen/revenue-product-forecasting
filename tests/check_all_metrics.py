
from pathlib import Path

import pandas as pd
import numpy as np

from forecast.config import load_config
from forecast.constants import TARGETS
from forecast.folds import build_fold
from forecast.metrics import evaluate, summarise, report_table

# --------------------------------------------------
# 1. Load configuration and evaluation fold
# --------------------------------------------------

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

output_dir = Path("artifacts/first_fold")
output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load saved forecasts
# --------------------------------------------------

models = [
    "lightgbm",
    "gru",
    "tft",
    "seasonal_naive",
]

all_results = []

for model in models:

    filepath = output_dir / f"{model}_forecasts.npz"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {filepath}"
        )

    print(f"\nEvaluating {model}")

    predictions = {}

    with np.load(filepath, allow_pickle=False) as saved:
        for product in fd.products:
            predictions[product] = {}

            for target in TARGETS:
                key = f"{product}__{target}"
                predictions[product][target] = saved[key]

    # --------------------------------------------------
    # 3. Calculate evaluation metrics
    # --------------------------------------------------

    model_results = evaluate(
        fd,
        predictions,
        model=model.replace("_", " ").title(),
        phase="First_Fold",
    )

    all_results.append(model_results)

# --------------------------------------------------
# 4. Combine results
# --------------------------------------------------

results_df = pd.concat(all_results, ignore_index=True)

results_df.to_csv(
    output_dir / "first_fold_metrics.csv",
    index=False,
)

# --------------------------------------------------
# 5. Summarise results
# --------------------------------------------------

summary_df = summarise(results_df)

summary_df.to_csv(
    output_dir / "first_fold_summary.csv",
    index=False,
)

# --------------------------------------------------
# 6. Generate comparison table
# --------------------------------------------------

comparison_df = report_table(results_df)

comparison_df.to_csv(
    output_dir / "first_fold_comparison.csv",
    index=False,
)

print("\nFIRST-FOLD MODEL COMPARISON")
print(comparison_df.to_string(index=False))

print("\nAll evaluation results saved successfully.")
