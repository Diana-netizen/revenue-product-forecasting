
from pathlib import Path

import pandas as pd

from forecast.config import load_config
from forecast.constants import TARGETS
from forecast.folds import build_fold
from forecast.models import get_registry


def main():
    cfg = load_config()

    data_path = Path(cfg["paths"]["processed"]) / "model_df.parquet"
    df = pd.read_parquet(data_path)

    cv_origins = cfg["validation"]["cv_origins"]
    val_names = cfg["validation"]["validation_origins"]

    registry = get_registry(cfg)

    print("\nCONFIGURATION")
    print("CV origins:", cv_origins)
    print("Validation origins:", val_names)
    print("Models:", list(registry))

    print("\nMODEL TUNING SETTINGS")
    for name, spec in registry.items():
        print(
            name,
            {
                "n_configs": spec["n_configs"],
                "max_steps": spec["max_steps"],
                "tune_targets": spec["tune_targets"],
            },
        )

    print("\nFOLD VALIDATION")

    fold_names = []

    for origin in cv_origins:
        name = pd.Timestamp(origin).strftime("%Y-%m-%d")

        fd = build_fold(
            df,
            origin=pd.Timestamp(origin),
            name=name,
        )

        fold_names.append(fd.name)

        eval_days = sum(
            len(dates) for dates in fd.eval_dates.values()
        )

        print(
            f"{fd.name}: "
            f"products={len(fd.eval_dates)}, "
            f"evaluation_days={eval_days}"
        )

    print("\nCHECKS")

    assert len(fold_names) == 5, "Expected exactly five CV folds."

    missing_validation = set(val_names) - set(fold_names)
    assert not missing_validation, (
        f"Validation folds not found: {missing_validation}"
    )

    assert all(
        name in fold_names for name in val_names
    ), "Validation fold names do not match CV fold names."

    print("PASS: Five folds available.")
    print("PASS: Validation folds match.")
    print("PASS: Model tuning settings loaded correctly.")


if __name__ == "__main__":
    main()
