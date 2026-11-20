
"""Forecast API service using saved production forecasts."""

from functools import lru_cache
from pathlib import Path

import pandas as pd
import os

from forecast.api.schemas import (
    ForecastRequest,
    ForecastPoint,
    ForecastResponse,
)

# PROJECT_ROOT = Path(__file__).resolve().parents[3]   #for local deployment

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/opt/render/project/src"))  # For github deployment
FORECAST_FILE = PROJECT_ROOT / "reports" / "production_forecast.csv"


REQUIRED_COLUMNS = {
    "Date",
    "Product",
    "Target",
    "q10",
    "q50",
    "q90",
}


@lru_cache(maxsize=1)
def load_production_forecasts():
    """Load and cache the generated production forecasts."""

    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"Production forecast file not found: {FORECAST_FILE}"
        )

    df = pd.read_csv(FORECAST_FILE)

    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            f"Production forecast file is missing columns: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    return df


def generate_forecast(
    request: ForecastRequest
) -> ForecastResponse:
    """Retrieve the requested production forecast."""

    # Only LightGBM has been trained and saved for production.
    if request.model != "lgbm":
        raise ValueError(
            "Only LightGBM forecasts are currently available. "
            "GRU and TFT production models have not been generated."
        )

    df = load_production_forecasts()

    # Filter by product and forecasting target.
    selected = df[
        (df["Product"].astype(str) == request.product)
        & (df["Target"] == request.target)
    ].copy()

    if selected.empty:
        available_products = sorted(
            df["Product"].astype(str).unique().tolist()
        )

        raise ValueError(
            f"No {request.target} forecasts found for "
            f"'{request.product}'. Available products: "
            f"{available_products}"
        )

    selected = selected.sort_values("Date")

    # Ensure the requested origin exists in the production forecast.
    available_origin = selected["Date"].min()

    if request.forecast_origin != available_origin:
        raise ValueError(
            f"Forecast origin {request.forecast_origin} is unavailable. "
            f"The available production forecast begins on "
            f"{available_origin}."
        )

    # Select the requested forecast horizon.
    selected = selected[
        selected["Date"] >= request.forecast_origin
    ].head(request.horizon)

    if len(selected) < request.horizon:
        raise ValueError(
            f"Requested horizon of {request.horizon} days exceeds "
            f"the available production forecast of {len(selected)} days."
        )

    # Convert quantile forecasts into API response points.
    forecasts = [
        ForecastPoint(
            date=row.Date,
            lower=float(row.q10),
            median=float(row.q50),
            upper=float(row.q90),
        )
        for row in selected.itertuples(index=False)
    ]

    return ForecastResponse(
        product=request.product,
        model="LightGBM",
        forecast_origin=request.forecast_origin,
        horizon=len(forecasts),
        forecasts=forecasts,
    )
