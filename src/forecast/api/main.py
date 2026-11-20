import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException

from forecast.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
)

from forecast.api.service import generate_forecast


app = FastAPI(
    title="Revenue Forecasting API",
    description=(
        "API for multi-horizon probabilistic revenue forecasting "
        "using LightGBM."
    ),
    version="1.0.0",
)


@app.get(
    "/",
    response_model=HealthResponse
)
def home():
    return HealthResponse(
        status="online",
        message="Revenue Forecasting API is running.",
    )


@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check():
    return HealthResponse(
        status="healthy",
        message="API is operational.",
    )


@app.post(
    "/forecast",
    response_model=ForecastResponse
)
def forecast(request: ForecastRequest):

    try:
        return generate_forecast(request)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {str(exc)}",
        )

@app.get("/historical")
def historical_revenue(
    product: str | None = None,
    target: str = "Revenue"
):
    try:
        data_path = Path("/app/data/processed/model_df.parquet")

        if not data_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Historical dataset not found: {data_path}"
            )

        #df = pd.read_csv(data_path, parse_dates=["Date"])
        df = pd.read_parquet(data_path)
        df["Date"] = pd.to_datetime(df["Date"])

        if product:
            df = df[df["Product"] == product]

        if target == "Revenue":
            value_column = "Net Price"
        elif target == "Quantity":
            value_column = "Quantity"
        else:
            raise HTTPException(
                status_code=400,
                detail="Target must be Revenue or Quantity"
            )

        df = df.dropna(subset=[value_column])

        return {
            "target": target,
            "data": [
                {
                    "date": row["Date"].strftime("%Y-%m-%d"),
                    "product": row["Product"],
                    "value": float(row[value_column])
                }
                for _, row in df.iterrows()
            ]
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )
