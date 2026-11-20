
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException

from forecast.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
)
from forecast.api.service import generate_forecast


# Works in both local development and Docker
# ROOT_DIR = Path.cwd()    # For local deployment
ROOT_DIR = Path(os.environ.get("PROJECT_ROOT", "/opt/render/project/src"))   # for github
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

app = FastAPI(
    title="Revenue Forecasting API",
    description="API for multi-horizon probabilistic revenue forecasting using LightGBM.",
    version="1.0.0",
)


@app.get("/", response_model=HealthResponse)
def home():
    return HealthResponse(
        status="online",
        message="Revenue Forecasting API is running.",
    )


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        message="API is operational.",
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    try:
        return generate_forecast(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/historical")
def historical_revenue(
    product: str | None = None,
    target: str = "Revenue"
):
    data_path = DATA_DIR / "processed" / "model_df.parquet"

    if not data_path.exists():
        raise HTTPException(404, f"Dataset not found: {data_path}")

    df = pd.read_parquet(data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    column = "Net Price" if target == "Revenue" else "Quantity"

    if column not in df.columns:
        raise HTTPException(400, f"Column {column} not found")

    if product:
        df = df[df["Product"] == product]

    df = df.dropna(subset=[column]).sort_values("Date")

    return {
        "target": target,
        "data": [
            {
                "date": row.Date.strftime("%Y-%m-%d"),
                "product": row.Product,
                "value": float(getattr(row, column))
            }
            for row in df.itertuples(index=False)
        ]
    }


@app.get("/accuracy")
def accuracy():
    report_path = REPORTS_DIR / "test_summary.csv"

    if not report_path.exists():
        raise HTTPException(404, f"Test summary not found: {report_path}")

    df = pd.read_csv(report_path)

    return {"results": df.to_dict(orient="records")}


@app.get("/model-comparison")
def model_comparison():
    report_path = REPORTS_DIR / "test_summary.csv"

    if not report_path.exists():
        raise HTTPException(404, f"Test summary not found: {report_path}")

    df = pd.read_csv(report_path)

    return {
        "models": df.to_dict(orient="records"),
        "production_model": "LightGBM"
    }
