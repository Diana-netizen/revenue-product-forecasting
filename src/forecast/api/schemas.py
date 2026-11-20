
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    product: str = Field(
        ...,
        description="Product or publication edition"
    )
    target: Literal["Revenue", "Quantity"] = "Revenue"

    model: Literal["lgbm", "gru", "tft"] = "lgbm"

    horizon: int = Field(
        default=365,
        ge=1,
        le=365
    )

    forecast_origin: date


class ForecastPoint(BaseModel):
    date: date

    lower: float
    median: float
    upper: float


class ForecastResponse(BaseModel):
    product: str
    model: str
    forecast_origin: date
    horizon: int

    forecasts: list[ForecastPoint]


class HealthResponse(BaseModel):
    status: str
    message: str
