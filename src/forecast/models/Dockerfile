
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml .
COPY src/ ./src/
COPY configs/ ./configs/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "forecast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
