# Revenue Forecasting System

## 1. Project Overview

This project develops a machine learning-based revenue forecasting system for the media industry. It uses historical revenue and quantity data to generate daily probabilistic forecasts across multiple media products.

The system is designed to forecast up to 365 days into the future, providing three prediction intervals:

- **P10:** Lower forecast estimate.
- **P50:** Median forecast estimate.
- **P90:** Upper forecast estimate.

The project combines data engineering, feature engineering, machine learning, time-series validation, probabilistic forecasting, and an interactive dashboard.

### Key Objectives

1. Develop a reliable daily forecasting pipeline for revenue and quantity.
2. Compare different machine learning and deep learning forecasting models.
3. Optimise model hyperparameters using cross-validation.
4. Evaluate forecasting accuracy against a seasonal naive benchmark.
5. Generate probabilistic forecasts to represent forecast uncertainty.
6. Deploy the forecasting application through a web-based dashboard and API.

---

## 2. Models

The project implements three forecasting approaches:

| Model | Description |
|---|---|
| LightGBM | Gradient boosting framework for structured forecasting data. |
| GRU | Gated Recurrent Unit neural network for sequential data modelling. |
| TFT | Temporal Fusion Transformer for multi-horizon time-series forecasting. |

Each model is evaluated on two forecasting targets:

- Revenue
- Quantity

The models generate forecasts at the 10th, 50th, and 90th percentiles.

---

## 3. Project Structure

```text
capstone/
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── model_df.parquet
│   │   └── model_df_production.parquet
│   └── sample/
│
├── artifacts/
│   ├── tuning/
│   ├── models/
│   └── production/
│
├── reports/
│
├── src/
│   └── forecast/
│       ├── __init__.py
│       ├── run.py
│       ├── constants.py
│       ├── ingest.py
│       ├── features.py
│       ├── folds.py
│       ├── tune.py
│       ├── metrics.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── lightgbm.py
│       │   ├── gru.py
│       │   └── tft.py
│       │
│       └── api/
│           └── main.py
│
├── frontend/
│   ├── src/
│   ├── package.json
│   ├── Dockerfile
│   └── nginx.conf
│
├── notebooks/
│
├── Dockerfile
├── Dockerfile.pipeline
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
├── requirements-serve.txt
├── README.md
└── .gitignore
```

*Note: The directory tree is a representation of the project structure. Update it if files are added, removed, or renamed.*

---

## 4. Technology Stack

### Data Processing
- Python
- Pandas
- NumPy
- PyArrow

### Machine Learning
- LightGBM
- Scikit-learn
- PyTorch
- PyTorch Forecasting

### Data Visualisation
- Matplotlib
- Seaborn
- Apache ECharts

### Backend
- FastAPI
- Uvicorn

### Frontend
- React
- Vite
- Tailwind CSS
- ECharts

### Deployment
- Docker
- Docker Compose
- Nginx

---

## 5. Installation and Setup

### Prerequisites

Install the following:

- Python 3.12
- Node.js and npm
- Git
- Docker Desktop (for containerised deployment)

### Clone the Repository

```bash
git clone <repository-url>
cd capstone
```

### Create a Python Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install Python Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

For API development:

```powershell
pip install -r requirements-serve.txt
```

### Install Frontend Dependencies

```powershell
cd frontend
npm install
cd ..
```

---

## 6. Configuration

The project configuration is maintained in:

```text
configs/default.yaml
```

This file defines the forecasting and model-training parameters, including:

- Cross-validation forecast origins.
- Validation forecast origins.
- Final test forecast origin.
- Hyperparameter tuning budgets.
- Model training steps.
- Production forecasting period.

The main forecasting constants are defined in `src/forecast/constants.py`.

Current forecasting configuration includes:

| Parameter | Value |
|---|---:|
| Lookback window | 90 days |
| Forecast horizon | 365 days |
| Forecast quantiles | 0.10, 0.50, 0.90 |
| Seasonal lag | 364 days |
| Cross-validation stride | 7 days |
| Random seed | 42 |

---

## 7. Data Pipeline

The data pipeline transforms the raw data into a structured forecasting dataset.

### Data Ingestion

Implemented in:

```text
src/forecast/ingest.py
```

The ingestion process includes:

1. Loading raw revenue and quantity data.
2. Cleaning and preparing the source data.
3. Aggregating observations at the daily level.
4. Aligning observations to publication dates.
5. Creating a continuous daily calendar.
6. Generating calendar and lag features.
7. Handling leap-day observations.

Processed datasets are saved in Parquet format:

```text
data/processed/model_df.parquet
data/processed/model_df_production.parquet
```

### Feature Engineering

Implemented in:

```text
src/forecast/features.py
```

Features are constructed using historical observations and known future calendar information.

Historical features include lagged revenue and quantity values, including seasonal lag features.

Future-known features include calendar and product-related information.

Feature preprocessing is fitted using historical training data to prevent information leakage.

---

## 8. Model Training and Evaluation

### Hyperparameter Tuning

Implemented in:

```text
src/forecast/tune.py
```

The tuning process uses successive halving to allocate computational resources to promising configurations.

The process includes:

1. Generating candidate hyperparameter configurations.
2. Evaluating configurations on validation folds.
3. Retaining the strongest configurations.
4. Evaluating surviving configurations across the cross-validation folds.
5. Selecting the best configuration for each model.

Tuning checkpoints are saved to support recovery after interrupted runs.

Execute:

```powershell
python -m forecast.run tune --config configs/default.yaml
```

### Backtesting

Backtesting evaluates the tuned models on the designated validation forecast origins.

Execute:

```powershell
python -m forecast.run backtest --config configs/default.yaml
```

The backtesting process compares model performance across the validation folds and selects the best-performing model using mean full-year MASE.

### Final Testing

The final test evaluates the selected forecasting approach on the independent test forecast origin.

Execute:

```powershell
python -m forecast.run test --config configs/default.yaml
```

The test dataset is kept separate from hyperparameter tuning and model selection.

### Production Forecasting

The production stage trains the selected model using the available historical data and generates forecasts for the configured production period.

Execute:

```powershell
python -m forecast.run production --config configs/default.yaml
```

### Run the Complete Pipeline

To execute the pipeline stages sequentially:

```powershell
python -m forecast.run all --config configs/default.yaml
```

For individual stages, use:

```text
ingest
tune
backtest
test
production
```

---

## 9. Forecast Evaluation Metrics

The project uses multiple metrics to assess forecasting performance.

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute forecast error. |
| MASE | Measures forecast error relative to a scaled benchmark. |
| sMAPE | Measures symmetric percentage error. |
| Pinball Loss | Evaluates quantile forecast accuracy. |
| Scaled Pinball Loss | Enables comparison of probabilistic forecasting performance across targets. |
| MAE vs Seasonal Naive | Compares model MAE against the seasonal naive benchmark. |

The seasonal naive model provides a benchmark against which the machine learning models are evaluated.

---

## 10. API and Dashboard

The project includes a FastAPI backend and a React-based frontend.

### Backend

The backend is located in:

```text
src/forecast/api/
```

It provides the API layer for the forecasting application.

The health endpoint can be accessed at:

```text
http://localhost:8000/health
```

The API documentation is available at:

```text
http://localhost:8000/docs
```

**Implementation status:** The health endpoint is available. Actual model inference through the forecast endpoint must be connected to the saved production model before the dashboard can return genuine forecasts.

### Frontend

The frontend is located in:

```text
frontend/
```

It provides an interactive forecasting dashboard with controls for:

- Product selection.
- Forecasting model.
- Forecast target.
- Forecast horizon.
- Forecast origin.

The dashboard is designed to display forecast outputs and visualise the predicted values.

Demo chart values should not be interpreted as actual model forecasts.

---

## 11. Running the Application Locally

### Start the Backend

From the project root:

```powershell
uvicorn forecast.api.main:app --reload --port 8000
```

### Start the Frontend

Open another terminal:

```powershell
cd frontend
npm run dev
```

The Vite development server will display the local frontend address in the terminal.

---

## 12. Docker Deployment

The project includes Docker configurations for the backend and frontend.

Build and start the containers:

```powershell
docker compose up --build -d
```

Check running containers:

```powershell
docker compose ps
```

View logs:

```powershell
docker compose logs -f
```

Stop the application:

```powershell
docker compose down
```

The expected application endpoints are:

| Component | Address |
|---|---|
| Frontend | http://localhost |
| Backend | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

The production model and its required artefacts must be generated before the application can serve actual forecasts.

---

## 13. Reproducibility

The project uses a fixed random seed:

```python
SEED = 42
```

This supports reproducibility across model training and evaluation.

The configuration file controls the training and evaluation settings. Tuning checkpoints allow completed fold evaluations to be reused when a training run is interrupted, provided the configuration remains compatible.

---

## 14. Current Project Status

The project is being developed in stages.

| Component | Status |
|---|---|
| Data ingestion and processing | Implemented |
| Feature engineering | Implemented |
| LightGBM model | Implemented |
| GRU model | Implemented |
| TFT model | Implemented |
| Hyperparameter tuning | Implemented |
| Resumable tuning checkpoints | Implemented |
| Validation backtesting | In progress |
| Independent final testing | Pending |
| Production model generation | Pending |
| FastAPI health endpoint | Implemented |
| Production model inference API | Pending |
| Frontend dashboard | Implemented |
| Docker configuration | Prepared |

Update this section as each stage is completed.

---

## 15. Limitations

The following considerations apply:

- Deep learning model training can be computationally expensive, particularly when using CPU-only hardware.
- Forecast accuracy depends on historical data quality and the availability of relevant explanatory features.
- Probabilistic forecasts represent uncertainty and should not be interpreted as guaranteed revenue outcomes.
- The final model should only be selected after validation backtesting.
- The independent test results should not be used to tune or select models.
- The dashboard should only be used for operational forecasting once the production inference endpoint has been connected and tested.

---

## 16. Author

**Project:** Daily Multi-Horizon Probabilistic Revenue Forecasting System

**Author:** Diana Otieno

**Programme:** MSc Data Science and Analytics

**Institution:** Strathmore University

**Application Domain:** Media Industry

---

## 17. Licence

This project was developed for academic and analytical purposes. Usage and distribution are subject to the applicable institutional and organisational data policies.