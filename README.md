# Revenue Product Forecasting System

A daily, multi-horizon probabilistic forecasting system for revenue and quantity for 4 products. It compares machine-learning and deep-learning approaches and provides a pipeline for data preparation, model evaluation, and forecast generation.

## 1. Project objectives

- Develop a repeatable daily forecasting pipeline for revenue and quantity.
- Compare LightGBM, GRU, and Temporal Fusion Transformer (TFT) models.
- Tune model hyperparameters and evaluate performance using time-series validation.
- Compare model forecasts with a seasonal-naive benchmark.
- Generate probabilistic forecasts at the 10th, 50th, and 90th percentiles (corresponding to lower, median and upper forecast estimates).
-The configured forecast horizon is up to 365 days.


## 2. Methodology and project workflow

The project follows a pipeline-based forecasting workflow:

1. **Data preparation:** load and clean source data, aggregate observations daily, align publication dates, create a continuous calendar, and handle leap-day observations.
2. **Feature engineering:** construct historical lag and seasonal features, together with known calendar and product-related features.
3. **Model development:** train LightGBM, GRU, and TFT models for the forecasting targets.
4. **Hyperparameter tuning:** evaluate candidate configurations using the configured validation and cross-validation procedure. The tuning implementation uses successive halving.
5. **Backtesting and selection:** compare models across validation forecast origins. The project documentation specifies mean full-year MASE as the model-selection criterion.
6. **Final testing:** evaluate the selected approach on the independent test origin. Test results are not used for tuning or model selection.
7. **Production forecasting:** train the selected approach on available historical data and generate the configured production forecast.

### 3. Evaluation metrics

| Metric | Purpose |
|---|---|
| MAE | Average absolute forecast error |
| MASE | Forecast error scaled against a benchmark |
| sMAPE | Symmetric percentage error |
| Pinball loss | Quantile forecast accuracy |
| Scaled pinball loss | Scaled probabilistic forecast comparison |
| MAE versus seasonal naive | Comparison with the benchmark |

The exact forecast origins, folds, and experiment settings are controlled through `configs/default.yaml`.


## 4. Repository structure

```text
revenue-product-forecasting/
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── artifacts/
│   ├── tuning/
│   ├── models/
│   └── production/
├── reports/
│   ├── figures/
│   ├── interpretation/
│   └── tables/
├── src/
│   └── forecast/
│       ├── api/
│       ├── models/
│       ├── config.py
│       ├── constants.py
│       ├── features.py
│       ├── folds.py
│       ├── ingest.py
│       ├── interpret.py
│       ├── metrics.py
│       ├── run.py
│       └── tune.py
├── frontend/
├── tests/
├── Dockerfile
├── Dockerfile.pipeline
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── requirements-serve.txt
```

## 5. Technology stack

- **Data processing:** Python, pandas, NumPy, PyArrow
- **Modelling:** LightGBM, scikit-learn, PyTorch, PyTorch Forecasting
- **Visualisation:** Matplotlib, Seaborn, Apache ECharts
- **Backend:** FastAPI, Uvicorn
- **Frontend:** React, Vite, Tailwind CSS, ECharts
- **Deployment:** Docker, Docker Compose, Nginx

## 6. Setup and execution

### Requirements

- Python 3.12
- Git
- Node.js and npm (for the frontend)
- Docker Desktop (for containerised execution)

### Clone the repository

```bash
git clone https://github.com/Diana-netizen/revenue-product-forecasting.git
cd revenue-product-forecasting
```

### Set up Python

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-serve.txt
```


### Run pipeline stages

Run from the project root:

```powershell
python -m forecast.run ingest --config configs/default.yaml
python -m forecast.run tune --config configs/default.yaml
python -m forecast.run backtest --config configs/default.yaml
python -m forecast.run test --config configs/default.yaml
python -m forecast.run production --config configs/default.yaml
```

To run all stages:

```powershell
python -m forecast.run all --config configs/default.yaml
```

Check available commands with:

```powershell
python -m forecast.run --help
```

### Run the application

Start the backend:

```powershell
uvicorn forecast.api.main:app --reload --port 8000
```

The documented API endpoints are:

- Health check: `http://localhost:8000/health`
- API documentation: `http://localhost:8000/docs`

If the frontend is available, install and run it in a separate terminal:

```powershell
cd frontend
npm install
npm run dev
```

Docker configuration:

```powershell
docker compose up --build -d
docker compose ps
docker compose down
```


## 8. Testing and reproducibility

The `tests/` directory contains tests and diagnostic scripts covering ingestion, feature engineering, fold construction, model inputs, tuning setup, predictions, and metrics.

Run the pytest suite with:

```powershell
pytest
```

## 9. Limitations and data considerations

- Forecast performance depends on the quality and coverage of historical data.
- Deep-learning model training may require substantial computing resources.
- Probabilistic forecasts represent uncertainty and are not guarantees.
- The independent test set must remain separate from model tuning and selection.
- Data access and publication must comply with applicable organisational and institutional requirements.

