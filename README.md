# Self-Healing ML Pipeline (Auto Debugging AI System)

An end-to-end MLOps project that detects model failures (data drift and performance drop), retrains automatically, versions new models, and serves the latest model through an API with monitoring and visualization.

## What This Project Demonstrates

- Automated tabular data ingestion and preprocessing
- Model training with experiment tracking support (MLflow)
- Data drift detection using Population Stability Index (PSI)
+ - Self-healing loop that retrains and redeploys automatically
- Model versioning and rollback API
- Workflow orchestration with Apache Airflow DAGs
- Inference API with FastAPI
- Monitoring with Prometheus and Grafana
- Frontend visualization with Streamlit

## High-Level Architecture

1. Data pipeline prepares baseline and incoming data.
2. Training pipeline creates model artifacts and versions them.
3. Drift and performance monitors evaluate model quality.
4. Self-healing module decides whether to retrain.
5. Serving layer loads the latest model dynamically.
6. Monitoring layer exposes request/latency/drift metrics.
7. Dashboard layer visualizes outputs for demo and review.

## Project Structure

```text
mlops-self-healing/
├── api/
│   ├── __init__.py
│   └── main.py
├── config/
│   └── .env.example
├── dags/
│   ├── data_ingestion_dag.py
│   ├── drift_detection_dag.py
│   ├── model_training_dag.py
│   ├── retraining_trigger_dag.py
│   └── self_healing_pipeline_dag.py
├── data/
│   ├── processed/
│   └── raw/
├── docker/
│   ├── airflow/
│   │   └── Dockerfile
│   ├── api/
│   │   └── Dockerfile
│   ├── frontend/
│   │   └── Dockerfile
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   └── self_healing_dashboard.json
│   │   └── provisioning/
│   │       ├── dashboards/
│   │       │   └── dashboard.yml
│   │       └── datasources/
│   │           └── datasource.yml
│   ├── mlflow/
│   │   └── Dockerfile
│   └── prometheus/
│       └── prometheus.yml
├── frontend/
│   └── app.py
├── models/
├── src/
│   ├── data/
│   ├── drift/
│   ├── features/
│   ├── monitoring/
│   ├── pipeline/
│   ├── serving/
│   ├── training/
│   └── utils/
├── tests/
├── .gitignore
├── docker-compose.yml
├── README.md
└── requirements.txt
```

## API Endpoints

- GET /health
   - Returns service health, model loaded status, and active version.
- POST /predict
   - Performs batch inference on rows of 4 iris features.
- POST /rollback
   - Rolls back active model to a previous version.
- GET /metrics
   - Exposes Prometheus metrics.

## Monitored Metrics

- ml_api_request_total (counter by endpoint)
- ml_api_request_latency_seconds (histogram)
- ml_prediction_class_count (gauge by class)
- ml_drift_score (gauge)

## Environment Configuration

Use values from config/.env.example or export environment variables directly.

Key variables:
- PROJECT_ROOT
- MLFLOW_TRACKING_URI
- MODEL_NAME
- DRIFT_THRESHOLD
- PERFORMANCE_THRESHOLD
- MIN_ACCEPTABLE_IMPROVEMENT
- RANDOM_STATE
- LOG_LEVEL

## Local Run (Without Docker)

### 1) Create and activate virtual environment

Windows PowerShell:

```powershell
cd C:\Users\Jerwin titus\Desktop\MLOPS\mlops-self-healing
python -m venv ..\.venv
..\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Bootstrap data + baseline model

```powershell
python -m src.pipeline.bootstrap
```

### 4) Start FastAPI service

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 5) Start Streamlit dashboard (new terminal)

```powershell
cd C:\Users\Jerwin titus\Desktop\MLOPS\mlops-self-healing
..\.venv\Scripts\Activate.ps1
$env:API_BASE_URL="http://127.0.0.1:8000"
streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
```

## Docker Compose Run (Full Stack)

Prerequisite: Docker Desktop must be running.

```bash
docker compose up --build -d
```

Service URLs:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend Dashboard: http://localhost:8501
- MLflow: http://localhost:5000
- Airflow: http://localhost:8080 (admin/admin)
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Verification Checklist

### Health check

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET | ConvertTo-Json
```

Expected fields:
- status: ok
- model_loaded: true
- model_version: integer

### Prediction check

```powershell
$body = @{ rows = @(@(5.1,3.5,1.4,0.2), @(6.2,2.8,4.8,1.8)) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -ContentType "application/json" -Body $body | ConvertTo-Json
```

### Metrics check

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/metrics" -UseBasicParsing |
   Select-Object -ExpandProperty Content |
   Select-String "ml_api_request_total|ml_drift_score|ml_prediction_class_count"
```

### Self-healing cycle check

```powershell
python -c "from src.drift.detector import detect_data_drift; import json; print(json.dumps(detect_data_drift(), indent=2))"
python -c "from src.training.self_heal import run_self_healing_cycle; import json; print(json.dumps(run_self_healing_cycle(), indent=2))"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET | ConvertTo-Json
```

Expected:
- drift_detected true when synthetic shift is present
- self-heal action retrained or rolled_back
- model_version increments (or rolls back by policy)

## Frontend Dashboard Features

At http://127.0.0.1:8501, the dashboard shows:
- Live model health and active version
- Drift score summary
- PSI per-feature bar chart
- Prediction form and response preview
- Model registry/history table
- Prometheus metrics preview

## Airflow Orchestration

Primary DAG:
- self_healing_ml_pipeline

Modular DAGs:
- ml_data_ingestion
- ml_model_training
- ml_drift_detection
- ml_retraining_trigger

Task flow:
1. Ingest
2. Preprocess
3. Baseline train (if missing)
4. Drift detect
5. Performance check
6. Branch decision
7. Retrain and redeploy or skip

## Rollback Example

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/rollback" -Method POST -ContentType "application/json" -Body '{"version": 1}' | ConvertTo-Json
```

## Testing

```bash
pytest -q
```

Current unit tests include:
- PSI behavior checks
- Model registry versioning checks

## Faculty Demo Flow (5 minutes)

1. Show architecture and modules from project structure.
2. Run health endpoint and explain current model version.
3. Run predict endpoint and show response.
4. Show dashboard charts and model history.
5. Trigger drift + self-healing, then show version update.
6. Show alerts log and rollback endpoint.

## Troubleshooting

### Docker compose fails with dockerDesktopLinuxEngine pipe error

Cause: Docker Desktop not running.

Fix:
- Start Docker Desktop.
- Re-run docker compose up --build -d.

### 500 error on /health due to numpy or sklearn mismatch

Cause: old pickled model created with different package versions.

Fix:
```powershell
Remove-Item "models\current_model.joblib" -Force -ErrorAction SilentlyContinue
python -c "from src.training.train import train_model; train_model(run_name='fresh_start')"
```

### Streamlit not found

Fix:
```powershell
pip install -r requirements.txt
```

### Port already in use

Use alternate ports, for example:
- API: 8001
- Streamlit: 8502

## Production Hardening Suggestions

- Use a remote MLflow backend store and artifact store.
- Add authentication and authorization for API and Airflow.
- Terminate TLS via reverse proxy.
- Add canary or shadow deployment before full replacement.
- Integrate real alerting channels (SMTP, Slack, PagerDuty).
- Add schema validation and data contracts for incoming payloads.

## License

Use this project for academic and learning purposes, then adapt governance, security, and compliance controls for production environments.
