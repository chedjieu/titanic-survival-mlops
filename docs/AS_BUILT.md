# Build Guide — titanic-survival-mlops

How to install, train, package, containerize, and verify this MLOps platform locally and in CI.

## Prerequisites

| Tool | Version / notes |
|------|-----------------|
| Python | **3.11+** (`requires-python = ">=3.11"`) |
| pip | recent (`pip install --upgrade pip`) |
| Docker + Compose | Required for full stack and image builds |
| Make (optional) | Convenience targets in `Makefile` |
| kubectl (optional) | Kubernetes deploy under `infra/k8s/` |

Clone and enter the repo:

```bash
cd 11.titanic-survival-mlops
```

## 1. Local Python package build

Create a virtualenv, install the editable package with dev extras, and copy env defaults:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
copy .env.example .env          # Windows
# cp .env.example .env          # Linux / macOS
```

Equivalent Makefile target:

```bash
make install
```

### What gets installed

- **Runtime** (`pyproject.toml` dependencies): numpy, pandas, scikit-learn, MLflow, BentoML, Kafka client, Pandera, Prometheus client, etc.
- **Dev extras**: pytest, pytest-cov, ruff
- **Optional**: `pip install -e ".[airflow]"` for local Airflow (compose uses its own image)

Console entry points after install:

| Command | Module |
|---------|--------|
| `titanic-train` | `titanic_mlops.training.train` |
| `titanic-produce` | `titanic_mlops.serving.kafka_producer` |
| `titanic-consume` | `titanic_mlops.serving.kafka_runner` |
| `titanic-monitor` | `titanic_mlops.monitoring.drift` |

Or via module CLI:

```bash
python -m titanic_mlops train|produce|consume|monitor
```

## 2. Train (artifact build)

Default tracking uses SQLite (`MLFLOW_TRACKING_URI` in `.env`). Training data: `data/raw/titanic_data.csv`.

```bash
# Local train without registry registration (CI / smoke)
python -m titanic_mlops.training.train --no-register

# Full train + register (needs MLflow server or compatible URI)
python -m titanic_mlops.training.train
# or
make train
```

With the compose stack up, point tracking at the server:

```bash
# .env
MLFLOW_TRACKING_URI=http://localhost:5000
```

Artifacts typically land under `models/` (e.g. `latest_train_result.json`).

## 3. Serve locally (BentoML)

```bash
# After a successful train / with a loadable model
make serve
# or
bentoml serve src.titanic_mlops.serving.service:SurvivalService --reload
```

Bento package definition: `src/titanic_mlops/serving/bentofile.yaml`.

Build a Bento package:

```bash
make bento-build
# or
bentoml build -f src/titanic_mlops/serving/bentofile.yaml
```

Default API port when served via Docker: **3000**. Health: `GET /healthz`.

## 4. Docker image builds

Build context is the **repo root**. Dockerfiles live under `infra/`.

### App image (training / Kafka consumer)

```bash
docker build -f infra/Dockerfile.app -t titanic-mlops-app:local .
```

- Base: `python:3.11-slim`
- Installs the package editable (`pip install -e .`)
- Default CMD: train with `--no-register`
- Compose overrides CMD for `kafka-consumer` → `python -m titanic_mlops.serving.kafka_runner`

### Bento serving image

```bash
docker build -f infra/Dockerfile.bento -t titanic-survival-service:latest .
```

- Same base + package install + `bentoml`
- Exposes **3000**
- CMD: `bentoml serve titanic_mlops.serving.service:SurvivalService --host 0.0.0.0 --port 3000`
- Expects model/MLflow config via env (`MLFLOW_TRACKING_URI`, `MODEL_NAME`, `MODEL_VERSION`)

CI tags the same builds as `:ci` (see [CI build](#7-ci-build)).

## 5. Full stack (Compose)

```bash
docker compose -f infra/docker-compose.yml up -d --build
# or
make compose-up
```

Tear down (including volumes):

```bash
make compose-down
# or
docker compose -f infra/docker-compose.yml down -v
```

| Service | URL / address |
|---------|----------------|
| Airflow | http://localhost:8080 (`admin` / `admin`) |
| MLflow | http://localhost:5000 |
| Bento API | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 (`admin` / `admin`) |
| Kafka | `localhost:9092` |

Produce demo events (host venv, stack running):

```bash
python -m titanic_mlops.serving.kafka_producer --limit 20
```

Promote a registered model, then rebuild serving:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000   # or set in .env
python scripts/promote_model.py --version <N>

docker compose -f infra/docker-compose.yml up -d --build bento kafka-consumer
```

Ops details (rollback, DLQ, drift): [RUNBOOK.md](RUNBOOK.md).

## 6. Verify the build

```bash
# Lint
ruff check src tests pipelines
# or: make lint

# Tests + coverage
pytest -q --cov=titanic_mlops --cov-report=term-missing
# or: make test

# Train smoke
set MLFLOW_TRACKING_URI=sqlite:///./mlflow.db   # Windows PowerShell: $env:MLFLOW_TRACKING_URI=...
python -m titanic_mlops.training.train --no-register
```

## 7. CI build

Workflow: `.github/workflows/ci.yml` (on `push` to `main`/`master` and PRs).

Order of steps:

1. Checkout + Python 3.11  
2. `pip install -e ".[dev]"`  
3. `ruff check src tests pipelines`  
4. `pytest` with `MLFLOW_TRACKING_URI=sqlite:///./mlflow.db`  
5. Train smoke (`--no-register`, `MIN_F1_SCORE=0.50`)  
6. `docker build -f infra/Dockerfile.app -t titanic-mlops-app:ci .`  
7. `docker build -f infra/Dockerfile.bento -t titanic-survival-service:ci .`  

Match CI locally by running those same commands in order.

## 8. Kubernetes (optional)

After building/pushing `titanic-survival-service:latest`:

1. Create secrets from `infra/k8s/secrets.example.yaml`  
2. Apply deploy + service:

```bash
kubectl apply -f infra/k8s/secrets.example.yaml   # after filling real values / renaming
kubectl apply -f infra/k8s/bento-deployment.yaml
```

Deployment probes `/healthz` on port 3000; Service maps cluster port 80 → 3000.

## 9. Makefile reference

| Target | Action |
|--------|--------|
| `make install` | `pip install -e ".[dev]"` |
| `make test` | pytest + coverage |
| `make lint` | ruff |
| `make train` | full training entrypoint |
| `make serve` | BentoML serve with reload |
| `make bento-build` | `bentoml build` from bentofile |
| `make compose-up` | compose up `--build` |
| `make compose-down` | compose down `-v` |
| `make promote` | `scripts/promote_model.py` |
| `make monitor` | drift monitor module |

## 10. Build checklist (happy path)

1. `pip install -e ".[dev]"` and configure `.env`  
2. `pytest -q` and `ruff check src tests pipelines`  
3. `python -m titanic_mlops.training.train --no-register`  
4. `docker build -f infra/Dockerfile.app` and `Dockerfile.bento`  
5. `docker compose -f infra/docker-compose.yml up -d --build`  
6. Hit MLflow / Bento health / Airflow UI  
7. Promote model → rebuild `bento` when ready for Production serving  

## Related docs

- [README.md](../README.md) — architecture overview  
- [RUNBOOK.md](RUNBOOK.md) — promote, rollback, DLQ, drift  
- [MODEL_CARD.md](MODEL_CARD.md) — model documentation  
