# titanic-survival-mlops

Enterprise reference MLOps platform for **Titanic passenger survival** prediction.

Orchestrates training with **Airflow**, tracks/registers models with **MLflow**, serves with **BentoML**, and streams inference via **Kafka**.

## Architecture

```text
CSV / Kafka events → Airflow train_and_register → MLflow Registry
                              ↓
                     BentoML SurvivalService (REST)
                              ↓
              Kafka passenger.events → predictions (+ DLQ)
                              ↓
                     monitor_drift → retrain trigger
```

See [docs/AS_BUILT.md](docs/AS_BUILT.md) for the full build, package, Docker, and CI guide.

## Quick start (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# Train (local sqlite tracking by default; use http://localhost:5000 with compose)
python -m titanic_mlops.training.train --no-register

pytest -q
```

### Full stack

```bash
docker compose -f infra/docker-compose.yml up -d --build
```

| Service    | URL                         |
|-----------|-----------------------------|
| Airflow   | http://localhost:8080 (admin/admin) |
| MLflow    | http://localhost:5000       |
| Bento API | http://localhost:3000       |
| Prometheus| http://localhost:9090       |
| Grafana   | http://localhost:3001 (admin/admin) |
| Kafka     | localhost:9092              |

Produce demo events:

```bash
python -m titanic_mlops.serving.kafka_producer --limit 20
```

## Project layout

- `src/titanic_mlops/features` — shared transforms (train = serve)
- `src/titanic_mlops/training` — MLflow-backed training + gates
- `src/titanic_mlops/serving` — BentoML service + Kafka consumer/producer
- `src/titanic_mlops/monitoring` — PSI / prediction-rate drift
- `pipelines/airflow/dags` — train, deploy, monitor DAGs
- `infra/` — docker-compose, Prometheus, Grafana, K8s, Terraform
- `research/` — original Titanic notebook (provenance)

## Model promotion

```bash
python scripts/promote_model.py --version 1
```

See [docs/RUNBOOK.md](docs/RUNBOOK.md) for rollback and DLQ replay.

## Legacy Azure ML stub

The unfinished Azure ML sample was moved to `research/azure-ml-legacy/` and credentials were removed. Use `.example.json` as a template; never commit real workspace configs.
