# Legacy Azure ML stub (retired)

This folder preserves the original EuroPython / golden-scenario Azure ML submit scaffold.

**Status:** non-functional for production. `train.py` historically contained imports only.

Do **not** commit real `config.json` credentials. Copy `config.example.json` locally if you experiment with Azure ML outside this MLOps stack.

The supported path is local/docker **Airflow + MLflow + BentoML + Kafka** under `infra/docker-compose.yml`.
