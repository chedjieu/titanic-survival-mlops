# Operations Runbook — titanic-survival-mlops

## Promote a model to Production

1. Confirm the run passed the F1 quality gate in MLflow UI.
2. Promote:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python scripts/promote_model.py --version <N>
```

3. Redeploy serving:

```bash
docker compose -f infra/docker-compose.yml up -d --build bento kafka-consumer
```

Or trigger Airflow DAG `deploy_bento`.

## Rollback Bento image

```bash
# redeploy previous known-good image tag
docker tag titanic-survival-service:<previous> titanic-survival-service:latest
docker compose -f infra/docker-compose.yml up -d --no-deps bento
curl -sf http://localhost:3000/healthz
```

In Kubernetes, roll back the deployment:

```bash
kubectl rollout undo deployment/titanic-survival-service
```

Also transition the prior MLflow model version back to `Production` and archive the bad one.

## Replay DLQ

Messages that fail validation/inference land on `survival.dlq`.

1. Inspect:

```bash
docker compose -f infra/docker-compose.yml exec kafka \
  kafka-console-consumer --bootstrap-server kafka:29092 --topic survival.dlq --from-beginning --timeout-ms 5000
```

2. Fix payload issues, then republish cleaned events to `passenger.events` with the same `idempotency_key`.

## Drift response

If `monitor_drift` sets `drift_detected=true`:

1. Review `models/latest_drift_result.json`
2. Confirm live traffic quality (schema changes, producer bugs)
3. Allow the triggered `train_and_register` DAG to complete, or run training manually
4. Promote only if the gate passes; then redeploy Bento

## Secrets

- Never commit `config.json`, `.env`, or cloud keys
- Use `infra/k8s/secrets.example.yaml` as a template for cluster secrets
- Rotate any credentials that were previously committed in legacy Azure configs
