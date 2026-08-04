.PHONY: install test lint train serve compose-up compose-down bento-build promote

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q --cov=titanic_mlops --cov-report=term-missing

lint:
	ruff check src tests pipelines

train:
	python -m titanic_mlops.training.train

serve:
	bentoml serve src.titanic_mlops.serving.service:SurvivalService --reload

bento-build:
	bentoml build -f src/titanic_mlops/serving/bentofile.yaml

compose-up:
	docker compose -f infra/docker-compose.yml up -d --build

compose-down:
	docker compose -f infra/docker-compose.yml down -v

promote:
	python scripts/promote_model.py

monitor:
	python -m titanic_mlops.monitoring.drift
