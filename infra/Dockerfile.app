FROM python:3.11-slim

WORKDIR /opt/titanic-survival-mlops

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY scripts ./scripts

RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/opt/titanic-survival-mlops/src
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "titanic_mlops.training.train", "--no-register"]
