"""Replay Titanic CSV rows (or sample events) onto Kafka passenger.events."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path

from confluent_kafka import Producer

from titanic_mlops.config import get_settings
from titanic_mlops.features.transform import load_raw_dataframe

logger = logging.getLogger(__name__)


def _delivery_report(err, msg) -> None:  # noqa: ANN001
    if err is not None:
        logger.error("Delivery failed: %s", err)
    else:
        logger.debug("Delivered to %s [%s]", msg.topic(), msg.partition())


def produce_from_csv(csv_path: Path, limit: int | None = None, sleep_s: float = 0.0) -> int:
    settings = get_settings()
    df = load_raw_dataframe(csv_path)
    producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})

    count = 0
    for idx, row in df.iterrows():
        if limit is not None and count >= limit:
            break
        event_id = str(uuid.uuid4())
        payload = {
            "event_id": event_id,
            "idempotency_key": event_id,
            "passenger_id": str(idx),
            "Pclass": int(row["Pclass"]),
            "Sex": str(row["Sex"]),
            "Age": None if pd_isna(row.get("Age")) else float(row["Age"]),
            "SibSp": int(row["SibSp"]),
            "Parch": int(row["Parch"]),
            "Fare": None if pd_isna(row.get("Fare")) else float(row["Fare"]),
            "Embarked": None if pd_isna(row.get("Embarked")) else str(row["Embarked"]),
            "ts": time.time(),
        }
        producer.produce(
            settings.kafka_topic_events,
            key=event_id.encode("utf-8"),
            value=json.dumps(payload).encode("utf-8"),
            callback=_delivery_report,
        )
        producer.poll(0)
        count += 1
        if sleep_s:
            time.sleep(sleep_s)

    producer.flush()
    logger.info("Produced %s events to %s", count, settings.kafka_topic_events)
    return count


def pd_isna(value) -> bool:  # noqa: ANN001
    import pandas as pd

    return bool(pd.isna(value))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Produce passenger events to Kafka")
    parser.add_argument("--data-path", type=Path, default=settings.data_path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args(argv)
    produce_from_csv(args.data_path, limit=args.limit, sleep_s=args.sleep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
