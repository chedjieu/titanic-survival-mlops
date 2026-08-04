"""Drain a batch of prediction messages from Kafka into a JSONL file."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from confluent_kafka import Consumer, KafkaException

from titanic_mlops.config import get_settings

logger = logging.getLogger(__name__)


def collect(output: Path, max_messages: int, timeout: float) -> int:
    settings = get_settings()
    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": f"{settings.kafka_group_id}-collector-{int(time.time())}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe([settings.kafka_topic_predictions])
    output.parent.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + timeout
    count = 0
    with output.open("w", encoding="utf-8") as fh:
        while count < max_messages and time.time() < deadline:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            fh.write(msg.value().decode("utf-8") + "\n")
            count += 1

    consumer.close()
    logger.info("Wrote %s predictions to %s", count, output)
    return count


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-messages", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args(argv)
    collect(args.output, args.max_messages, args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
