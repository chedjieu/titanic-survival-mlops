"""Kafka consumer: passenger.events → model → survival.predictions (+ DLQ)."""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from typing import Any

from confluent_kafka import Consumer, KafkaException, Producer

from titanic_mlops.config import get_settings
from titanic_mlops.serving.model_loader import load_pipeline
from titanic_mlops.serving.schemas import PassengerFeatures, PredictionResponse

logger = logging.getLogger(__name__)
_RUNNING = True


def _handle_signal(signum, frame):  # noqa: ANN001, ARG001
    global _RUNNING
    _RUNNING = False


def _delivery_report(err, msg) -> None:  # noqa: ANN001
    if err is not None:
        logger.error("Kafka delivery failed: %s", err)


def process_message(pipeline: Any, model_version: str, payload: dict) -> dict:
    passenger = PassengerFeatures.model_validate(payload)
    import pandas as pd

    frame = pd.DataFrame([passenger.to_record()])
    proba = float(pipeline.predict_proba(frame)[0, 1])
    label = int(proba >= 0.5)
    response = PredictionResponse(
        survived=label,
        survival_probability=proba,
        model_version=model_version,
        passenger_id=passenger.passenger_id or payload.get("passenger_id"),
    )
    result = response.model_dump()
    result["event_id"] = payload.get("event_id")
    result["idempotency_key"] = payload.get("idempotency_key") or payload.get("event_id")
    return result


def run_consumer() -> int:
    settings = get_settings()
    pipeline, model_version = load_pipeline()

    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": settings.kafka_group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})
    consumer.subscribe([settings.kafka_topic_events])
    logger.info(
        "Consuming %s → %s (dlq=%s)",
        settings.kafka_topic_events,
        settings.kafka_topic_predictions,
        settings.kafka_topic_dlq,
    )

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    while _RUNNING:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            raise KafkaException(msg.error())

        try:
            payload = json.loads(msg.value().decode("utf-8"))
            result = process_message(pipeline, model_version, payload)
            producer.produce(
                settings.kafka_topic_predictions,
                key=(result.get("idempotency_key") or "").encode("utf-8"),
                value=json.dumps(result).encode("utf-8"),
                callback=_delivery_report,
            )
            producer.poll(0)
            consumer.commit(asynchronous=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process message; sending to DLQ")
            dlq_payload = {
                "error": str(exc),
                "raw": msg.value().decode("utf-8", errors="replace"),
                "ts": time.time(),
            }
            producer.produce(
                settings.kafka_topic_dlq,
                value=json.dumps(dlq_payload).encode("utf-8"),
                callback=_delivery_report,
            )
            producer.poll(0)
            consumer.commit(asynchronous=False)

    producer.flush()
    consumer.close()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run_consumer()


if __name__ == "__main__":
    sys.exit(main())
