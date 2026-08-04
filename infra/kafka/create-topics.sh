#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP="${KAFKA_BOOTSTRAP_SERVERS:-kafka:29092}"

create_topic() {
  local topic="$1"
  kafka-topics --bootstrap-server "$BOOTSTRAP" --create --if-not-exists \
    --topic "$topic" --partitions 3 --replication-factor 1
}

create_topic "passenger.events"
create_topic "survival.predictions"
create_topic "survival.dlq"

echo "Kafka topics ready"
