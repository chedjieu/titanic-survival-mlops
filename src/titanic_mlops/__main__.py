"""CLI: python -m titanic_mlops train|produce|consume|monitor"""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m titanic_mlops [train|produce|consume|monitor]")
        return 1
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    if cmd == "train":
        from titanic_mlops.training.train import main as train_main

        return train_main(rest)
    if cmd == "produce":
        from titanic_mlops.serving.kafka_producer import main as produce_main

        return produce_main(rest)
    if cmd == "consume":
        from titanic_mlops.serving.kafka_runner import main as consume_main

        return consume_main()
    if cmd == "monitor":
        from titanic_mlops.monitoring.drift import main as monitor_main

        return monitor_main(rest)
    print(f"Unknown command: {cmd}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
