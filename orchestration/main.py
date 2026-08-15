from __future__ import annotations

import argparse
import json

from orchestration.flow import WaterForecastFlow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the validated CrewAI phase flow.")
    parser.add_argument(
        "--phase",
        action="append",
        required=True,
        help="Explicit phase to run; repeat to run multiple phases in order.",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    flow = WaterForecastFlow(phases=args.phase, max_attempts=args.max_attempts)
    # Invoke the decorated deterministic entrypoint directly: this uses the
    # supported Flow definition without enabling asynchronous event handlers.
    flow.orchestrate()
    print(json.dumps(flow.state.model_dump(), indent=2, sort_keys=True))
    return 0 if flow.state.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
