from __future__ import annotations

import argparse
import json

from orchestration.auto import AutoRunner
from orchestration.flow import WaterForecastFlow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the validated CrewAI phase flow.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--phase",
        action="append",
        help="Explicit phase to run; repeat to run multiple phases in order.",
    )
    mode.add_argument(
        "--auto",
        action="store_true",
        help="Resume at the first incomplete phase and run through Phase 13.",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.auto:
        try:
            summary = AutoRunner(max_attempts=args.max_attempts).run()
        except (RuntimeError, ValueError) as exc:
            summary = {"status": "failed", "warnings": [str(exc)]}
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "completed" else 1
    flow = WaterForecastFlow(phases=args.phase, max_attempts=args.max_attempts)
    # Invoke the decorated deterministic entrypoint directly: this uses the
    # supported Flow definition without enabling asynchronous event handlers.
    flow.orchestrate()
    print(json.dumps(flow.state.model_dump(), indent=2, sort_keys=True))
    return 0 if flow.state.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
