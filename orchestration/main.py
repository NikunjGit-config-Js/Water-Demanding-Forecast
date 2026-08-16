from __future__ import annotations

import argparse
import json

from orchestration.auto import AutoRunner
from orchestration.flow import WaterForecastFlow
from orchestration.context import RunContext
from orchestration.comparison import build_comparison_row, write_multi_city_comparison
from orchestration.data import DataStatus, prepare_city_data


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
    cities = parser.add_mutually_exclusive_group()
    cities.add_argument("--city", default="london")
    cities.add_argument("--cities", nargs="+", metavar="CITY")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.phase is not None and args.cities is not None:
        parser.error("--cities is supported only with --auto")
    selected_cities = args.cities or [args.city]
    contexts = [RunContext.for_city(city) for city in selected_cities]
    if args.auto:
        summaries = []
        for context in contexts:
            try:
                if not context.legacy_london:
                    data_report = prepare_city_data(
                        context.city, context.dataset_path.parents[1]
                    )
                    if data_report.status != DataStatus.READY:
                        summaries.append({
                            "city": context.city,
                            "status": data_report.status.value,
                            "warnings": list(data_report.reasons),
                        })
                        continue
                runner_kwargs = {"max_attempts": args.max_attempts}
                if not context.legacy_london or args.city != "london" or args.cities is not None:
                    runner_kwargs["run_context"] = context
                summary = AutoRunner(**runner_kwargs).run()
            except (RuntimeError, ValueError) as exc:
                summary = {"status": "failed", "warnings": [str(exc)]}
            summary = {"city": context.city, **summary}
            summaries.append(summary)
        if args.cities is not None:
            completed_rows = [
                build_comparison_row(
                    RunContext.for_city(summary["city"]),
                    summary.get("final_validator_status", "NOT_RUN"),
                )
                for summary in summaries if summary.get("status") == "completed"
            ]
            if len(completed_rows) >= 2:
                write_multi_city_comparison(
                    completed_rows, RunContext.for_city("london").report_root / "cities"
                )
            result = {
                "status": "completed" if all(s["status"] == "completed" for s in summaries) else "failed",
                "cities": summaries,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "completed" else 1
        summary = summaries[0]
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "completed" else 1
    flow = WaterForecastFlow(
        phases=args.phase, max_attempts=args.max_attempts, run_context=contexts[0]
    )
    # Invoke the decorated deterministic entrypoint directly: this uses the
    # supported Flow definition without enabling asynchronous event handlers.
    flow.orchestrate()
    print(json.dumps(flow.state.model_dump(), indent=2, sort_keys=True))
    return 0 if flow.state.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
