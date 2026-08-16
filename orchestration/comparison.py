from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from orchestration.context import RunContext
from orchestration.data.provenance import ProvenanceManifest
from orchestration.data.registry import get_city_source


FIELDS = (
    "city", "source", "date_start", "date_end", "rows", "unit",
    "best_naive_baseline", "naive_mae", "best_conventional_ml", "ml_mae",
    "ml_rmse", "ml_r2", "best_time_series_model", "time_series_mae",
    "modern_model", "modern_model_mae", "validator_status",
)


def write_multi_city_comparison(rows: list[dict[str, Any]], report_root: Path) -> tuple[Path, Path]:
    """Write only supplied, protocol-compatible values; never infer missing metrics."""
    report_root.mkdir(parents=True, exist_ok=True)
    normalized = [{field: row.get(field) for field in FIELDS} for row in rows]
    csv_path = report_root / "multi_city_comparison.csv"
    json_path = report_root / "multi_city_comparison.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(normalized)
    json_path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def _best_metrics(context: RunContext, phase: int, *, excluded: set[str] | None = None) -> dict[str, str] | None:
    candidates = sorted(context.phase_artifact_root(phase).glob("*/metrics.csv"))
    if not candidates:
        return None
    with candidates[-1].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    excluded = excluded or set()
    eligible = [row for row in rows if row.get("model") not in excluded and row.get("mae")]
    return min(eligible, key=lambda row: float(row["mae"])) if eligible else None


def build_comparison_row(context: RunContext, validator_status: str) -> dict[str, Any]:
    """Extract same-phase metrics; absent evidence remains null."""
    row: dict[str, Any] = {
        "city": context.city,
        "source": get_city_source(context.city).source_name,
        "validator_status": validator_status,
    }
    provenance_path = context.dataset_path.parent / "provenance.json"
    if provenance_path.exists():
        provenance = ProvenanceManifest.read(provenance_path)
        row.update(date_start=provenance.date_start, date_end=provenance.date_end,
                   rows=provenance.row_count, unit=provenance.canonical_unit)
    elif context.dataset_path.exists():
        frame = pd.read_csv(context.dataset_path, usecols=["Date"])
        dates = pd.to_datetime(frame["Date"], errors="raise")
        row.update(date_start=dates.min().date().isoformat(), date_end=dates.max().date().isoformat(),
                   rows=len(frame), unit=None)

    phase5 = _best_metrics(context, 5)
    if phase5:
        with sorted(context.phase_artifact_root(5).glob("*/metrics.csv"))[-1].open(
            newline="", encoding="utf-8"
        ) as stream:
            metrics = list(csv.DictReader(stream))
        naive = min((r for r in metrics if r["model"].startswith("naive")),
                    key=lambda r: float(r["mae"]), default=None)
        conventional = min((r for r in metrics if not r["model"].startswith("naive")),
                           key=lambda r: float(r["mae"]), default=None)
        if naive:
            row.update(best_naive_baseline=naive["model"], naive_mae=float(naive["mae"]))
        if conventional:
            row.update(best_conventional_ml=conventional["model"], ml_mae=float(conventional["mae"]),
                       ml_rmse=float(conventional["rmse"]), ml_r2=float(conventional["r2"]))
    phase9 = _best_metrics(context, 9, excluded={"naive_last", "seasonal_naive_7"})
    if phase9:
        row.update(best_time_series_model=phase9["model"], time_series_mae=float(phase9["mae"]))
    phase10 = _best_metrics(context, 10, excluded={"naive_lag1"})
    if phase10:
        row.update(modern_model=phase10["model"], modern_model_mae=float(phase10["mae"]))
    return row
