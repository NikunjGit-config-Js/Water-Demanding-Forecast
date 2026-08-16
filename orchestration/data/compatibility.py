from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import ProvenanceManifest, sha256_file
from .status import DataStatus


MINIMUM_OBSERVATIONS = 523


@dataclass(frozen=True)
class CompatibilityReport:
    city: str
    status: DataStatus
    checks: dict[str, bool]
    capabilities: dict[str, bool]
    reasons: tuple[str, ...]
    rows: int
    date_start: str | None
    date_end: str | None
    gap_count: int
    minimum_observations: int = MINIMUM_OBSERVATIONS
    schema_version: int = 1

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self); payload["status"] = self.status.value
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)


def evaluate_compatibility(city: str, dataset: Path, provenance: ProvenanceManifest | None) -> CompatibilityReport:
    reasons: list[str] = []
    try:
        raw = pd.read_csv(dataset)
    except Exception as exc:
        return CompatibilityReport(city, DataStatus.DATA_INCOMPATIBLE, {"dataset_readable": False}, {},
                                   (f"dataset unreadable: {type(exc).__name__}",), 0, None, None, 0)
    exact = list(raw.columns) == ["Date", "Consumption"]
    dates = pd.to_datetime(raw.get("Date"), errors="coerce") if "Date" in raw else pd.Series(dtype="datetime64[ns]")
    target = pd.to_numeric(raw.get("Consumption"), errors="coerce") if "Consumption" in raw else pd.Series(dtype=float)
    valid_dates = len(dates) == len(raw) and not dates.isna().any()
    numeric = len(target) == len(raw) and not target.isna().any()
    finite = numeric and bool(np.isfinite(target).all())
    chronological = valid_dates and dates.is_monotonic_increasing and not dates.duplicated().any()
    provenance_present = provenance is not None
    hash_valid = bool(provenance and provenance.canonical_sha256 == sha256_file(dataset))
    unit_recorded = bool(provenance and provenance.source_unit and provenance.canonical_unit)
    daily_semantics = bool(provenance and provenance.expected_frequency.lower() in {"daily", "hourly", "subdaily", "15min", "30min"})
    sufficient = len(raw) >= MINIMUM_OBSERVATIONS
    span = bool(valid_dates and len(raw) and (dates.max() - dates.min()).days >= 365)
    training_rows = int(np.floor(len(raw) * .70))
    checks = {"canonical_columns": exact, "dates_valid": valid_dates, "chronological_unique": chronological,
              "target_numeric": numeric, "target_finite": finite, "provenance_present": provenance_present,
              "dataset_hash_valid": hash_valid, "source_unit_recorded": unit_recorded,
              "daily_compatible_frequency": daily_semantics, "sufficient_observations": sufficient,
              "at_least_one_year_span": span}
    for name, passed in checks.items():
        if not passed: reasons.append(name)
    capabilities = {"lag_365": len(raw) >= 366 and span,
                    "lag_365_in_training_prefix": training_rows >= 366 and span,
                    "phase0_window": int(.90 * len(raw)) > 300,
                    "time_series_cv_5_fold": len(raw) >= 36,
                    "split_70_15_15": len(raw) >= 7,
                    "locked_test": len(raw) >= 7,
                    "deep_time_series_models": len(raw) >= MINIMUM_OBSERVATIONS}
    gaps = int((dates.sort_values().diff().dt.days.dropna() != 1).sum()) if valid_dates else 0
    ready = all(checks.values()) and all(capabilities.values())
    return CompatibilityReport(city, DataStatus.READY if ready else DataStatus.DATA_INCOMPATIBLE,
                               checks, capabilities, tuple(reasons), len(raw),
                               dates.min().date().isoformat() if valid_dates and len(raw) else None,
                               dates.max().date().isoformat() if valid_dates and len(raw) else None, gaps)
