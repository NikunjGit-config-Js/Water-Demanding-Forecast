from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import sha256_file
from .registry import CitySource
from .status import DataStatus


INCOMPATIBLE_FREQUENCIES = {"monthly", "quarterly", "annual", "yearly", "irregular", "summary"}
SUBDAILY_FREQUENCIES = {"hourly", "subdaily", "15min", "30min"}


@dataclass(frozen=True)
class NormalizationResult:
    status: DataStatus
    canonical_path: Path | None = None
    canonical_sha256: str | None = None
    frame: pd.DataFrame | None = None
    transformations: tuple[str, ...] = ()
    wards_aggregated: bool = False
    reason: str | None = None


def normalize(frame: pd.DataFrame, source: CitySource, canonical_path: Path) -> NormalizationResult:
    frequency = (source.expected_frequency or "").lower()
    if frequency in INCOMPATIBLE_FREQUENCIES:
        return NormalizationResult(DataStatus.DATA_INCOMPATIBLE,
                                   reason=f"{frequency} targets cannot be expanded to daily observations")
    if not source.date_column or not source.consumption_column:
        return NormalizationResult(DataStatus.DATA_INCOMPATIBLE, reason="source column mapping is missing")
    required = {source.date_column, source.consumption_column}
    if source.zone_column: required.add(source.zone_column)
    if not required.issubset(frame.columns):
        return NormalizationResult(DataStatus.DATA_INCOMPATIBLE, reason=f"missing source columns: {sorted(required-frame.columns)}")
    work = frame.loc[:, list(required)].copy()
    work["Date"] = pd.to_datetime(work[source.date_column], errors="coerce")
    work["Consumption"] = pd.to_numeric(work[source.consumption_column], errors="coerce")
    if work["Date"].isna().any() or work["Consumption"].isna().any() or not np.isfinite(work["Consumption"]).all():
        return NormalizationResult(DataStatus.DATA_INCOMPATIBLE, reason="dates and targets must be valid and finite")
    transformations: list[str] = []
    if source.unit_multiplier != 1.0:
        work["Consumption"] *= source.unit_multiplier
        transformations.append(f"unit multiplied by {source.unit_multiplier:g}")
    wards = bool(source.zone_column)
    if wards:
        transformations.append(f"aggregated {source.zone_column} rows by daily sum")
    if frequency in SUBDAILY_FREQUENCIES:
        if source.aggregation != "sum" or not source.unit or not source.canonical_unit:
            return NormalizationResult(DataStatus.DATA_INCOMPATIBLE,
                                       reason="sub-daily aggregation requires additive unit and explicit daily sum")
        work["Date"] = work["Date"].dt.floor("D")
        transformations.append("aggregated genuine sub-daily observations by daily sum")
    duplicates = work["Date"].duplicated(keep=False).any()
    if duplicates and source.aggregation != "sum":
        return NormalizationResult(DataStatus.DATA_INCOMPATIBLE,
                                   reason="duplicate dates require an explicit deterministic aggregation")
    if duplicates:
        work = work.groupby("Date", as_index=False, sort=True)["Consumption"].sum()
        if not (wards or frequency in SUBDAILY_FREQUENCIES):
            transformations.append("resolved duplicate dates by configured sum")
    else:
        work = work.loc[:, ["Date", "Consumption"]]
    work = work.sort_values("Date", kind="stable").reset_index(drop=True)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = canonical_path.with_suffix(canonical_path.suffix + ".tmp")
    work.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(canonical_path)
    return NormalizationResult(DataStatus.READY, canonical_path, sha256_file(canonical_path), work,
                               tuple(transformations), wards)
