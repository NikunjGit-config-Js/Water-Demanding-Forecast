from __future__ import annotations

from pathlib import Path
from typing import Callable

from .acquisition import AcquisitionResult, acquire
from .adapters import read_source
from .compatibility import CompatibilityReport, evaluate_compatibility
from .normalize import normalize
from .provenance import ProvenanceManifest
from .registry import CitySource, get_city_source
from .status import DataStatus


def _stopped_report(city: str, status: DataStatus, reason: str) -> CompatibilityReport:
    return CompatibilityReport(
        city=city,
        status=status,
        checks={"source_configured": False},
        capabilities={},
        reasons=(reason,),
        rows=0,
        date_start=None,
        date_end=None,
        gap_count=0,
    )


def prepare_city_data(
    city: str,
    city_data_root: Path,
    *,
    source: CitySource | None = None,
    acquirer: Callable[[CitySource, Path], AcquisitionResult] = acquire,
) -> CompatibilityReport:
    """Acquire, normalize, record provenance, and gate one non-London city."""
    source = source or get_city_source(city)
    canonical_dir = city_data_root / "canonical"
    report_path = canonical_dir / "city_data_compatibility.json"
    if not source.configured:
        report = _stopped_report(city, DataStatus.DATA_SOURCE_REQUIRED, "verified daily source is not configured")
        report.write(report_path)
        return report

    acquisition = acquirer(source, city_data_root / "raw")
    if acquisition.status != DataStatus.READY or acquisition.raw_path is None:
        report = _stopped_report(city, acquisition.status, acquisition.reason or "acquisition failed")
        report.write(report_path)
        return report
    try:
        frame = read_source(acquisition.raw_path, source)
    except (OSError, ValueError) as exc:
        report = _stopped_report(city, DataStatus.DATA_INCOMPATIBLE, f"adapter failed: {type(exc).__name__}")
        report.write(report_path)
        return report

    canonical_path = canonical_dir / "water_demand.csv"
    normalized = normalize(frame, source, canonical_path)
    if normalized.status != DataStatus.READY or normalized.frame is None:
        report = _stopped_report(city, normalized.status, normalized.reason or "normalization failed")
        report.write(report_path)
        return report
    dates = normalized.frame["Date"]
    provenance = ProvenanceManifest(
        city=city,
        source_name=source.source_name or "",
        source_url=acquisition.source_url,
        official=source.official,
        acquisition_method=acquisition.method or "",
        downloaded_at_utc=acquisition.downloaded_at_utc,
        raw_format=source.format or acquisition.raw_path.suffix.lstrip("."),
        raw_path=str(acquisition.raw_path),
        raw_sha256=acquisition.sha256 or "",
        canonical_path=str(canonical_path),
        canonical_sha256=normalized.canonical_sha256 or "",
        source_unit=source.unit,
        canonical_unit=source.canonical_unit,
        adapter_type=source.adapter_type or "",
        expected_frequency=source.expected_frequency or "",
        transformations=normalized.transformations,
        wards_aggregated=normalized.wards_aggregated,
        row_count=len(normalized.frame),
        date_start=dates.min().date().isoformat(),
        date_end=dates.max().date().isoformat(),
    )
    provenance.write(canonical_dir / "provenance.json")
    report = evaluate_compatibility(city, canonical_path, provenance)
    report.write(report_path)
    return report
