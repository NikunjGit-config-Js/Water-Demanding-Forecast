from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from .acquisition import AcquisitionResult, acquire
from .adapters import read_source
from .compatibility import CompatibilityReport, evaluate_compatibility
from .normalize import normalize
from .provenance import ProvenanceManifest, sha256_file
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


def _run_djb_archive_pipeline(
    city: str,
    city_data_root: Path,
    source: CitySource,
) -> CompatibilityReport:
    """Run the Delhi DJB multi-document archive pipeline."""
    from .sources.delhi_djb import (
        AuditReport,
        run_djb_acquisition,
        sha256_bytes,
        write_audit_report,
        write_canonical_csv,
    )

    canonical_dir = city_data_root / "canonical"
    report_path = canonical_dir / "city_data_compatibility.json"

    try:
        df, audit, classifications = run_djb_acquisition(
            city_data_root,
            live_network=True,
        )
    except Exception as exc:
        report = _stopped_report(city, DataStatus.ACQUISITION_FAILED, f"DJB archive error: {type(exc).__name__}: {exc}")
        report.write(report_path)
        return report

    if df.empty:
        report = _stopped_report(
            city, DataStatus.DATA_INCOMPATIBLE,
            f"DJB archive yielded no daily production observations; "
            f"discovered={audit.documents_discovered}, accepted={audit.documents_accepted}, "
            f"rejected={audit.documents_rejected}",
        )
        report.write(report_path)
        write_audit_report(audit, canonical_dir / "djb_audit.json")
        return report

    canonical_path = canonical_dir / "water_demand.csv"
    canonical_hash = write_canonical_csv(df, canonical_path)

    docs_dir = city_data_root / "documents"
    raw_files = list(docs_dir.glob("*.pdf")) if docs_dir.exists() else []
    combined_raw_hash = sha256_bytes(
        b"".join(f.read_bytes() for f in sorted(raw_files)[:10])
    ) if raw_files else ""

    provenance = ProvenanceManifest(
        city=city,
        source_name="Delhi Jal Board - Daily Water Production Report Archive",
        source_url="https://delhijalboard.delhi.gov.in/daily-water-production-report",
        official=True,
        acquisition_method="djb_archive_crawl",
        downloaded_at_utc=datetime.now(UTC).isoformat(),
        raw_format="pdf_archive",
        raw_path=str(docs_dir),
        raw_sha256=combined_raw_hash,
        canonical_path=str(canonical_path),
        canonical_sha256=canonical_hash,
        source_unit="MGD",
        canonical_unit="MGD",
        adapter_type="djb_archive",
        expected_frequency="daily",
        transformations=(
            f"classified {audit.documents_discovered} archive entries",
            f"accepted {audit.documents_accepted} production documents",
            f"rejected {audit.documents_rejected} non-production documents",
            f"extracted {audit.unique_observations} daily observations",
            f"resolved {audit.duplicate_count} duplicates",
            f"excluded {audit.conflicting_duplicate_count} conflicting duplicates",
        ),
        wards_aggregated=False,
        row_count=len(df),
        date_start=audit.date_range_start or "",
        date_end=audit.date_range_end or "",
    )
    provenance.write(canonical_dir / "provenance.json")

    write_audit_report(audit, canonical_dir / "djb_audit.json")

    report = evaluate_compatibility(city, canonical_path, provenance)
    report.write(report_path)
    return report


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

    if source.adapter_type == "djb_archive":
        return _run_djb_archive_pipeline(city, city_data_root, source)

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
