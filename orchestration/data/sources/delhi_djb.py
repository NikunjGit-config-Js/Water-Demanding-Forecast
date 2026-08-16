"""Delhi Jal Board daily water production adapter.

Discovers the paginated DJB archive, downloads documents, classifies them,
and extracts verified daily total water production in MGD.

Accept only genuine municipal potable-water production records.
Reject water sample collection, quality-only, STP/sewage, wastewater,
notices/circulars, and ambiguous documents.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Sequence

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore[assignment]

ARCHIVE_BASE = "https://delhijalboard.delhi.gov.in"
ARCHIVE_URL = f"{ARCHIVE_BASE}/daily-water-production-report"
MAX_PAGES = 20
REJECT_TITLE_PATTERNS = (
    re.compile(r"sample", re.IGNORECASE),
    re.compile(r"collection\s+report", re.IGNORECASE),
    re.compile(r"quality", re.IGNORECASE),
    re.compile(r"stp", re.IGNORECASE),
    re.compile(r"sewage", re.IGNORECASE),
    re.compile(r"wastewater", re.IGNORECASE),
    re.compile(r"water\s+quality", re.IGNORECASE),
    re.compile(r"drinking\s+water\s+quality", re.IGNORECASE),
    re.compile(r"suo-moto", re.IGNORECASE),
    re.compile(r"circular", re.IGNORECASE),
    re.compile(r"notice", re.IGNORECASE),
)
ACCEPT_TITLE_PATTERNS = (
    re.compile(r"daily\s+water\s+production", re.IGNORECASE),
    re.compile(r"water\s+production", re.IGNORECASE),
    re.compile(r"total\s+water\s+production", re.IGNORECASE),
    re.compile(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s+to\s+\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"),
)
MGD_KEYWORDS = (
    "total", "production", "mgd", "mld", "supplied", "water supply",
    "delhi water", "jal board", "daily production",
)
MGD_EXCLUSION_KEYWORDS = ("sample", "quality", "stp", "sewage", "wastewater", "test")
MIN_VALID_MGD = 500.0
MAX_VALID_MGD = 1500.0


@dataclass(frozen=True)
class ArchiveEntry:
    page: int
    row_index: int
    title: str
    date_text: str
    pdf_url: str | None
    has_file: bool


@dataclass
class DocumentClassification:
    entry: ArchiveEntry
    category: str  # "production", "sample_report", "stpw", "unknown"
    reason: str
    downloaded: bool = False
    file_path: str | None = None
    sha256: str | None = None


@dataclass
class DailyObservation:
    date: date
    consumption_mgd: float
    source_document: str
    source_sha256: str
    archive_title: str
    extraction_method: str
    explicit_total: bool
    quality_flags: tuple[str, ...] = ()


@dataclass
class AuditReport:
    pages_discovered: int = 0
    documents_discovered: int = 0
    documents_accepted: int = 0
    documents_rejected: int = 0
    documents_unparsed: int = 0
    unique_observations: int = 0
    date_range_start: str | None = None
    date_range_end: str | None = None
    missing_days: int = 0
    calendar_span_days: int = 0
    coverage_pct: float = 0.0
    duplicate_count: int = 0
    conflicting_duplicate_count: int = 0
    total_entries_in_archive: int = 0
    entries_with_files: int = 0
    production_doc_count: int = 0
    sample_report_count: int = 0
    stpw_count: int = 0
    unknown_count: int = 0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_page(session: requests.Session, url: str, timeout: int = 30) -> str:
    resp = session.get(url, timeout=timeout, headers={
        "User-Agent": "water-forecast-data/1 (academic research)"
    })
    resp.raise_for_status()
    return resp.text


def _parse_archive_page(html: str, page_number: int) -> list[ArchiveEntry]:
    soup = BeautifulSoup(html, "lxml")
    rows = soup.select("table.views-table tbody tr")
    entries: list[ArchiveEntry] = []
    for idx, tr in enumerate(rows):
        cells = tr.find_all("td")
        if len(cells) < 3:
            continue
        if len(cells) >= 4:
            title_text = cells[1].get_text(strip=True)
            date_text = cells[2].get_text(strip=True)
            link_cell = cells[3]
        else:
            title_text = cells[1].get_text(strip=True)
            date_text = ""
            link_cell = cells[2]

        link_tag = link_cell.find("a", href=True)
        pdf_url = None
        if link_tag:
            href = link_tag["href"]
            if href.startswith("/"):
                pdf_url = ARCHIVE_BASE + href
            elif href.startswith("http"):
                pdf_url = href
        entries.append(ArchiveEntry(
            page=page_number,
            row_index=idx,
            title=title_text,
            date_text=date_text,
            pdf_url=pdf_url,
            has_file=pdf_url is not None,
        ))
    return entries


def _discover_archive(
    session: requests.Session,
    *,
    max_pages: int = MAX_PAGES,
    timeout: int = 30,
) -> tuple[list[ArchiveEntry], int]:
    all_entries: list[ArchiveEntry] = []
    seen_titles: set[str] = set()
    pages_fetched = 0

    for page_num in range(max_pages):
        url = f"{ARCHIVE_URL}?page={page_num}"
        try:
            html = _fetch_page(session, url, timeout=timeout)
        except requests.RequestException:
            break
        entries = _parse_archive_page(html, page_num)
        if not entries:
            break
        page_titles = tuple(e.title for e in entries)
        if page_titles in ({t for t in seen_titles} if seen_titles else set()) or all(
            e.title in seen_titles for e in entries
        ):
            break
        seen_titles.update(page_titles)
        all_entries.extend(entries)
        pages_fetched += 1
        has_next = f"?page={page_num + 1}" in html
        if not has_next:
            break

    return all_entries, pages_fetched


def classify_entry(entry: ArchiveEntry) -> str:
    title = entry.title.strip()
    if not title:
        return "unknown"
    for pattern in REJECT_TITLE_PATTERNS:
        if pattern.search(title):
            return "sample_report"
    for pattern in ACCEPT_TITLE_PATTERNS:
        if pattern.search(title):
            return "production"
    if re.search(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", title):
        return "production"
    return "unknown"


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    if pdfplumber is None:
        return ""
    import io
    text_parts: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
    except Exception:
        return ""
    return "\n".join(text_parts)


def _extract_tables_from_pdf(pdf_bytes: bytes) -> list[list[list[str]]]:
    if pdfplumber is None:
        return []
    import io
    all_tables: list[list[list[str]]] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
    except Exception:
        return []
    return all_tables


def _classify_pdf_content(text: str, title: str) -> str:
    text_lower = text.lower()
    title_lower = title.lower()
    sample_keywords = ("sample", "collection report", "water quality", "suo-moto")
    stp_keywords = ("stp", "sewage treatment", "wastewater", "effluent")
    for kw in sample_keywords:
        if kw in title_lower or kw in text_lower[:500]:
            return "sample_report"
    for kw in stp_keywords:
        if kw in title_lower or kw in text_lower[:500]:
            return "stpw"
    for kw in MGD_KEYWORDS:
        if kw in text_lower:
            return "production"
    return "unknown"


def _parse_date_from_text(text: str) -> date | None:
    patterns = [
        (r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", "ymd"),
        (r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", "dmy4"),
        (r"(\d{1,2})[./-](\d{1,2})[./-](\d{2})\b", "dmy2"),
    ]
    for pat, fmt in patterns:
        m = re.search(pat, text)
        if m:
            groups = m.groups()
            try:
                if fmt == "ymd":
                    return date(int(groups[0]), int(groups[1]), int(groups[2]))
                elif fmt == "dmy4":
                    return date(int(groups[2]), int(groups[1]), int(groups[0]))
                else:
                    year = 2000 + int(groups[2])
                    return date(year, int(groups[1]), int(groups[0]))
            except (ValueError, OverflowError):
                continue
    return None


def _parse_date_range_from_title(title: str) -> tuple[date | None, date | None]:
    date_pattern = r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})"
    matches = re.findall(date_pattern, title)
    dates_list: list[date] = []
    for groups in matches:
        try:
            if len(groups[2]) == 4:
                d = date(int(groups[2]), int(groups[1]), int(groups[0]))
            elif len(groups[0]) == 4:
                d = date(int(groups[0]), int(groups[1]), int(groups[2]))
            else:
                d = date(2000 + int(groups[2]), int(groups[1]), int(groups[0]))
            dates_list.append(d)
        except (ValueError, OverflowError):
            continue
    if len(dates_list) >= 2:
        return min(dates_list), max(dates_list)
    elif len(dates_list) == 1:
        return dates_list[0], dates_list[0]
    return None, None


def _extract_daily_production_from_text(
    text: str,
    source_doc: str,
    source_sha: str,
    archive_title: str,
) -> list[DailyObservation]:
    observations: list[DailyObservation] = []
    text_lower = text.lower()

    has_mgd = "mgd" in text_lower
    has_production = any(kw in text_lower for kw in ("production", "total water", "water supply"))
    if not (has_mgd or has_production):
        return observations

    if any(kw in text_lower for kw in MGD_EXCLUSION_KEYWORDS):
        return observations

    lines = text.split("\n")
    header_line_idx = -1
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if ("total" in line_lower or "production" in line_lower or "mgd" in line_lower) and (
            any(c.isdigit() for c in line)
        ):
            header_line_idx = i
            break

    if header_line_idx == -1:
        for i, line in enumerate(lines):
            if re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", line):
                header_line_idx = i
                break

    if header_line_idx == -1:
        return observations

    for line in lines[header_line_idx:]:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        row_date = _parse_date_from_text(line_stripped)
        if row_date is None:
            continue

        numbers = re.findall(r"\b(\d+\.?\d*)\b", line_stripped)
        if not numbers:
            continue

        total_candidates: list[tuple[float, bool]] = []
        for num_str in numbers:
            try:
                val = float(num_str)
            except ValueError:
                continue
            if MIN_VALID_MGD <= val <= MAX_VALID_MGD:
                total_candidates.append((val, True))
            elif 100 < val < MIN_VALID_MGD:
                total_candidates.append((val, False))

        if not total_candidates:
            continue

        mgd_value = max(total_candidates, key=lambda x: x[0])
        explicit_total = mgd_value[1]

        obs = DailyObservation(
            date=row_date,
            consumption_mgd=mgd_value[0],
            source_document=source_doc,
            source_sha256=source_sha,
            archive_title=archive_title,
            extraction_method="pdf_text_table",
            explicit_total=explicit_total,
            quality_flags=(),
        )
        observations.append(obs)

    return observations


def _extract_daily_production_from_tables(
    tables: list[list[list[str]]],
    source_doc: str,
    source_sha: str,
    archive_title: str,
) -> list[DailyObservation]:
    observations: list[DailyObservation] = []

    for table in tables:
        if not table or len(table) < 2:
            continue

        header = [str(cell).lower().strip() if cell else "" for cell in table[0]]
        date_col = -1
        total_col = -1

        for i, h in enumerate(header):
            if any(kw in h for kw in ("date", "dt", "day")):
                date_col = i
            if any(kw in h for kw in ("total", "mgd", "production", "supply")):
                total_col = i

        if date_col == -1 or total_col == -1:
            continue

        for row in table[1:]:
            if len(row) <= max(date_col, total_col):
                continue
            cell_date = str(row[date_col]).strip()
            cell_total = str(row[total_col]).strip()

            obs_date = _parse_date_from_text(cell_date)
            if obs_date is None:
                continue

            try:
                val = float(cell_total.replace(",", ""))
            except (ValueError, AttributeError):
                continue

            if not (MIN_VALID_MGD <= val <= MAX_VALID_MGD):
                continue

            obs = DailyObservation(
                date=obs_date,
                consumption_mgd=val,
                source_document=source_doc,
                source_sha256=source_sha,
                archive_title=archive_title,
                extraction_method="pdf_structured_table",
                explicit_total=True,
                quality_flags=(),
            )
            observations.append(obs)

    return observations


def _resolve_observations(
    all_observations: list[DailyObservation],
) -> tuple[pd.DataFrame, AuditReport]:
    audit = AuditReport()

    if not all_observations:
        return pd.DataFrame(columns=["Date", "Consumption"]), audit

    by_date: dict[date, list[DailyObservation]] = {}
    for obs in all_observations:
        by_date.setdefault(obs.date, []).append(obs)

    canonical_rows: list[dict] = []
    duplicate_count = 0
    conflict_count = 0

    for obs_date in sorted(by_date.keys()):
        entries = by_date[obs_date]
        unique_values = set(e.consumption_mgd for e in entries)

        if len(unique_values) == 1:
            best = entries[0]
            canonical_rows.append({
                "Date": best.date.isoformat(),
                "Consumption": best.consumption_mgd,
            })
            if len(entries) > 1:
                duplicate_count += 1
        else:
            conflict_count += 1

    audit.duplicate_count = duplicate_count
    audit.conflicting_duplicate_count = conflict_count

    df = pd.DataFrame(canonical_rows)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    return df, audit


def audit_coverage(df: pd.DataFrame, audit: AuditReport) -> AuditReport:
    if df.empty:
        return audit

    dates = df["Date"].dt.date
    earliest = dates.min()
    latest = dates.max()
    calendar_span = (latest - earliest).days + 1

    expected_dates = set()
    current = earliest
    while current <= latest:
        expected_dates.add(current)
        current = date.fromordinal(current.toordinal() + 1)

    actual_dates = set(dates)
    missing = expected_dates - actual_dates
    coverage = (len(actual_dates) / calendar_span * 100) if calendar_span > 0 else 0.0

    longest_block = 0
    if missing:
        sorted_missing = sorted(missing)
        block = 1
        for i in range(1, len(sorted_missing)):
            if (sorted_missing[i] - sorted_missing[i - 1]).days == 1:
                block += 1
            else:
                longest_block = max(longest_block, block)
                block = 1
        longest_block = max(longest_block, block)

    audit.unique_observations = len(df)
    audit.date_range_start = earliest.isoformat()
    audit.date_range_end = latest.isoformat()
    audit.missing_days = len(missing)
    audit.calendar_span_days = calendar_span
    audit.coverage_pct = round(coverage, 2)

    return audit


def run_djb_acquisition(
    work_dir: Path,
    *,
    session: requests.Session | None = None,
    max_pages: int = MAX_PAGES,
    live_network: bool = True,
    timeout: int = 30,
) -> tuple[pd.DataFrame, AuditReport, list[DocumentClassification]]:
    """Run full DJB archive acquisition and extraction pipeline.

    Returns:
        canonical DataFrame (Date,Consumption), audit report, classifications
    """
    if session is None:
        session = requests.Session()

    work_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = work_dir / "documents"
    docs_dir.mkdir(exist_ok=True)

    audit = AuditReport()

    if not live_network:
        return pd.DataFrame(columns=["Date", "Consumption"]), audit, []

    entries, pages = _discover_archive(session, max_pages=max_pages, timeout=timeout)
    audit.pages_discovered = pages
    audit.total_entries_in_archive = len(entries)
    audit.documents_discovered = len(entries)
    audit.entries_with_files = sum(1 for e in entries if e.has_file)

    classifications: list[DocumentClassification] = []
    all_observations: list[DailyObservation] = []

    for entry in entries:
        cat = classify_entry(entry)
        classification = DocumentClassification(entry=entry, category=cat, reason=f"title matched {cat}")
        classifications.append(classification)

        if cat == "sample_report":
            audit.sample_report_count += 1
            continue
        if cat == "unknown" and not entry.has_file:
            audit.unknown_count += 1
            continue

        if not entry.has_file or not entry.pdf_url:
            audit.unknown_count += 1
            classification.category = "unknown"
            classification.reason = "no file available"
            continue

        try:
            resp = session.get(entry.pdf_url, timeout=timeout, headers={
                "User-Agent": "water-forecast-data/1 (academic research)"
            })
            resp.raise_for_status()
            pdf_bytes = resp.content
        except requests.RequestException:
            audit.unknown_count += 1
            classification.reason = "download failed"
            continue

        file_hash = sha256_bytes(pdf_bytes)
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", entry.title)[:80]
        file_path = docs_dir / f"{entry.page}_{entry.row_index}_{safe_name}.pdf"
        file_path.write_bytes(pdf_bytes)

        classification.downloaded = True
        classification.file_path = str(file_path)
        classification.sha256 = file_hash

        text = _extract_text_from_pdf(pdf_bytes)
        tables = _extract_tables_from_pdf(pdf_bytes)

        content_category = _classify_pdf_content(text, entry.title)

        if content_category == "sample_report":
            classification.category = "sample_report"
            classification.reason = "PDF content classified as sample report"
            audit.sample_report_count += 1
            continue
        if content_category == "stpw":
            classification.category = "stpw"
            classification.reason = "PDF content classified as STP/sewage"
            audit.stpw_count += 1
            continue

        text_obs = _extract_daily_production_from_text(
            text, str(file_path), file_hash, entry.title,
        )
        table_obs = _extract_daily_production_from_tables(
            tables, str(file_path), file_hash, entry.title,
        )

        combined = table_obs if table_obs else text_obs
        if combined:
            classification.category = "production"
            classification.reason = f"extracted {len(combined)} daily observations"
            audit.production_doc_count += 1
            all_observations.extend(combined)
        else:
            if content_category == "production":
                classification.category = "production"
                classification.reason = "accepted by title but no parseable data"
                audit.production_doc_count += 1
            else:
                classification.category = "unknown"
                classification.reason = "unable to extract daily production data"
                audit.unknown_count += 1

    canonical_df, resolve_audit = _resolve_observations(all_observations)
    audit.duplicate_count = resolve_audit.duplicate_count
    audit.conflicting_duplicate_count = resolve_audit.conflicting_duplicate_count

    canonical_df = canonical_df[["Date", "Consumption"]]
    audit = audit_coverage(canonical_df, audit)

    audit.documents_accepted = audit.production_doc_count
    audit.documents_rejected = audit.sample_report_count + audit.stpw_count + audit.unknown_count
    audit.documents_unparsed = sum(
        1 for c in classifications
        if c.category == "production" and "no parseable" in c.reason
    )

    return canonical_df, audit, classifications


def write_canonical_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df[["Date", "Consumption"]].to_csv(tmp, index=False, date_format="%Y-%m-%d")
    tmp.replace(path)
    return sha256_file(path)


def write_audit_report(audit: AuditReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pages_discovered": audit.pages_discovered,
        "documents_discovered": audit.documents_discovered,
        "production_doc_count": audit.production_doc_count,
        "sample_report_count": audit.sample_report_count,
        "stpw_count": audit.stpw_count,
        "unknown_count": audit.unknown_count,
        "documents_accepted": audit.documents_accepted,
        "documents_rejected": audit.documents_rejected,
        "documents_unparsed": audit.documents_unparsed,
        "unique_observations": audit.unique_observations,
        "date_range_start": audit.date_range_start,
        "date_range_end": audit.date_range_end,
        "calendar_span_days": audit.calendar_span_days,
        "missing_days": audit.missing_days,
        "coverage_pct": audit.coverage_pct,
        "duplicate_count": audit.duplicate_count,
        "conflicting_duplicate_count": audit.conflicting_duplicate_count,
        "total_entries_in_archive": audit.total_entries_in_archive,
        "entries_with_files": audit.entries_with_files,
        "unit": "MGD",
        "source": "Delhi Jal Board",
        "official_url": ARCHIVE_URL,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
