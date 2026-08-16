"""Comprehensive mocked tests for Delhi DJB source adapter.

All tests use mocked network and PDF content.
No live-network dependency.
"""
from __future__ import annotations

import io
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from orchestration.data.sources.delhi_djb import (
    ARCHIVE_BASE,
    ArchiveEntry,
    AuditReport,
    DailyObservation,
    DocumentClassification,
    _classify_pdf_content,
    _discover_archive,
    _extract_daily_production_from_tables,
    _extract_daily_production_from_text,
    _extract_tables_from_pdf,
    _extract_text_from_pdf,
    _parse_archive_page,
    _parse_date_from_text,
    _parse_date_range_from_title,
    _resolve_observations,
    audit_coverage,
    classify_entry,
    run_djb_acquisition,
    sha256_bytes,
    write_audit_report,
    write_canonical_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    *,
    title: str = "01.01.2022 to 05.01.2022",
    date_text: str = "01-01-2022",
    pdf_url: str | None = f"{ARCHIVE_BASE}/sites/default/files/test.pdf",
    page: int = 0,
    row_index: int = 0,
) -> ArchiveEntry:
    return ArchiveEntry(
        page=page,
        row_index=row_index,
        title=title,
        date_text=date_text,
        pdf_url=pdf_url,
        has_file=pdf_url is not None,
    )


SAMPLE_HTML_PAGE1 = """<html><body>
<table class="views-table">
<tbody>
<tr>
<td>1</td>
<td>Water samples collection reports on 01.03.2024 to 10.03.2024</td>
<td>11-03-2024</td>
<td><a href="/sites/default/files/sample.pdf">PDF</a></td>
</tr>
<tr>
<td>2</td>
<td>01.01.2022 to 05.01.2022</td>
<td></td>
<td><a href="/sites/default/files/prod1.pdf">PDF</a></td>
</tr>
</tbody>
</table>
<nav class="pager"><a href="?page=1" class="page-link">&gt;&gt;</a></nav>
</body></html>"""

SAMPLE_HTML_PAGE2 = """<html><body>
<table class="views-table">
<tbody>
<tr>
<td>3</td>
<td>06.01.2022 to 10.01.2022</td>
<td></td>
<td><a href="/sites/default/files/prod2.pdf">PDF</a></td>
</tr>
</tbody>
</table>
</body></html>"""

SAMPLE_HTML_EMPTY = """<html><body>
<table class="views-table"><tbody></tbody></table>
</body></html>"""


def _make_session(pages: dict[int, str] | None = None) -> MagicMock:
    session = MagicMock()
    page_map = pages or {0: SAMPLE_HTML_PAGE1, 1: SAMPLE_HTML_PAGE2}

    def get(url: str, **kwargs: Any) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "page=" in url:
            page_num = int(url.split("page=")[-1])
            resp.text = page_map.get(page_num, SAMPLE_HTML_EMPTY)
        else:
            resp.text = page_map.get(0, SAMPLE_HTML_PAGE1)
        resp.content = b"%PDF-1.4 fake content"
        return resp

    session.get.side_effect = get
    return session


# ---------------------------------------------------------------------------
# Tests: Archive parsing
# ---------------------------------------------------------------------------

class TestArchiveParsing:
    def test_parse_archive_page_extracts_entries(self) -> None:
        entries = _parse_archive_page(SAMPLE_HTML_PAGE1, page_number=0)
        assert len(entries) == 2
        assert entries[0].title.startswith("Water samples")
        assert entries[0].has_file is True
        assert entries[0].pdf_url is not None

    def test_parse_archive_page_empty_table(self) -> None:
        entries = _parse_archive_page(SAMPLE_HTML_EMPTY, page_number=0)
        assert entries == []

    def test_discover_archive_stops_on_empty_page(self) -> None:
        session = _make_session({0: SAMPLE_HTML_EMPTY})
        entries, pages = _discover_archive(session, max_pages=5)
        assert pages == 0
        assert entries == []

    def test_discover_archive_stops_on_duplicate_page(self) -> None:
        session = _make_session({0: SAMPLE_HTML_PAGE1, 1: SAMPLE_HTML_PAGE1})
        entries, pages = _discover_archive(session, max_pages=5)
        assert pages == 1

    def test_discover_archive_follows_next_page(self) -> None:
        session = _make_session()
        entries, pages = _discover_archive(session, max_pages=5)
        assert pages == 2
        assert len(entries) == 3


# ---------------------------------------------------------------------------
# Tests: Classification
# ---------------------------------------------------------------------------

class TestClassification:
    @pytest.mark.parametrize("title,expected", [
        ("Water samples collection reports on 01.03.2024", "sample_report"),
        ("Drinking Water Quality report", "sample_report"),
        ("STP daily report", "sample_report"),
        ("Sewage treatment summary", "sample_report"),
        ("Wastewater analysis", "sample_report"),
        ("Daily Water Production Report", "production"),
        ("Water production 01.01.2022 to 05.01.2022", "production"),
        ("01.01.2022 to 05.01.2022", "production"),
        ("Total Water Production 10.03.2024", "production"),
        ("random nonsense", "unknown"),
    ])
    def test_classify_entry_title_patterns(self, title: str, expected: str) -> None:
        entry = _make_entry(title=title)
        assert classify_entry(entry) == expected

    def test_classify_entry_no_file_is_unknown(self) -> None:
        entry = _make_entry(title="01.01.2022 to 05.01.2022", pdf_url=None)
        assert classify_entry(entry) == "production"

    def test_classify_pdf_content_production(self) -> None:
        text = "Total Water Production in MGD: 1050.5"
        assert _classify_pdf_content(text, "01.01.2022 to 05.01.2022") == "production"

    def test_classify_pdf_content_sample_report(self) -> None:
        text = "Water sample collection report from zone A"
        assert _classify_pdf_content(text, "report") == "sample_report"

    def test_classify_pdf_content_stpw(self) -> None:
        text = "STP Daily Report - Sewage Treatment Plant"
        assert _classify_pdf_content(text, "report") == "stpw"


# ---------------------------------------------------------------------------
# Tests: Date parsing
# ---------------------------------------------------------------------------

class TestDateParsing:
    def test_parse_date_dd_mm_yyyy(self) -> None:
        assert _parse_date_from_text("15-03-2022") == date(2022, 3, 15)

    def test_parse_date_dd_mm_yy(self) -> None:
        assert _parse_date_from_text("15.03.22") == date(2022, 3, 15)

    def test_parse_date_yyyy_mm_dd(self) -> None:
        assert _parse_date_from_text("2022-03-15") == date(2022, 3, 15)

    def test_parse_date_returns_none_for_no_date(self) -> None:
        assert _parse_date_from_text("no date here") is None

    def test_parse_date_range_from_title(self) -> None:
        start, end = _parse_date_range_from_title("01.01.2022 to 05.01.2022")
        assert start == date(2022, 1, 1)
        assert end == date(2022, 1, 5)

    def test_parse_date_range_single_date(self) -> None:
        start, end = _parse_date_range_from_title("15.03.2022")
        assert start == date(2022, 3, 15)
        assert end == date(2022, 3, 15)

    def test_parse_date_range_no_match(self) -> None:
        start, end = _parse_date_range_from_title("no dates here")
        assert start is None
        assert end is None


# ---------------------------------------------------------------------------
# Tests: PDF extraction
# ---------------------------------------------------------------------------

class TestPDFExtraction:
    def test_extract_text_returns_string(self) -> None:
        with patch("orchestration.data.sources.delhi_djb.pdfplumber") as mock_pdf:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Total production MGD 1050"
            mock_pdf.open.return_value.__enter__ = MagicMock(return_value=MagicMock(pages=[mock_page]))
            mock_pdf.open.return_value.__exit__ = MagicMock(return_value=False)
            result = _extract_text_from_pdf(b"fake pdf bytes")
            assert isinstance(result, str)

    def test_extract_text_handles_import_error(self) -> None:
        import orchestration.data.sources.delhi_djb as mod
        original = mod.pdfplumber
        try:
            mod.pdfplumber = None
            result = _extract_text_from_pdf(b"fake pdf bytes")
            assert result == ""
        finally:
            mod.pdfplumber = original

    def test_extract_tables_returns_list(self) -> None:
        tables = _extract_tables_from_pdf(b"not a real pdf")
        assert isinstance(tables, list)

    def test_extract_daily_production_from_text_valid(self) -> None:
        text = """Daily Water Production Report
Date\t\tTotal (MGD)
01-01-2022\t\t1050.5
02-01-2022\t\t1062.3
03-01-2022\t\t1048.1"""
        obs = _extract_daily_production_from_text(
            text, "doc.pdf", "abc123", "01.01.2022 to 03.01.2022"
        )
        assert len(obs) >= 1
        for o in obs:
            assert 100 < o.consumption_mgd < 2000
            assert o.extraction_method == "pdf_text_table"

    def test_extract_daily_production_returns_empty_for_no_mgd(self) -> None:
        text = "This is a random document with no production data at all."
        obs = _extract_daily_production_from_text(
            text, "doc.pdf", "abc123", "random doc"
        )
        assert obs == []

    def test_extract_daily_production_rejects_quality_keywords(self) -> None:
        text = "Water quality sample test results MGD 1050"
        obs = _extract_daily_production_from_text(
            text, "doc.pdf", "abc123", "quality report"
        )
        assert obs == []

    def test_extract_from_tables_with_header(self) -> None:
        tables = [[
            ["Date", "Total Water Production (MGD)", "Remarks"],
            ["01-01-2022", "1050.5", "Normal"],
            ["02-01-2022", "1062.3", "Normal"],
        ]]
        obs = _extract_daily_production_from_tables(
            tables, "doc.pdf", "abc123", "production report"
        )
        assert len(obs) == 2
        assert obs[0].consumption_mgd == 1050.5
        assert obs[1].consumption_mgd == 1062.3


# ---------------------------------------------------------------------------
# Tests: Observation resolution
# ---------------------------------------------------------------------------

class TestObservationResolution:
    def test_resolve_single_day(self) -> None:
        obs = [
            DailyObservation(
                date=date(2022, 1, 1), consumption_mgd=1050.0,
                source_document="a.pdf", source_sha256="sha_a",
                archive_title="doc A", extraction_method="text",
                explicit_total=True,
            ),
        ]
        df, audit = _resolve_observations(obs)
        assert len(df) == 1
        assert audit.duplicate_count == 0

    def test_resolve_duplicate_same_value(self) -> None:
        obs = [
            DailyObservation(date=date(2022, 1, 1), consumption_mgd=1050.0,
                             source_document="a.pdf", source_sha256="sha_a",
                             archive_title="doc A", extraction_method="text", explicit_total=True),
            DailyObservation(date=date(2022, 1, 1), consumption_mgd=1050.0,
                             source_document="b.pdf", source_sha256="sha_b",
                             archive_title="doc B", extraction_method="text", explicit_total=True),
        ]
        df, audit = _resolve_observations(obs)
        assert len(df) == 1
        assert audit.duplicate_count == 1
        assert audit.conflicting_duplicate_count == 0

    def test_resolve_duplicate_conflicting_values_excluded(self) -> None:
        obs = [
            DailyObservation(date=date(2022, 1, 1), consumption_mgd=1050.0,
                             source_document="a.pdf", source_sha256="sha_a",
                             archive_title="doc A", extraction_method="text", explicit_total=True),
            DailyObservation(date=date(2022, 1, 1), consumption_mgd=1100.0,
                             source_document="b.pdf", source_sha256="sha_b",
                             archive_title="doc B", extraction_method="text", explicit_total=True),
        ]
        df, audit = _resolve_observations(obs)
        assert len(df) == 0
        assert audit.conflicting_duplicate_count == 1

    def test_resolve_empty(self) -> None:
        df, audit = _resolve_observations([])
        assert len(df) == 0
        assert audit.unique_observations == 0


# ---------------------------------------------------------------------------
# Tests: Coverage audit
# ---------------------------------------------------------------------------

class TestCoverageAudit:
    def test_audit_full_coverage(self) -> None:
        df = pd.DataFrame({
            "Date": pd.date_range("2022-01-01", periods=5),
            "Consumption": [1050.0] * 5,
        })
        audit = AuditReport()
        audit = audit_coverage(df, audit)
        assert audit.missing_days == 0
        assert audit.coverage_pct == 100.0
        assert audit.calendar_span_days == 5
        assert audit.unique_observations == 5

    def test_audit_partial_coverage(self) -> None:
        df = pd.DataFrame({
            "Date": ["2022-01-01", "2022-01-03", "2022-01-05"],
            "Consumption": [1050.0, 1050.0, 1050.0],
        })
        df["Date"] = pd.to_datetime(df["Date"])
        audit = AuditReport()
        audit = audit_coverage(df, audit)
        assert audit.missing_days == 2
        assert audit.coverage_pct == pytest.approx(60.0, abs=0.1)

    def test_audit_empty_df(self) -> None:
        df = pd.DataFrame(columns=["Date", "Consumption"])
        audit = AuditReport()
        audit = audit_coverage(df, audit)
        assert audit.unique_observations == 0


# ---------------------------------------------------------------------------
# Tests: Write canonical CSV
# ---------------------------------------------------------------------------

class TestWriteCanonicalCSV:
    def test_write_csv_creates_file(self, tmp_path: Path) -> None:
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2022-01-01", "2022-01-02"]),
            "Consumption": [1050.0, 1060.0],
        })
        path = tmp_path / "canonical" / "water_demand.csv"
        file_hash = write_canonical_csv(df, path)
        assert path.exists()
        assert len(file_hash) == 64
        loaded = pd.read_csv(path)
        assert list(loaded.columns) == ["Date", "Consumption"]
        assert len(loaded) == 2

    def test_write_audit_report(self, tmp_path: Path) -> None:
        audit = AuditReport(
            pages_discovered=13,
            documents_discovered=130,
            production_doc_count=100,
            documents_accepted=100,
            documents_rejected=30,
            unique_observations=900,
            date_range_start="2021-01-01",
            date_range_end="2025-03-10",
            coverage_pct=95.5,
        )
        path = tmp_path / "audit.json"
        write_audit_report(audit, path)
        assert path.exists()
        payload = json.loads(path.read_text())
        assert payload["pages_discovered"] == 13
        assert payload["unit"] == "MGD"


# ---------------------------------------------------------------------------
# Tests: End-to-end mocked acquisition
# ---------------------------------------------------------------------------

class TestEndToEndAcquisition:
    def test_run_djb_acquisition_live_network_false(self, tmp_path: Path) -> None:
        df, audit, classifications = run_djb_acquisition(
            tmp_path, live_network=False
        )
        assert len(df) == 0
        assert audit.documents_discovered == 0
        assert classifications == []

    def test_run_djb_acquisition_with_mocked_session(self, tmp_path: Path) -> None:
        page_map = {0: SAMPLE_HTML_PAGE1, 1: SAMPLE_HTML_EMPTY}

        def mock_get(url: str, **kwargs: Any) -> MagicMock:
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if ".pdf" in url:
                resp.content = b"%PDF-1.4 fake content"
            elif "page=" in url:
                page_num = int(url.split("page=")[-1])
                text = page_map.get(page_num, SAMPLE_HTML_EMPTY)
                resp.text = text
                resp.content = text.encode()
            else:
                resp.text = page_map.get(0, SAMPLE_HTML_PAGE1)
                resp.content = resp.text.encode()
            return resp

        session2 = MagicMock()
        session2.get.side_effect = mock_get

        df, audit, classifications = run_djb_acquisition(
            tmp_path, session=session2, live_network=True
        )
        assert isinstance(df, pd.DataFrame)
        assert isinstance(classifications, list)
        assert audit.documents_discovered >= 2

    def test_canonical_columns_are_correct(self, tmp_path: Path) -> None:
        df, _, _ = run_djb_acquisition(tmp_path, live_network=False)
        assert list(df.columns) == ["Date", "Consumption"]


# ---------------------------------------------------------------------------
# Tests: Hashing
# ---------------------------------------------------------------------------

class TestHashing:
    def test_sha256_bytes_deterministic(self) -> None:
        data = b"hello world"
        h1 = sha256_bytes(data)
        h2 = sha256_bytes(data)
        assert h1 == h2
        assert len(h1) == 64

    def test_sha256_bytes_different_inputs(self) -> None:
        assert sha256_bytes(b"hello") != sha256_bytes(b"world")
