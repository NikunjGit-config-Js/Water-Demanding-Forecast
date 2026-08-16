from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import orchestration.data.acquisition as acquisition_module
from orchestration.data.acquisition import acquire, selenium_download
from orchestration.data.compatibility import MINIMUM_OBSERVATIONS, evaluate_compatibility
from orchestration.data.normalize import normalize
from orchestration.data.provenance import ProvenanceManifest, sha256_file
from orchestration.data.registry import CitySource, get_city_source, supported_cities, validate_city_slug
from orchestration.data.status import DataStatus
from orchestration.data.pipeline import prepare_city_data
from orchestration.comparison import FIELDS, build_comparison_row, write_multi_city_comparison
from orchestration.context import RunContext


def source(**changes: object) -> CitySource:
    values = dict(city="bengaluru", configured=True, source_name="Test official source",
                  source_url="https://example.invalid/data.csv", official=True, format="csv",
                  expected_frequency="daily", date_column="when", consumption_column="value",
                  unit="ML", canonical_unit="ML", aggregation=None, adapter_type="direct")
    values.update(changes)
    return CitySource(**values)


def test_registry_contains_supported_cities_without_invented_india_sources() -> None:
    assert supported_cities() == ("london", "bengaluru", "delhi", "gurgaon", "hyderabad", "pune")
    for city in supported_cities()[1:]:
        entry = get_city_source(city)
        assert not entry.configured and entry.source_url is None and entry.source_name is None
    assert get_city_source("london").source_url is None


@pytest.mark.parametrize("slug", ["Bengaluru", "../delhi", "delhi_1", "", "-pune"])
def test_invalid_city_slugs_are_rejected(slug: str) -> None:
    with pytest.raises(ValueError): validate_city_slug(slug)


def test_missing_and_unknown_sources_fail_cleanly(tmp_path: Path) -> None:
    assert acquire(get_city_source("delhi"), tmp_path).status == DataStatus.DATA_SOURCE_REQUIRED
    with pytest.raises(KeyError, match="unknown city"): get_city_source("mumbai")


def test_canonical_conversion_sorting_duplicate_and_ward_aggregation(tmp_path: Path) -> None:
    frame = pd.DataFrame({"when": ["2024-01-02", "2024-01-01", "2024-01-01"],
                          "ward": ["A", "A", "B"], "value": [5, 2, 3]})
    result = normalize(frame, source(zone_column="ward", aggregation="sum"), tmp_path / "canonical.csv")
    assert result.status == DataStatus.READY and result.wards_aggregated
    assert result.frame.columns.tolist() == ["Date", "Consumption"]
    assert result.frame["Consumption"].tolist() == [5, 5]
    assert result.frame["Date"].is_monotonic_increasing


def test_duplicate_dates_without_declared_semantics_are_rejected(tmp_path: Path) -> None:
    frame = pd.DataFrame({"when": ["2024-01-01"] * 2, "value": [1, 2]})
    result = normalize(frame, source(), tmp_path / "out.csv")
    assert result.status == DataStatus.DATA_INCOMPATIBLE
    assert not (tmp_path / "out.csv").exists()


def test_genuine_subdaily_sum_and_unit_conversion(tmp_path: Path) -> None:
    frame = pd.DataFrame({"when": ["2024-01-01 00:00", "2024-01-01 12:00", "2024-01-02 00:00"],
                          "value": [1000, 2000, 4000]})
    spec = source(expected_frequency="hourly", aggregation="sum", unit="kL",
                  canonical_unit="ML", unit_multiplier=.001)
    result = normalize(frame, spec, tmp_path / "out.csv")
    assert result.status == DataStatus.READY
    assert result.frame["Consumption"].tolist() == [3.0, 4.0]


@pytest.mark.parametrize("frequency", ["monthly", "quarterly", "annual", "irregular"])
def test_summary_frequencies_are_rejected_without_synthetic_daily_rows(tmp_path: Path, frequency: str) -> None:
    frame = pd.DataFrame({"when": ["2024-01-01", "2024-02-01"], "value": [31, 29]})
    result = normalize(frame, source(expected_frequency=frequency), tmp_path / "out.csv")
    assert result.status == DataStatus.DATA_INCOMPATIBLE
    assert "cannot be expanded" in (result.reason or "")
    assert not (tmp_path / "out.csv").exists()


def test_invalid_values_are_rejected(tmp_path: Path) -> None:
    frame = pd.DataFrame({"when": ["bad", "2024-01-01"], "value": [np.inf, 2]})
    assert normalize(frame, source(), tmp_path / "out.csv").status == DataStatus.DATA_INCOMPATIBLE


class Response(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *args: object) -> None: self.close()


def test_direct_acquisition_records_hash_without_network(tmp_path: Path) -> None:
    result = acquire(source(), tmp_path, opener=lambda _: Response(b"when,value\n2024-01-01,1\n"))
    assert result.status == DataStatus.READY and result.method == "direct_download"
    assert result.raw_path and result.sha256 == sha256_file(result.raw_path)
    assert result.downloaded_at_utc and result.source_url == "https://example.invalid/data.csv"


def test_mocked_selenium_is_fallback_and_stays_in_raw_dir(tmp_path: Path) -> None:
    calls: list[str] = []
    def selenium(url: str, raw: Path) -> Path:
        calls.append(url); path = raw / "browser.csv"; path.write_bytes(b"x\n1\n"); return path
    result = acquire(source(), tmp_path, opener=lambda _: (_ for _ in ()).throw(OSError()),
                     selenium_fetcher=selenium)
    assert result.status == DataStatus.READY and result.method == "selenium_fallback" and len(calls) == 1


def test_concrete_selenium_is_headless_waited_confined_and_quits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}
    class Options:
        def __init__(self): self.arguments = []; self.experimental_options = {}
        def add_argument(self, value: str) -> None: self.arguments.append(value)
        def add_experimental_option(self, name: str, value: object) -> None:
            self.experimental_options[name] = value
    webdriver = types.ModuleType("selenium.webdriver"); webdriver.ChromeOptions = Options
    selenium = types.ModuleType("selenium"); selenium.webdriver = webdriver
    support = types.ModuleType("selenium.webdriver.support")
    ui = types.ModuleType("selenium.webdriver.support.ui"); ui.WebDriverWait = object
    monkeypatch.setitem(sys.modules, "selenium", selenium)
    monkeypatch.setitem(sys.modules, "selenium.webdriver", webdriver)
    monkeypatch.setitem(sys.modules, "selenium.webdriver.support", support)
    monkeypatch.setitem(sys.modules, "selenium.webdriver.support.ui", ui)
    class Driver:
        def get(self, url: str) -> None:
            observed["url"] = url
            (tmp_path / "download.csv").write_text("x\n1\n")
        def quit(self) -> None: observed["quit"] = True
    class Waiter:
        def __init__(self, driver: object, timeout: float): observed["timeout"] = timeout
        def until(self, predicate): return predicate(None)
    def factory(options: object) -> Driver:
        observed["arguments"] = tuple(options.arguments)  # type: ignore[attr-defined]
        observed["prefs"] = options.experimental_options["prefs"]  # type: ignore[attr-defined]
        return Driver()
    result = selenium_download("https://example.invalid/file", tmp_path, timeout_seconds=7,
                               driver_factory=factory, waiter_factory=Waiter)
    assert result == (tmp_path / "download.csv").resolve()
    assert "--headless=new" in observed["arguments"]
    assert observed["prefs"]["download.default_directory"] == str(tmp_path.resolve())
    assert observed["timeout"] == 7 and observed["quit"] is True


def test_missing_browser_dependency_fails_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(url: str, raw: Path) -> Path:
        raise RuntimeError("Selenium dependency is unavailable")
    monkeypatch.setattr(acquisition_module, "selenium_download", unavailable)
    result = acquire(source(), tmp_path, opener=lambda _: (_ for _ in ()).throw(OSError()),
                     allow_selenium=True)
    assert result.status == DataStatus.ACQUISITION_FAILED
    assert result.reason == "Selenium fallback failed: RuntimeError"


def _provenance(dataset: Path, rows: int, *, unit: str | None = "ML") -> ProvenanceManifest:
    return ProvenanceManifest("bengaluru", "Test", "https://example.invalid/data.csv", True,
        "direct_download", "2026-01-01T00:00:00+00:00", "csv", "raw.csv", "0" * 64,
        str(dataset), sha256_file(dataset), "ML", unit, "csv", "daily", (), False, rows,
        "2020-01-01", "2021-06-06")


def test_provenance_roundtrip_and_hash(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"; dataset.write_text("Date,Consumption\n2024-01-01,1\n")
    manifest = _provenance(dataset, 1); path = tmp_path / "provenance.json"; manifest.write(path)
    assert ProvenanceManifest.read(path) == manifest
    assert json.loads(path.read_text())["canonical_sha256"] == sha256_file(dataset)


def test_compatibility_minimum_and_capabilities(tmp_path: Path) -> None:
    dataset = tmp_path / "canonical.csv"
    pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=MINIMUM_OBSERVATIONS),
                  "Consumption": np.arange(MINIMUM_OBSERVATIONS)}).to_csv(dataset, index=False)
    report = evaluate_compatibility("bengaluru", dataset, _provenance(dataset, MINIMUM_OBSERVATIONS))
    assert report.status == DataStatus.READY
    assert all(report.capabilities.values()) and report.minimum_observations == 523


def test_insufficient_history_and_missing_provenance_stop_before_ml(tmp_path: Path) -> None:
    dataset = tmp_path / "canonical.csv"
    pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=100),
                  "Consumption": np.arange(100)}).to_csv(dataset, index=False)
    report = evaluate_compatibility("delhi", dataset, None)
    assert report.status == DataStatus.DATA_INCOMPATIBLE
    assert not report.checks["sufficient_observations"] and not report.checks["provenance_present"]
    assert not report.capabilities["lag_365"]


def test_hash_unit_and_chronology_gate(tmp_path: Path) -> None:
    dataset = tmp_path / "canonical.csv"
    pd.DataFrame({"Date": ["2024-01-02", "2024-01-01"], "Consumption": [1, 2]}).to_csv(dataset, index=False)
    provenance = _provenance(dataset, 2, unit=None)
    dataset.write_text(dataset.read_text() + "2024-01-03,3\n")
    report = evaluate_compatibility("pune", dataset, provenance)
    assert not report.checks["chronological_unique"]
    assert not report.checks["dataset_hash_valid"]
    assert not report.checks["source_unit_recorded"]


def test_configured_source_mocked_end_to_end_is_ready(tmp_path: Path) -> None:
    rows = MINIMUM_OBSERVATIONS
    csv = pd.DataFrame({"when": pd.date_range("2020-01-01", periods=rows),
                        "value": np.arange(rows, dtype=float)}).to_csv(index=False).encode()
    spec = source()
    acquired = acquire(spec, tmp_path / "raw", opener=lambda _: Response(csv))
    assert acquired.status == DataStatus.READY and acquired.raw_path
    adapted = pd.read_csv(acquired.raw_path)
    normalized = normalize(adapted, spec, tmp_path / "canonical" / "data.csv")
    assert normalized.status == DataStatus.READY and normalized.canonical_path
    manifest = ProvenanceManifest(
        spec.city, spec.source_name or "", spec.source_url, spec.official,
        acquired.method or "", acquired.downloaded_at_utc, spec.format or "",
        str(acquired.raw_path), acquired.sha256 or "", str(normalized.canonical_path),
        normalized.canonical_sha256 or "", spec.unit, spec.canonical_unit,
        spec.adapter_type or "", spec.expected_frequency or "", normalized.transformations,
        normalized.wards_aggregated, rows, "2020-01-01", "2021-06-06")
    report = evaluate_compatibility(spec.city, normalized.canonical_path, manifest)
    assert report.status == DataStatus.READY and all(report.checks.values())


def test_unconfigured_preflight_writes_machine_readable_gate_and_stops(tmp_path: Path) -> None:
    report = prepare_city_data("delhi", tmp_path)
    path = tmp_path / "canonical" / "city_data_compatibility.json"
    assert report.status == DataStatus.DATA_SOURCE_REQUIRED
    assert json.loads(path.read_text())["status"] == "DATA_SOURCE_REQUIRED"
    assert not (tmp_path / "raw").exists()


def test_preflight_never_calls_acquirer_for_unconfigured_source(tmp_path: Path) -> None:
    def forbidden(*_: object) -> object:
        raise AssertionError("network acquisition must not be attempted")
    report = prepare_city_data("pune", tmp_path, acquirer=forbidden)  # type: ignore[arg-type]
    assert report.status == DataStatus.DATA_SOURCE_REQUIRED


def test_comparison_preserves_missing_metrics_instead_of_fabricating(tmp_path: Path) -> None:
    csv_path, json_path = write_multi_city_comparison(
        [{"city": "london", "validator_status": "PASS"}], tmp_path
    )
    payload = json.loads(json_path.read_text())
    assert tuple(payload[0]) == tuple(sorted(FIELDS))
    assert payload[0]["ml_mae"] is None
    assert "ml_mae" in csv_path.read_text()


def test_comparison_extracts_only_available_phase_evidence() -> None:
    row = build_comparison_row(RunContext.for_city("london"), "PASS")
    assert row["rows"] == 3800
    assert row["best_naive_baseline"] == "naive_lag_1"
    assert row["best_conventional_ml"] == "linear_regression"
    assert row["best_time_series_model"] == "prophet"
    assert row["modern_model"] == "patchtst"
    assert row["unit"] is None
