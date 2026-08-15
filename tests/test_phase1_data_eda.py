import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.phase1_data_eda import run_phase1_eda, sha256_file, validate_daily_consumption


def test_validation_reports_quality_issues_without_reordering(tmp_path: Path) -> None:
    dataset = tmp_path / "quality.csv"
    dataset.write_text(
        "Date,Consumption\n"
        "2024-01-01,10\n"
        "2024-01-03,10\n"
        "bad-date,-2\n"
        "2024-01-03,nope\n",
        encoding="utf-8",
    )

    frame, report = validate_daily_consumption(dataset)

    assert frame.loc[0, "Date"] == pd.Timestamp("2024-01-01")
    assert frame.loc[1, "Date"] == pd.Timestamp("2024-01-03")
    assert report["invalid_values"]["invalid_date_count"] == 1
    assert report["invalid_values"]["non_numeric_consumption_count"] == 1
    assert report["invalid_values"]["non_positive_consumption_count"] == 1
    assert report["duplicates"]["duplicate_date_count"] == 2
    assert report["chronology"]["gap_count"] == 1
    assert report["chronology"]["missing_calendar_day_count"] == 1


def test_phase1_runner_preserves_source_and_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "daily.csv"
    dates = pd.date_range("2022-01-01", periods=500, freq="D")
    values = 100 + 8 * np.sin(np.arange(500) * 2 * np.pi / 7)
    pd.DataFrame({"Date": dates, "Consumption": values}).to_csv(dataset, index=False)
    original_hash = sha256_file(dataset)

    output = run_phase1_eda(dataset, tmp_path / "artifacts", "test_run")

    assert sha256_file(dataset) == original_hash
    assert {path.name for path in output.iterdir()} == {
        "calendar_seasonality.png",
        "calendar_summary.csv",
        "config.json",
        "data_quality_report.json",
        "descriptive_statistics.csv",
        "suspicious_observations.csv",
        "target_distribution_boxplot.png",
        "time_series_trend.png",
    }
    report = json.loads((output / "data_quality_report.json").read_text())
    assert report["shape"] == {"rows": 500, "columns": 2}
    assert report["chronology"]["strictly_increasing"] is True
    assert report["chronology"]["gap_count"] == 0
    assert report["source_preservation"]["unchanged"] is True
    assert report["correlation_analysis"]["performed"] is False
    config = json.loads((output / "config.json").read_text())
    assert config["random_seed"] is None
    assert config["chronology_policy"].startswith("preserve input row order")
