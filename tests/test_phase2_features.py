import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.phase2_features import build_past_only_features, run_phase2_features, sha256_file


def _daily_frame(periods: int = 800) -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=periods, freq="D")
    return pd.DataFrame({"Date": dates, "Consumption": np.arange(periods, dtype=float) + 100.0})


def test_target_derived_features_are_strictly_past_only() -> None:
    frame = _daily_frame()
    original = build_past_only_features(frame)
    changed = frame.copy()
    changed.loc[500:, "Consumption"] += 1_000_000
    modified = build_past_only_features(changed)
    feature_columns = [column for column in original if column not in {"Date", "Consumption"}]

    pd.testing.assert_frame_equal(
        original.loc[:500, feature_columns], modified.loc[:500, feature_columns]
    )
    assert original.loc[500, "consumption_lag_1"] == frame.loc[499, "Consumption"]
    assert original.loc[500, "rolling_mean_7"] == frame.loc[493:499, "Consumption"].mean()
    assert original.loc[500, "day_over_day_change"] == 1.0
    assert original.loc[500, "growth_7d"] == 7.0 / frame.loc[492, "Consumption"]


def test_exact_calendar_offsets_do_not_bridge_missing_dates() -> None:
    frame = _daily_frame(500).drop(index=[1, 20]).reset_index(drop=True)
    features = build_past_only_features(frame)

    jan_28 = features.index[features["Date"] == pd.Timestamp("2018-01-28")][0]
    jan_2_2019 = features.index[features["Date"] == pd.Timestamp("2019-01-02")][0]
    assert np.isnan(features.loc[jan_28, "same_weekday_previous_week"])
    assert np.isnan(features.loc[jan_2_2019, "same_period_previous_year"])
    assert features.loc[jan_28, "consumption_lag_7"] == frame.loc[jan_28 - 7, "Consumption"]


def test_runner_preserves_source_and_records_no_selection(tmp_path: Path) -> None:
    dataset = tmp_path / "daily.csv"
    _daily_frame().to_csv(dataset, index=False)
    source_hash = sha256_file(dataset)

    output = run_phase2_features(dataset, tmp_path / "artifacts", "test_run")

    assert sha256_file(dataset) == source_hash
    assert {path.name for path in output.iterdir()} == {
        "config.json",
        "feature_manifest.json",
        "features.csv",
    }
    manifest = json.loads((output / "feature_manifest.json").read_text())
    assert manifest["feature_selection"]["performed"] is False
    assert manifest["feature_selection"]["validation_or_test_accessed"] is False
    assert manifest["feature_count"] == 40
    saved = pd.read_csv(output / "features.csv")
    assert len(saved) == 800
    assert saved.loc[365, "consumption_lag_365"] == 100.0
