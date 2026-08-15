import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.phase2_features import build_past_only_features
from experiments.phase3_feature_selection import (
    SelectionConfig,
    leakage_screen,
    load_training_prefix,
    run_phase3_selection,
    select_features,
    sha256_file,
)


def _phase2_artifact(root: Path, rows: int = 240) -> Path:
    phase2 = root / "phase2"
    phase2.mkdir()
    dates = pd.date_range("2018-01-01", periods=rows, freq="D")
    target = 100 + np.sin(np.arange(rows) / 7) * 10 + np.arange(rows) * 0.1
    features = build_past_only_features(pd.DataFrame({"Date": dates, "Consumption": target}))
    features.to_csv(phase2 / "features.csv", index=False, date_format="%Y-%m-%d")
    candidates = [column for column in features if column not in {"Date", "Consumption"}]
    manifest = {
        "date_column": "Date",
        "target_column": "Consumption",
        "feature_columns": candidates,
        "row_count": rows,
        "feature_selection": {"performed": False},
        "causality": {
            "target_derived_features": "past only",
            "rolling_policy": "shifted",
            "calendar_offset_policy": "exact",
            "calendar_features": "known date",
        },
    }
    (phase2 / "feature_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return phase2


def _config() -> SelectionConfig:
    return SelectionConfig(
        n_splits=2,
        n_estimators=12,
        permutation_repeats=2,
        max_features=10,
        stability_top_k=8,
    )


def test_training_loader_never_reads_beyond_training_prefix(tmp_path: Path, monkeypatch) -> None:
    phase2 = _phase2_artifact(tmp_path, rows=100)
    real_read_csv = pd.read_csv
    calls = []

    def guarded_read_csv(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    frame, _, training_rows, _ = load_training_prefix(phase2, _config())

    assert training_rows == 70
    assert len(frame) == 70
    assert calls == [{"nrows": 70, "usecols": list(real_read_csv(phase2 / "features.csv", nrows=0).columns)}]


def test_selection_is_unchanged_when_reserved_rows_change(tmp_path: Path) -> None:
    phase2 = _phase2_artifact(tmp_path)
    config = _config()
    training, manifest, training_rows, _ = load_training_prefix(phase2, config)
    first_ranking, first_report = select_features(training, manifest, config)

    full = pd.read_csv(phase2 / "features.csv")
    full.loc[training_rows:, "Consumption"] += 1_000_000
    full.loc[training_rows:, "consumption_lag_1"] -= 1_000_000
    full.to_csv(phase2 / "features.csv", index=False)
    changed_training, changed_manifest, _, _ = load_training_prefix(phase2, config)
    second_ranking, second_report = select_features(changed_training, changed_manifest, config)

    pd.testing.assert_frame_equal(first_ranking, second_ranking)
    assert first_report["selected_features"] == second_report["selected_features"]


def test_leakage_screen_rejects_exact_target_copy(tmp_path: Path) -> None:
    phase2 = _phase2_artifact(tmp_path)
    training, manifest, _, _ = load_training_prefix(phase2, _config())
    training["leaky_copy"] = training["Consumption"]
    candidates = [*manifest["feature_columns"], "leaky_copy"]

    with pytest.raises(ValueError, match="exact_target_matches"):
        leakage_screen(training, candidates, "Consumption")


def test_runner_preserves_input_and_writes_auditable_artifacts(tmp_path: Path) -> None:
    phase2 = _phase2_artifact(tmp_path)
    source_hash = sha256_file(phase2 / "features.csv")
    output = run_phase3_selection(phase2, tmp_path / "artifacts", "test_run", _config())

    assert sha256_file(phase2 / "features.csv") == source_hash
    assert {path.name for path in output.iterdir()} == {
        "config.json",
        "feature_ranking.csv",
        "selection_report.json",
    }
    report = json.loads((output / "selection_report.json").read_text())
    assert report["training_rows"] == 168
    assert report["validation_rows_loaded"] == 0
    assert report["locked_test_rows_loaded"] == 0
    assert 0 < report["selected_feature_count"] <= 10
    redundancy_map = report["redundancy"]["dropped_feature_to_retained_feature"]
    assert not set(redundancy_map.values()).intersection(redundancy_map)
    assert all(
        split["fit_end_row_inclusive"] < split["assessment_start_row"]
        for split in report["time_splits"]
    )
