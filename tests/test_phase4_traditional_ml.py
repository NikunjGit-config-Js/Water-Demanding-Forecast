import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from experiments.phase2_features import build_past_only_features
from experiments.phase3_feature_selection import SelectionConfig, run_phase3_selection
from experiments.phase4_traditional_ml import (
    Phase4Config,
    SCALED_MODELS,
    build_model_registry,
    load_selected_training_prefix,
    run_phase4_models,
    sha256_file,
)


def _inputs(root: Path, rows: int = 180) -> tuple[Path, Path]:
    phase2 = root / "phase2"
    phase2.mkdir()
    dates = pd.date_range("2019-01-01", periods=rows, freq="D")
    target = 100 + 8 * np.sin(np.arange(rows) * 2 * np.pi / 7) + np.arange(rows) * 0.05
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
    phase3 = run_phase3_selection(
        phase2,
        root / "phase3-artifacts",
        "selection",
        SelectionConfig(
            n_splits=2, n_estimators=8, permutation_repeats=1, max_features=8, stability_top_k=6
        ),
    )
    return phase2, phase3


def _fast_config() -> Phase4Config:
    return Phase4Config(
        random_forest_estimators=5,
        bagging_estimators=5,
        gradient_boosting_estimators=5,
        knn_neighbors=3,
    )


def test_registry_contains_only_approved_models_and_scales_required_families() -> None:
    registry = build_model_registry(_fast_config())
    assert set(registry) == {
        "linear_regression",
        "ridge",
        "lasso",
        "decision_tree",
        "knn",
        "svr",
        "random_forest",
        "bagging",
        "gradient_boosting",
        "voting",
    }
    for name, pipeline in registry.items():
        assert isinstance(pipeline, Pipeline)
        assert ("scaler" in pipeline.named_steps) == (name in SCALED_MODELS)


def test_loader_reads_no_reserved_rows(tmp_path: Path, monkeypatch) -> None:
    phase2, phase3 = _inputs(tmp_path, rows=100)
    calls = []
    real_read_csv = pd.read_csv

    def guarded_read_csv(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    frame, _, training_rows, _, _ = load_selected_training_prefix(
        phase2, phase3, _fast_config()
    )
    assert len(frame) == training_rows == 70
    phase4_calls = [call for call in calls if "nrows" in call]
    assert phase4_calls[-1]["nrows"] == 70


def test_reserved_row_changes_do_not_change_phase4_outputs(tmp_path: Path) -> None:
    phase2, phase3 = _inputs(tmp_path)
    config = _fast_config()
    first = run_phase4_models(phase2, phase3, tmp_path / "runs", "first", config)
    source = pd.read_csv(phase2 / "features.csv")
    source.loc[126:, "Consumption"] += 1_000_000
    source.loc[126:, "consumption_lag_1"] -= 1_000_000
    source.to_csv(phase2 / "features.csv", index=False)
    # Phase 3 pins the source hash, so altered reserved rows fail closed rather
    # than silently changing Phase 4 inputs or results.
    try:
        run_phase4_models(phase2, phase3, tmp_path / "runs", "second", config)
    except ValueError as error:
        assert "do not match" in str(error)
    else:
        raise AssertionError("Phase 4 accepted a feature artifact changed after selection")
    assert (first / "metrics.csv").is_file()


def test_runner_preserves_inputs_and_writes_models_and_baselines(tmp_path: Path) -> None:
    phase2, phase3 = _inputs(tmp_path)
    phase2_hash = sha256_file(phase2 / "features.csv")
    phase3_hash = sha256_file(phase3 / "selection_report.json")
    output = run_phase4_models(phase2, phase3, tmp_path / "runs", "phase4-test", _fast_config())

    assert sha256_file(phase2 / "features.csv") == phase2_hash
    assert sha256_file(phase3 / "selection_report.json") == phase3_hash
    report = json.loads((output / "model_report.json").read_text())
    metrics = pd.read_csv(output / "metrics.csv")
    assert report["reserved_rows_loaded"] == 0
    assert report["development_fit_rows"] < report["training_prefix_rows_loaded"]
    assert report["fit_end_date_inclusive"] < report["assessment_start_date"]
    assert set(metrics["model"]) == {"naive_lag_1", *report["model_names"]}
    assert (output / "baseline_actual_vs_predicted.png").stat().st_size > 0
    assert "zero reserved rows" in (output / "execution.log").read_text()
    assert set(path.stem for path in (output / "models").glob("*.joblib")) == set(
        report["model_names"]
    )
    linear = joblib.load(output / "models" / "linear_regression.joblib")
    assert isinstance(linear.named_steps["scaler"], RobustScaler)
