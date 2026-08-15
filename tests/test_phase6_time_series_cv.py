import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.phase4_traditional_ml import Phase4Config, sha256_file
from experiments.phase6_time_series_cv import Phase6Config, expanding_splits, run_phase6_cv
from tests.test_phase5_chronological_holdout import _inputs


def _models() -> Phase4Config:
    return Phase4Config(random_forest_estimators=3, bagging_estimators=3, gradient_boosting_estimators=3, knn_neighbors=3)


def test_expanding_splits_are_exactly_five_and_strictly_chronological() -> None:
    splits = expanding_splits(180, Phase6Config())
    assert len(splits) == 5
    previous_training_size = 0
    for train, validation in splits:
        assert np.array_equal(train, np.arange(len(train)))
        assert train[-1] < validation[0]
        assert len(train) > previous_training_size
        previous_training_size = len(train)


def test_cv_run_preserves_fold_and_aggregate_evidence(tmp_path: Path) -> None:
    phase2, phase3 = _inputs(tmp_path, 180)
    output = run_phase6_cv(phase2, phase3, tmp_path / "runs", "run", model_config=_models())
    required = {"config.json", "execution.log", "cv_report.json", "fold_metrics.csv", "metrics_summary.csv", "predictions.csv", "actual_vs_predicted.png", "residual_diagnostics.png"}
    assert required.issubset(path.name for path in output.iterdir())
    assert all((output / name).stat().st_size > 0 for name in required)
    report = json.loads((output / "cv_report.json").read_text())
    assert report["n_splits"] == 5 and report["shuffle_used"] is False
    assert report["validation_folds_used_for_fitting_or_tuning"] is False
    assert all(fold["training_end_date_inclusive"] < fold["validation_start_date"] for fold in report["folds"])
    metrics = pd.read_csv(output / "fold_metrics.csv")
    summary = pd.read_csv(output / "metrics_summary.csv")
    assert len(metrics) == 5 * 11 and len(summary) == 11
    expected = metrics.groupby("model")["mae"].agg(["mean", "std"]).sort_index()
    actual = summary.set_index("model")[["mae_mean", "mae_std"]].sort_index()
    np.testing.assert_allclose(actual["mae_mean"], expected["mean"])
    np.testing.assert_allclose(actual["mae_std"], expected["std"], atol=1e-12)
    assert set(pd.read_csv(output / "predictions.csv")["fold"]) == {1, 2, 3, 4, 5}


def test_each_fold_preprocessing_ignores_its_validation_features(tmp_path: Path) -> None:
    phase2, phase3 = _inputs(tmp_path, 180)
    first = run_phase6_cv(phase2, phase3, tmp_path / "runs", "first", model_config=_models())
    first_model = joblib.load(first / "models" / "fold_1" / "linear_regression.joblib")
    frame = pd.read_csv(phase2 / "features.csv")
    selected = json.loads((phase3 / "selection_report.json").read_text())["selected_features"]
    _, validation = expanding_splits(len(frame), Phase6Config())[0]
    frame.loc[validation, selected] = 1_000_000
    frame.to_csv(phase2 / "features.csv", index=False)
    selection_path = phase3 / "selection_report.json"
    selection = json.loads(selection_path.read_text())
    selection["phase2_features_sha256"] = sha256_file(phase2 / "features.csv")
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    second = run_phase6_cv(phase2, phase3, tmp_path / "runs", "second", model_config=_models())
    second_model = joblib.load(second / "models" / "fold_1" / "linear_regression.joblib")
    np.testing.assert_allclose(first_model.named_steps["imputer"].statistics_, second_model.named_steps["imputer"].statistics_)
    np.testing.assert_allclose(first_model.named_steps["scaler"].center_, second_model.named_steps["scaler"].center_)
