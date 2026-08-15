import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.phase4_traditional_ml import sha256_file
from experiments.phase3_feature_selection import _numeric_features
from experiments.phase8_cv_optuna import Phase8Config, expanding_splits, run_phase8_cv_optuna
from tests.test_phase5_chronological_holdout import _inputs


def _config() -> Phase8Config:
    return Phase8Config(trials_per_model=1)


def _isolation_config() -> Phase8Config:
    return Phase8Config(trials_per_model=2)


def test_phase8_splits_are_five_expanding_chronological_folds() -> None:
    splits = expanding_splits(180, _config())
    assert len(splits) == 5
    for train, validation in splits:
        assert np.array_equal(train, np.arange(len(train)))
        assert train[-1] < validation[0]


def test_phase8_preserves_trials_baselines_fold_results_and_artifacts(tmp_path: Path) -> None:
    phase2, phase3 = _inputs(tmp_path, 180)
    output = run_phase8_cv_optuna(phase2, phase3, tmp_path / "runs", "run", _config())
    required = {
        "config.json", "execution.log", "cv_selection_report.json", "optuna_trials.csv",
        "tuned_model_selection.csv", "fold_metrics.csv", "metrics_summary.csv", "predictions.csv",
        "actual_vs_predicted.png", "residual_diagnostics.png",
        "artifact_hashes.json",
    }
    assert required.issubset(path.name for path in output.iterdir())
    assert all((output / name).stat().st_size > 0 for name in required)
    report = json.loads((output / "cv_selection_report.json").read_text())
    assert report["n_splits"] == 5 and report["shuffle_used"] is False
    assert report["external_holdout_or_locked_test_used"] is False
    assert report["validation_folds_used_for_preprocessing_fit"] is False
    assert report["phase3_fixed_feature_set_used_for_modeling"] is False
    assert all(fold["training_end_date_inclusive"] < fold["validation_start_date"] for fold in report["folds"])
    trials = pd.read_csv(output / "optuna_trials.csv")
    assert len(trials) == 15 and set(trials["model"]) == {"ridge", "random_forest", "gradient_boosting"}
    metrics = pd.read_csv(output / "fold_metrics.csv")
    assert len(metrics) == 25
    assert set(metrics["model"]) == {"naive_lag_1", "linear_regression", "ridge", "random_forest", "gradient_boosting"}
    assert set(pd.read_csv(output / "predictions.csv")["fold"]) == {1, 2, 3, 4, 5}


def test_fold_selection_preprocessing_hyperparameters_and_predictions_ignore_future_rows(tmp_path: Path) -> None:
    phase2, phase3 = _inputs(tmp_path, 180)
    config = _isolation_config()
    first = run_phase8_cv_optuna(phase2, phase3, tmp_path / "runs", "first", config)
    first_report = json.loads((first / "cv_selection_report.json").read_text())
    first_trials = pd.read_csv(first / "optuna_trials.csv")
    frame = pd.read_csv(phase2 / "features.csv")
    original_frame = frame.copy()
    _, validation = expanding_splits(len(frame), config)[0]
    candidates = json.loads((phase2 / "feature_manifest.json").read_text())["feature_columns"]
    numeric_candidates = [name for name in candidates if name != "season"]
    frame.loc[validation, numeric_candidates] = 1_000_000
    frame.to_csv(phase2 / "features.csv", index=False)
    selection_path = phase3 / "selection_report.json"
    selection = json.loads(selection_path.read_text())
    selection["phase2_features_sha256"] = sha256_file(phase2 / "features.csv")
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    second = run_phase8_cv_optuna(phase2, phase3, tmp_path / "runs", "second", config)
    second_report = json.loads((second / "cv_selection_report.json").read_text())
    second_trials = pd.read_csv(second / "optuna_trials.csv")
    assert first_report["fold_selected_features"]["1"] == second_report["fold_selected_features"]["1"]
    assert first_report["best_parameters_by_outer_fold"]["1"] == second_report["best_parameters_by_outer_fold"]["1"]
    assert first_report["fold_selected_models"]["1"] == second_report["fold_selected_models"]["1"]
    first_ridge = joblib.load(first / "models" / "fold_1" / "ridge.joblib")
    second_ridge = joblib.load(second / "models" / "fold_1" / "ridge.joblib")
    np.testing.assert_allclose(first_ridge.named_steps["imputer"].statistics_, second_ridge.named_steps["imputer"].statistics_)
    np.testing.assert_allclose(first_ridge.named_steps["scaler"].center_, second_ridge.named_steps["scaler"].center_)
    model_name = first_report["fold_selected_models"]["1"]
    first_model = joblib.load(first / "models" / "fold_1" / f"{model_name}.joblib")
    second_model = joblib.load(second / "models" / "fold_1" / f"{model_name}.joblib")
    np.testing.assert_allclose(first_model.named_steps["imputer"].statistics_, second_model.named_steps["imputer"].statistics_)
    pd.testing.assert_frame_equal(
        first_trials.loc[first_trials["outer_fold"] == 1].reset_index(drop=True),
        second_trials.loc[second_trials["outer_fold"] == 1].reset_index(drop=True),
    )
    selected = first_report["fold_selected_features"]["1"]
    fixed_probe = _numeric_features(original_frame.iloc[validation], candidates)[selected]
    np.testing.assert_allclose(first_model.predict(fixed_probe), second_model.predict(fixed_probe))
