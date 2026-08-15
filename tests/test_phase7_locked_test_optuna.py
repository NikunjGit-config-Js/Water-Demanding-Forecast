import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.phase4_traditional_ml import sha256_file
from experiments.phase7_locked_test_optuna import Phase7Config, chronological_partitions, run_phase7_optuna
from tests.test_phase5_chronological_holdout import _inputs


def _config() -> Phase7Config:
    return Phase7Config(trials_per_model=2, random_seed=17)


def test_partitions_are_exact_floor_based_chronological_70_15_15(tmp_path: Path) -> None:
    phase2, phase3 = _inputs(tmp_path, 101)
    frame = pd.read_csv(phase2 / "features.csv", parse_dates=["Date"])
    train, validation, test = chronological_partitions(frame, _config())
    assert (len(train), len(validation), len(test)) == (70, 15, 16)
    assert train["Date"].iloc[-1] < validation["Date"].iloc[0]
    assert validation["Date"].iloc[-1] < test["Date"].iloc[0]


def test_run_preserves_trials_selection_locked_test_and_model_evidence(tmp_path: Path) -> None:
    phase2, phase3 = _inputs(tmp_path, 180)
    output = run_phase7_optuna(phase2, phase3, tmp_path / "runs", "run", _config())
    required = {
        "config.json", "execution.log", "optuna_trials.csv", "validation_metrics.csv",
        "locked_test_metrics.csv", "locked_test_predictions.csv", "selected_model.joblib",
        "linear_baseline.joblib", "locked_test_actual_vs_predicted.png", "split_and_selection_report.json",
        "actual_vs_predicted_scatter.png", "residual_vs_predicted.png",
        "residual_distribution.png", "error_over_time.png", "highest_error_dates.csv",
        "locked_test_diagnostics.csv",
    }
    assert required.issubset(path.name for path in output.iterdir())
    assert all((output / name).stat().st_size > 0 for name in required)
    report = json.loads((output / "split_and_selection_report.json").read_text())
    assert (report["training_rows"], report["validation_rows"], report["locked_test_rows"]) == (125, 28, 27)
    assert report["locked_test_used_for_tuning_selection_or_refit"] is False
    assert report["locked_test_evaluations_after_selection"] == 1
    assert report["shuffle_used"] is False and report["improvement_claimed"] is False
    trials = pd.read_csv(output / "optuna_trials.csv")
    assert len(trials) == 6 and set(trials["model"]) == set(report["tuned_models"])
    assert set(pd.read_csv(output / "locked_test_metrics.csv")["model"]) == {
        "naive_lag_1", "linear_regression", report["selected_model"]
    }
    assert len(pd.read_csv(output / "locked_test_predictions.csv")) == 27
    predictions = pd.read_csv(output / "locked_test_predictions.csv")
    diagnostics = pd.read_csv(output / "locked_test_diagnostics.csv")
    expected_residual = predictions["actual"] - predictions["selected_model"]
    np.testing.assert_allclose(diagnostics["residual"], expected_residual)
    np.testing.assert_allclose(diagnostics["absolute_error"], expected_residual.abs())
    highest_errors = pd.read_csv(output / "highest_error_dates.csv")
    np.testing.assert_allclose(
        highest_errors["residual"], highest_errors["actual"] - highest_errors["selected_model"]
    )
    np.testing.assert_allclose(highest_errors["absolute_error"], highest_errors["residual"].abs())
    assert highest_errors["absolute_error"].is_monotonic_decreasing


def test_locked_test_changes_cannot_change_trials_selection_or_fitted_preprocessing(tmp_path: Path) -> None:
    phase2, phase3 = _inputs(tmp_path, 180)
    first = run_phase7_optuna(phase2, phase3, tmp_path / "runs", "first", _config())
    first_report = json.loads((first / "split_and_selection_report.json").read_text())
    first_trials = pd.read_csv(first / "optuna_trials.csv")
    first_model = joblib.load(first / "selected_model.joblib")

    frame = pd.read_csv(phase2 / "features.csv")
    selected = json.loads((phase3 / "selection_report.json").read_text())["selected_features"]
    frame.loc[153:, selected] = 1_000_000
    frame.loc[153:, "Consumption"] = -1_000_000
    frame.to_csv(phase2 / "features.csv", index=False)
    selection_path = phase3 / "selection_report.json"
    selection = json.loads(selection_path.read_text())
    selection["phase2_features_sha256"] = sha256_file(phase2 / "features.csv")
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    second = run_phase7_optuna(phase2, phase3, tmp_path / "runs", "second", _config())
    second_report = json.loads((second / "split_and_selection_report.json").read_text())
    second_trials = pd.read_csv(second / "optuna_trials.csv")
    second_model = joblib.load(second / "selected_model.joblib")
    pd.testing.assert_frame_equal(first_trials, second_trials)
    assert first_report["selected_model"] == second_report["selected_model"]
    np.testing.assert_allclose(first_model.named_steps["imputer"].statistics_, second_model.named_steps["imputer"].statistics_)
    if "scaler" in first_model.named_steps:
        np.testing.assert_allclose(first_model.named_steps["scaler"].center_, second_model.named_steps["scaler"].center_)
