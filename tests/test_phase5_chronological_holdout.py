import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from experiments.phase2_features import build_past_only_features
from experiments.phase3_feature_selection import SelectionConfig, run_phase3_selection
from experiments.phase4_traditional_ml import Phase4Config
from experiments.phase5_chronological_holdout import Phase5Config, load_phase5_frame, run_phase5_holdout


def _inputs(root: Path, rows: int = 180) -> tuple[Path, Path]:
    phase2 = root / "phase2"
    phase2.mkdir()
    dates = pd.date_range("2019-01-01", periods=rows, freq="D")
    target = 100 + 8 * np.sin(np.arange(rows) * 2 * np.pi / 7) + np.arange(rows) * 0.05
    features = build_past_only_features(pd.DataFrame({"Date": dates, "Consumption": target}))
    features.to_csv(phase2 / "features.csv", index=False, date_format="%Y-%m-%d")
    candidates = [column for column in features if column not in {"Date", "Consumption"}]
    (phase2 / "feature_manifest.json").write_text(json.dumps({
        "date_column": "Date", "target_column": "Consumption", "feature_columns": candidates,
        "row_count": rows, "feature_selection": {"performed": False},
        "causality": {"target_derived_features": "past only", "rolling_policy": "shifted", "calendar_offset_policy": "exact", "calendar_features": "known date"},
    }), encoding="utf-8")
    phase3 = run_phase3_selection(phase2, root / "phase3", "selection", SelectionConfig(n_splits=2, n_estimators=8, permutation_repeats=1, max_features=8, stability_top_k=6))
    return phase2, phase3


def _models() -> Phase4Config:
    return Phase4Config(random_forest_estimators=5, bagging_estimators=5, gradient_boosting_estimators=5, knn_neighbors=3)


def test_loader_creates_exact_chronological_80_20_boundary(tmp_path: Path) -> None:
    phase2, phase3 = _inputs(tmp_path, 101)
    frame, selected, split, _ = load_phase5_frame(phase2, phase3, Phase5Config())
    assert len(frame) == 101 and split == 80 and selected
    assert frame.iloc[split - 1]["Date"] < frame.iloc[split]["Date"]


def test_holdout_run_preserves_complete_evidence_and_training_fitted_scaler(tmp_path: Path) -> None:
    phase2, phase3 = _inputs(tmp_path)
    output = run_phase5_holdout(phase2, phase3, tmp_path / "runs", "run", model_config=_models())
    required = {
        "config.json", "execution.log", "holdout_report.json", "metrics.csv",
        "predictions.csv", "actual_vs_predicted.png", "actual_vs_predicted_scatter.png",
        "residual_diagnostics.png", "error_over_time.png", "highest_error_dates.csv",
    }
    assert required.issubset(path.name for path in output.iterdir())
    assert all((output / name).stat().st_size > 0 for name in required)
    report = json.loads((output / "holdout_report.json").read_text())
    assert report["training_rows"] == 144 and report["holdout_rows"] == 36
    assert report["shuffle_used"] is False and report["holdout_used_for_fitting_or_tuning"] is False
    metrics = pd.read_csv(output / "metrics.csv")
    assert set(metrics["model"]) == {"naive_lag_1", *report["model_names"]}
    assert np.isfinite(metrics[["mae", "mse", "rmse", "r2"]]).all().all()
    linear = joblib.load(output / "models" / "linear_regression.joblib")
    assert isinstance(linear, Pipeline) and isinstance(linear.named_steps["scaler"], RobustScaler)

    predictions = pd.read_csv(output / "predictions.csv", parse_dates=["Date"])
    highest_errors = pd.read_csv(output / "highest_error_dates.csv", parse_dates=["date"])
    assert list(highest_errors.columns) == [
        "date", "model", "actual", "prediction", "residual", "absolute_error"
    ]
    assert highest_errors["model"].nunique() == 1
    diagnostic_model = highest_errors["model"].iloc[0]
    expected = pd.DataFrame({
        "date": predictions["Date"],
        "model": diagnostic_model,
        "actual": predictions["actual"],
        "prediction": predictions[diagnostic_model],
    })
    expected["residual"] = expected["actual"] - expected["prediction"]
    expected["absolute_error"] = expected["residual"].abs()
    expected = expected.sort_values(
        ["absolute_error", "date"], ascending=[False, True]
    ).head(20).reset_index(drop=True)
    pd.testing.assert_frame_equal(highest_errors, expected, check_exact=False, rtol=1e-12, atol=1e-12)


def test_fit_preprocessing_statistics_ignore_holdout_features(tmp_path: Path) -> None:
    phase2, phase3 = _inputs(tmp_path)
    first = run_phase5_holdout(phase2, phase3, tmp_path / "runs", "first", model_config=_models())
    first_model = joblib.load(first / "models" / "linear_regression.joblib")
    frame = pd.read_csv(phase2 / "features.csv")
    selected = json.loads((phase3 / "selection_report.json").read_text())["selected_features"]
    frame.loc[144:, selected] = 1_000_000
    frame.to_csv(phase2 / "features.csv", index=False)
    selection_path = phase3 / "selection_report.json"
    selection = json.loads(selection_path.read_text())
    from experiments.phase4_traditional_ml import sha256_file
    selection["phase2_features_sha256"] = sha256_file(phase2 / "features.csv")
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    second = run_phase5_holdout(phase2, phase3, tmp_path / "runs", "second", model_config=_models())
    second_model = joblib.load(second / "models" / "linear_regression.joblib")
    np.testing.assert_allclose(first_model.named_steps["imputer"].statistics_, second_model.named_steps["imputer"].statistics_)
    np.testing.assert_allclose(first_model.named_steps["scaler"].center_, second_model.named_steps["scaler"].center_)
