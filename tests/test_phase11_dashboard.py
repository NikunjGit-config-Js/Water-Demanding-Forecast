from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.phase11_dashboard import (
    EXPERIMENTS,
    ExperimentSpec,
    causal_naive_forecast,
    downloadable_forecast,
    downloadable_predictions,
    experiment_by_label,
    load_predictions,
    metric_table,
    regression_metrics,
    trailing_window,
)


def test_allowlist_contains_only_final_or_approved_artifacts() -> None:
    paths = [spec.artifact_directory for spec in EXPERIMENTS]
    assert len(paths) == len(set(paths)) == 6
    assert all("phase8_attempt_1" not in path for path in paths)
    assert all("phase8_attempt_2_20260815T235500Z" not in path for path in paths)
    assert all("phase9_attempt_1_20260816T010000Z" not in path for path in paths)
    for spec in EXPERIMENTS:
        assert spec.predictions_path.is_file()


def test_all_preserved_predictions_load_with_declared_models() -> None:
    for spec in EXPERIMENTS:
        frame = load_predictions(spec)
        assert not frame.empty
        assert frame["Date"].is_monotonic_increasing
        assert set(("Date", "actual", *spec.model_columns)).issubset(frame.columns)


def test_prediction_loading_accepts_an_explicit_city_root(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts/cities/pune/phase5/run"
    artifact.mkdir(parents=True)
    pd.DataFrame(
        {"Date": ["2024-01-01"], "actual": [1.0], "model": [1.5]}
    ).to_csv(artifact / "predictions.csv", index=False)
    spec = ExperimentSpec(
        "Pune", "Phase 5", "holdout", "artifacts/cities/pune/phase5/run",
        "predictions.csv", ("model",),
    )

    assert load_predictions(spec, tmp_path).loc[0, "model"] == 1.5


def test_trailing_window_uses_dates_not_row_count() -> None:
    frame = pd.DataFrame(
        {"Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]), "actual": range(4)}
    )
    result = trailing_window(frame, 2)
    assert result["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-02", "2024-01-03"]
    with pytest.raises(ValueError, match="at least one"):
        trailing_window(frame, 0)


def test_regression_metrics_and_export_are_exact() -> None:
    frame = pd.DataFrame(
        {"Date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "actual": [1.0, 3.0], "model": [2.0, 3.0]}
    )
    metrics = regression_metrics(frame["actual"], frame["model"])
    assert metrics == pytest.approx({"MAE": 0.5, "MSE": 0.5, "RMSE": np.sqrt(0.5), "R2": 0.5})
    table = metric_table(frame, ("model",))
    assert table.loc[0, "MAE"] == pytest.approx(0.5)
    exported = pd.read_csv(pd.io.common.BytesIO(downloadable_predictions(frame, "model")))
    assert exported.columns.tolist() == ["Date", "actual", "predicted", "residual"]
    assert exported["residual"].tolist() == [-1.0, 0.0]


def test_unknown_experiment_and_outside_artifact_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown approved"):
        experiment_by_label("not approved")
    spec = ExperimentSpec("bad", "Phase 11", "bad", str(tmp_path), "predictions.csv", ("model",))
    with pytest.raises(ValueError, match="artifact root"):
        load_predictions(spec)


def test_forecast_horizon_changes_output_and_preserves_daily_alignment() -> None:
    history = pd.DataFrame(
        {"Date": pd.date_range("2024-01-01", periods=10), "actual": np.arange(10, dtype=float)}
    )
    short = causal_naive_forecast(history, 2, "Seasonal naive (7 days)")
    long = causal_naive_forecast(history, 9, "Seasonal naive (7 days)")
    assert len(short) == 2
    assert len(long) == 9
    assert short.equals(long.iloc[:2].reset_index(drop=True))
    assert long["Date"].tolist() == list(pd.date_range("2024-01-11", periods=9))
    assert long["forecast"].tolist() == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 3.0, 4.0]


def test_future_targets_cannot_affect_forecast_or_enter_export() -> None:
    origin_history = pd.DataFrame(
        {"Date": pd.date_range("2024-01-01", periods=7), "actual": np.arange(1.0, 8.0)}
    )
    future_actuals = pd.DataFrame(
        {"Date": pd.date_range("2024-01-08", periods=3), "actual": [9999.0, -9999.0, 1234.0]}
    )
    complete = pd.concat([origin_history, future_actuals], ignore_index=True)
    expected = causal_naive_forecast(complete, 3, forecast_origin="2024-01-07")
    actuals_mutated_elsewhere = future_actuals.assign(actual=[-1.0, -2.0, -3.0])
    mutated_complete = pd.concat([origin_history, actuals_mutated_elsewhere], ignore_index=True)
    assert causal_naive_forecast(mutated_complete, 3, forecast_origin="2024-01-07").equals(expected)
    exported = pd.read_csv(pd.io.common.BytesIO(downloadable_forecast(expected)))
    assert exported.columns.tolist() == ["Date", "forecast", "method"]
    assert "actual" not in exported.columns
    assert exported["forecast"].tolist() == [1.0, 2.0, 3.0]


def test_streamlit_app_renders_accessible_controls() -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app = streamlit_testing.AppTest.from_file("app/app.py", default_timeout=20).run()
    assert not app.exception
    assert {widget.label for widget in app.selectbox} == {
        "Evaluation experiment", "Model", "Highlighted metric", "Forecast method"
    }
    assert {widget.label for widget in app.slider} == {"Evaluation window (days)", "Forecast horizon (days)"}
    assert [widget.label for widget in app.radio] == ["Plot"]
    assert len(app.metric) == 4
