import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.phase9_time_series_baselines import (
    MODEL_NAMES,
    Phase9Config,
    make_windows,
    recursive_neural_forecast,
    recursive_seasonal_naive,
    run_phase9,
)


def _dataset(path: Path, rows: int = 100) -> Path:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    values = 100 + np.arange(rows) * 0.1 + 5 * np.sin(np.arange(rows) * 2 * np.pi / 7)
    pd.DataFrame({"Date": dates, "Consumption": values}).to_csv(path, index=False)
    return path


def test_recursive_baselines_never_read_holdout_actuals() -> None:
    history = np.arange(1, 15, dtype=float)
    first = recursive_seasonal_naive(history, 12)
    changed_unknown_future = recursive_seasonal_naive(history, 12)
    np.testing.assert_array_equal(first, changed_unknown_future)
    np.testing.assert_array_equal(first[:7], history[-7:])
    np.testing.assert_array_equal(first[7:], first[:5])

    class PlusOne:
        def predict(self, window, verbose=0):
            return np.asarray([[window[0, -1, 0] + 1]])

    neural = recursive_neural_forecast(PlusOne(), history, 3, 4)
    np.testing.assert_array_equal(neural, [15, 16, 17])


def test_windows_and_split_are_strictly_chronological() -> None:
    values = np.arange(12, dtype=float)
    x, y = make_windows(values, 4)
    np.testing.assert_array_equal(x[0, :, 0], [0, 1, 2, 3])
    assert y[0] == 4 and x[-1, -1, 0] == 10 and y[-1] == 11
    split, neural_fit = Phase9Config(window_size=4).validate(100)
    assert split == 80 and neural_fit == 68


def test_run_preserves_complete_causal_evidence(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "series.csv")
    calls: dict[str, object] = {}

    def classical(train, future_dates, config, models_dir):
        calls["classical_train_end"] = train["Date"].iloc[-1]
        calls["future_start"] = future_dates.iloc[0]
        horizon = len(future_dates)
        base = float(train["Consumption"].iloc[-1])
        return {name: np.full(horizon, base + offset) for offset, name in enumerate(("arima", "sarimax", "prophet"), 1)}

    def neural(name, train_values, horizon, neural_fit_row, config, model_path):
        calls.setdefault("neural", []).append((name, len(train_values), neural_fit_row, horizon))
        return np.full(horizon, train_values[-1]), {"device": "CPU", "epochs_completed": 1, "best_validation_loss": 0.1, "scaler_fit_end_row_exclusive": neural_fit_row, "shuffle_used": False}

    output = run_phase9(
        dataset, tmp_path / "runs", "run", Phase9Config(window_size=7, epochs=1),
        classical_runner=classical, neural_runner=neural,
    )
    required = {"config.json", "execution.log", "forecast_report.json", "artifact_hashes.json", "metrics.csv", "predictions.csv", "actual_vs_predicted.png", "actual_vs_predicted_scatter.png", "residual_diagnostics.png", "error_over_time.png", "highest_error_dates.csv"}
    assert required.issubset(path.name for path in output.iterdir())
    report = json.loads((output / "forecast_report.json").read_text())
    assert report["training_rows"] == 80 and report["holdout_rows"] == 20
    assert report["shuffle_used"] is False
    assert report["holdout_used_for_fitting_tuning_or_early_stopping"] is False
    assert report["model_names"] == list(MODEL_NAMES)
    assert calls["classical_train_end"] < calls["future_start"]
    assert all(item[1:] == (80, 68, 20) for item in calls["neural"])
    metrics = pd.read_csv(output / "metrics.csv")
    predictions = pd.read_csv(output / "predictions.csv")
    assert set(metrics["model"]) == set(MODEL_NAMES)
    assert set(predictions) == {"Date", "actual", *MODEL_NAMES}
    assert np.isfinite(metrics[["mae", "mse", "rmse", "r2"]]).all().all()
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert "metrics.csv" in hashes and "forecast_report.json" in hashes
