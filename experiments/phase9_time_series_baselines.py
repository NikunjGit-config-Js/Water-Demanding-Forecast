"""Run Phase 9 classical and neural time-series forecasting baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET = Path("data/preprocessed/all/preprocessed_data.csv")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/phase9")
MODEL_NAMES = ("naive_last", "seasonal_naive_7", "arima", "sarimax", "prophet", "lstm", "gru", "cnn_1d")


@dataclass(frozen=True)
class Phase9Config:
    """Fixed, reproducible baseline configuration; no holdout tuning is performed."""

    training_fraction: float = 0.80
    neural_validation_fraction: float = 0.15
    window_size: int = 30
    epochs: int = 30
    batch_size: int = 32
    patience: int = 5
    neural_units: int = 24
    cnn_filters: int = 24
    random_seed: int = 42
    arima_order: tuple[int, int, int] = (7, 1, 1)
    sarimax_order: tuple[int, int, int] = (2, 1, 1)
    sarimax_seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7)

    def validate(self, row_count: int) -> tuple[int, int]:
        if not 0 < self.training_fraction < 1 or not 0 < self.neural_validation_fraction < 0.5:
            raise ValueError("training and neural-validation fractions are invalid")
        if self.window_size < 2 or self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("neural configuration values must be positive")
        split = int(np.floor(row_count * self.training_fraction))
        neural_fit = split - int(np.floor(split * self.neural_validation_fraction))
        if neural_fit <= self.window_size or split >= row_count:
            raise ValueError("dataset is too short for the requested chronological splits")
        return split, neural_fit


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_series(dataset: Path) -> pd.DataFrame:
    """Load only the dated target and enforce the univariate forecasting contract."""
    frame = pd.read_csv(dataset, usecols=["Date", "Consumption"])
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame["Consumption"] = pd.to_numeric(frame["Consumption"], errors="raise")
    if frame.empty or frame["Date"].duplicated().any() or not frame["Date"].is_monotonic_increasing:
        raise ValueError("dates must be non-empty, unique, and strictly chronological")
    if frame["Consumption"].isna().any() or not np.isfinite(frame["Consumption"]).all():
        raise ValueError("Consumption must be finite and complete")
    return frame


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mean_squared_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def recursive_seasonal_naive(history: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    """Forecast recursively, never reading an actual value from the holdout."""
    values = list(np.asarray(history, dtype=float))
    if len(values) < season:
        raise ValueError("history is shorter than the seasonal period")
    for _ in range(horizon):
        values.append(values[-season])
    return np.asarray(values[-horizon:])


def make_windows(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) <= window_size:
        raise ValueError("not enough values to create windows")
    x = np.stack([values[index - window_size:index] for index in range(window_size, len(values))])
    y = values[window_size:]
    return x[..., np.newaxis], y


def recursive_neural_forecast(model: Any, scaled_history: np.ndarray, horizon: int, window_size: int) -> np.ndarray:
    """Recursive multi-step forecast using predictions, not holdout observations."""
    history = list(np.asarray(scaled_history, dtype=float).reshape(-1))
    for _ in range(horizon):
        window = np.asarray(history[-window_size:], dtype=np.float32).reshape(1, window_size, 1)
        # Direct eager calls avoid rebuilding a tf.data input pipeline for every
        # recursive step. Lightweight test doubles may expose predict only.
        raw_prediction = (
            model(window, training=False).numpy()
            if callable(model)
            else model.predict(window, verbose=0)
        )
        prediction = float(np.asarray(raw_prediction).reshape(-1)[0])
        if not np.isfinite(prediction):
            raise RuntimeError("neural model produced a non-finite prediction")
        history.append(prediction)
    return np.asarray(history[-horizon:])


def configure_tensorflow(seed: int) -> tuple[Any, str]:
    """Import TensorFlow lazily and enable laptop-safe GPU memory growth."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return tf, ("GPU" if gpus else "CPU")


def fit_neural_baseline(
    name: str,
    train_values: np.ndarray,
    horizon: int,
    neural_fit_row: int,
    config: Phase9Config,
    model_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one fixed neural architecture with chronological early stopping."""
    tf, device = configure_tensorflow(config.random_seed)
    scaler = RobustScaler()
    scaler.fit(np.asarray(train_values[:neural_fit_row]).reshape(-1, 1))
    scaled = scaler.transform(np.asarray(train_values).reshape(-1, 1)).reshape(-1)
    x_fit, y_fit = make_windows(scaled[:neural_fit_row], config.window_size)
    validation_context = scaled[neural_fit_row - config.window_size:]
    x_validation, y_validation = make_windows(validation_context, config.window_size)
    inputs = tf.keras.Input(shape=(config.window_size, 1))
    if name == "lstm":
        encoded = tf.keras.layers.LSTM(config.neural_units)(inputs)
    elif name == "gru":
        encoded = tf.keras.layers.GRU(config.neural_units)(inputs)
    elif name == "cnn_1d":
        encoded = tf.keras.layers.Conv1D(config.cnn_filters, 3, activation="relu")(inputs)
        encoded = tf.keras.layers.GlobalAveragePooling1D()(encoded)
    else:
        raise ValueError(f"unknown neural baseline: {name}")
    outputs = tf.keras.layers.Dense(1)(encoded)
    model = tf.keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config.patience, restore_best_weights=True
    )
    history = model.fit(
        x_fit, y_fit, validation_data=(x_validation, y_validation), epochs=config.epochs,
        batch_size=config.batch_size, shuffle=False, verbose=0, callbacks=[callback],
    )
    forecast_scaled = recursive_neural_forecast(model, scaled, horizon, config.window_size)
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).reshape(-1)
    model.save(model_path)
    joblib.dump(scaler, model_path.with_suffix(".scaler.joblib"))
    return forecast, {
        "device": device,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_loss": float(min(history.history["val_loss"])),
        "scaler_fit_end_row_exclusive": neural_fit_row,
        "shuffle_used": False,
    }


def fit_classical_baselines(train: pd.DataFrame, future_dates: pd.Series, config: Phase9Config, models_dir: Path) -> dict[str, np.ndarray]:
    values = train["Consumption"].to_numpy(dtype=float)
    horizon = len(future_dates)
    arima = ARIMA(values, order=config.arima_order).fit()
    arima.save(models_dir / "arima.pkl")
    sarimax = SARIMAX(
        values, order=config.sarimax_order, seasonal_order=config.sarimax_seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    sarimax.save(models_dir / "sarimax.pkl")
    from prophet import Prophet

    prophet_frame = train.rename(columns={"Date": "ds", "Consumption": "y"})
    prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet.fit(prophet_frame)
    prophet_prediction = prophet.predict(pd.DataFrame({"ds": future_dates.to_numpy()}))["yhat"].to_numpy()
    with (models_dir / "prophet.json").open("w", encoding="utf-8") as stream:
        from prophet.serialize import model_to_json
        stream.write(model_to_json(prophet))
    return {
        "arima": np.asarray(arima.forecast(horizon)),
        "sarimax": np.asarray(sarimax.forecast(horizon)),
        "prophet": prophet_prediction,
    }


def _save_diagnostics(predictions: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    best = str(metrics.iloc[0]["model"])
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(predictions["Date"], predictions["actual"], label="Actual", color="black", linewidth=1.4)
    for model in ("seasonal_naive_7", best):
        axis.plot(predictions["Date"], predictions[model], label=model, alpha=0.8)
    axis.set(title="Phase 9 chronological recursive forecasts", xlabel="Date", ylabel="Consumption")
    axis.legend()
    figure.tight_layout(); figure.savefig(output / "actual_vs_predicted.png", dpi=150); plt.close(figure)
    residual = predictions["actual"] - predictions[best]
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(predictions["actual"], predictions[best], s=12, alpha=0.65)
    lower = float(min(predictions["actual"].min(), predictions[best].min()))
    upper = float(max(predictions["actual"].max(), predictions[best].max()))
    axis.plot([lower, upper], [lower, upper], color="black", linewidth=1, linestyle="--")
    axis.set(xlabel="Actual consumption", ylabel="Predicted consumption", title=f"Actual versus predicted: {best}")
    figure.tight_layout(); figure.savefig(output / "actual_vs_predicted_scatter.png", dpi=150); plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(predictions[best], residual, s=10, alpha=0.6); axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Predicted", ylabel="Residual", title=f"Residuals: {best}")
    axes[1].hist(residual, bins=30); axes[1].set(xlabel="Residual", ylabel="Count", title="Residual distribution")
    figure.tight_layout(); figure.savefig(output / "residual_diagnostics.png", dpi=150); plt.close(figure)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(predictions["Date"], residual, label="Residual", alpha=0.75)
    axis.plot(predictions["Date"], residual.abs(), label="Absolute error", linewidth=1.3)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Date", ylabel="Error", title=f"Dated forecast error: {best}")
    axis.legend()
    figure.tight_layout(); figure.savefig(output / "error_over_time.png", dpi=150); plt.close(figure)
    errors = pd.DataFrame({"date": predictions["Date"], "model": best, "actual": predictions["actual"], "prediction": predictions[best], "residual": residual})
    errors["absolute_error"] = errors["residual"].abs()
    errors.sort_values(["absolute_error", "date"], ascending=[False, True]).head(20).to_csv(output / "highest_error_dates.csv", index=False, date_format="%Y-%m-%d")


def run_phase9(
    dataset: Path = DEFAULT_DATASET,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    experiment_id: str | None = None,
    config: Phase9Config | None = None,
    classical_runner: Callable[..., dict[str, np.ndarray]] = fit_classical_baselines,
    neural_runner: Callable[..., tuple[np.ndarray, dict[str, Any]]] = fit_neural_baseline,
) -> Path:
    config = config or Phase9Config()
    frame = load_series(dataset.resolve())
    split, neural_fit = config.validate(len(frame))
    train, holdout = frame.iloc[:split].copy(), frame.iloc[split:].copy()
    experiment_id = experiment_id or datetime.now(timezone.utc).strftime("phase9_%Y%m%dT%H%M%SZ")
    output = artifact_root.resolve() / experiment_id
    output.mkdir(parents=True, exist_ok=False)
    models_dir = output / "models"; models_dir.mkdir()
    started = datetime.now(timezone.utc)
    train_values = train["Consumption"].to_numpy(dtype=float)
    predictions = pd.DataFrame({"Date": holdout["Date"], "actual": holdout["Consumption"]})
    predictions["naive_last"] = train_values[-1]
    predictions["seasonal_naive_7"] = recursive_seasonal_naive(train_values, len(holdout))
    for name, forecast in classical_runner(train, holdout["Date"], config, models_dir).items():
        predictions[name] = forecast
    neural_metadata: dict[str, Any] = {}
    for name in ("lstm", "gru", "cnn_1d"):
        forecast, metadata = neural_runner(name, train_values, len(holdout), neural_fit, config, models_dir / f"{name}.keras")
        predictions[name] = forecast
        neural_metadata[name] = metadata
    if set(predictions.columns) != {"Date", "actual", *MODEL_NAMES} or not np.isfinite(predictions[list(MODEL_NAMES)]).all().all():
        raise RuntimeError("one or more Phase 9 forecasts are missing or non-finite")
    metrics = pd.DataFrame([{"model": name, **regression_metrics(predictions["actual"].to_numpy(), predictions[name].to_numpy())} for name in MODEL_NAMES]).sort_values(["mae", "model"]).reset_index(drop=True)
    predictions.to_csv(output / "predictions.csv", index=False, date_format="%Y-%m-%d")
    metrics.to_csv(output / "metrics.csv", index=False)
    _save_diagnostics(predictions, metrics, output)
    report = {
        "phase": "Phase 9", "experiment": "original time-series model baselines",
        "training_rows": len(train), "holdout_rows": len(holdout),
        "training_end_date_inclusive": train["Date"].iloc[-1].date().isoformat(),
        "holdout_start_date": holdout["Date"].iloc[0].date().isoformat(),
        "chronological_order_preserved": True, "shuffle_used": False,
        "forecast_protocol": "fixed-origin recursive multi-step; no actual holdout target is fed to any model",
        "holdout_used_for_fitting_tuning_or_early_stopping": False,
        "neural_validation_scope": "last chronological portion of training only; scaler fitted on earlier neural-fit prefix only",
        "hyperparameter_optimization_performed": False, "improvement_claimed": False,
        "simple_baselines": ["naive_last", "seasonal_naive_7"],
        "model_names": list(MODEL_NAMES), "neural_training": neural_metadata,
        "lowest_holdout_mae_model_descriptive_only": str(metrics.iloc[0]["model"]),
        "dataset_sha256": sha256_file(dataset.resolve()),
    }
    _write_json(output / "forecast_report.json", report)
    environment = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import statsmodels
        environment["statsmodels"] = statsmodels.__version__
        import prophet
        environment["prophet"] = prophet.__version__
        import tensorflow as tf
        environment["tensorflow"] = tf.__version__
    except Exception as error:
        environment["version_capture_error"] = repr(error)
    _write_json(output / "config.json", {"phase": "Phase 9", "experiment_id": experiment_id, "configuration": asdict(config), "random_seed": config.random_seed, "dataset": str(dataset.resolve()), "environment": environment, "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    devices = sorted({str(details["device"]) for details in neural_metadata.values()})
    (output / "execution.log").write_text(f"Phase 9 completed with {len(train)} chronological training and {len(holdout)} untouched forecast rows.\nAll forecasts are fixed-origin recursive multi-step forecasts. Seed: {config.random_seed}. Neural device(s): {', '.join(devices)}. No tuning or improvement claim.\n", encoding="utf-8")
    hashes = {path.relative_to(output).as_posix(): sha256_file(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"}
    _write_json(output / "artifact_hashes.json", hashes)
    return output


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-id")
    args = parser.parse_args(argv)
    print(run_phase9(args.dataset, args.artifact_root, args.experiment_id))


if __name__ == "__main__":
    main()
