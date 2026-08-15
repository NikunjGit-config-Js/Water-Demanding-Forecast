"""Reproduce the historical Phase 0 linear-regression baseline with artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


DEFAULT_DATASET = Path("data/preprocessed/all/preprocessed_data.csv")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/phase0")
HISTORICAL_METRIC_NAMES = ("MAE", "MAPE", "MSE", "RMSE")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_windows(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Create the same past-only flattened windows as the historical model."""
    if values.ndim != 2 or len(values) <= window_size:
        raise ValueError("values must be 2-D with more rows than window_size")
    features = np.empty((len(values) - window_size, window_size * values.shape[1]))
    targets = values[window_size:].copy()
    for index in range(len(features)):
        features[index] = values[index : index + window_size].reshape(-1)
    return features, targets


def recursive_forecast(
    model: LinearRegression, recent_window: np.ndarray, horizon: int
) -> np.ndarray:
    """Match the original recursive forecast, feeding each prediction forward."""
    current = recent_window.copy()
    predictions = np.empty(horizon)
    for index in range(horizon):
        prediction = float(model.predict(current.reshape(1, -1))[0, 0])
        predictions[index] = prediction
        current = np.roll(current, -1)
        current[-1] = prediction
    return predictions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_phase0_baseline(
    dataset_path: Path = DEFAULT_DATASET,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    experiment_id: str | None = None,
    test_fraction: float = 0.1,
    window_size: int = 300,
) -> Path:
    """Run the deterministic historical baseline and return its artifact directory."""
    dataset_path = dataset_path.resolve()
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    if experiment_id is None:
        experiment_id = datetime.now(timezone.utc).strftime("phase0_%Y%m%dT%H%M%SZ")
    output_dir = artifact_root.resolve() / experiment_id
    output_dir.mkdir(parents=True, exist_ok=False)

    log_path = output_dir / "execution.log"
    logger = logging.getLogger(f"phase0.{experiment_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)sZ %(levelname)s %(message)s"))
    logger.handlers = [handler]

    started_at = datetime.now(timezone.utc)
    logger.info("Starting deterministic Phase 0 baseline reproduction")
    logger.info("Dataset: %s", dataset_path)

    frame = pd.read_csv(dataset_path)
    if list(frame.columns) != ["Date", "Consumption"]:
        raise ValueError("dataset must contain exactly Date and Consumption columns")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if not frame["Date"].is_monotonic_increasing:
        raise ValueError("dataset must be chronological")
    if frame.isna().any().any():
        raise ValueError("dataset contains missing values")

    split_index = int((1 - test_fraction) * len(frame))
    train = frame.iloc[:split_index].copy()
    test = frame.iloc[split_index:].copy()
    if len(train) <= window_size or test.empty:
        raise ValueError("split does not provide enough training and test rows")

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train[["Consumption"]])
    train_features, train_targets = make_windows(scaled_train, window_size)
    model = LinearRegression()
    model.fit(train_features, train_targets)

    # This intentionally preserves the original implementation's seed window:
    # X_train[-1], rather than constructing a new window ending at train.iloc[-1].
    scaled_recursive = recursive_forecast(model, train_features[-1], len(test))
    forecasts = scaler.inverse_transform(scaled_recursive.reshape(-1, 1)).ravel()

    combined_scaled = np.vstack([scaled_train[-window_size:], scaler.transform(test[["Consumption"]])])
    test_features, _ = make_windows(combined_scaled, window_size)
    one_step = scaler.inverse_transform(model.predict(test_features)).ravel()

    predictions = pd.DataFrame(
        {
            "Date": test["Date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "actual": test["Consumption"].to_numpy(dtype=float),
            "forecast": forecasts,
            "error": test["Consumption"].to_numpy(dtype=float) - forecasts,
            "absolute_error": np.abs(test["Consumption"].to_numpy(dtype=float) - forecasts),
            "absolute_percentage_error": np.abs(
                (test["Consumption"].to_numpy(dtype=float) - forecasts)
                / test["Consumption"].to_numpy(dtype=float)
            ),
            "one_step_prediction": one_step,
        }
    )
    error = predictions["error"].to_numpy()
    historical_metrics = {
        "MAE": float(np.mean(np.abs(error))),
        "MAPE": float(np.mean(predictions["absolute_percentage_error"])),
        "MSE": float(np.mean(error**2)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
    }
    metrics = {
        "historical_baseline_metrics": historical_metrics,
        "supplemental_phase0_metrics": {"R2": float(r2_score(predictions["actual"], forecasts))},
        "metric_policy_note": (
            "The historical implementation reported MAE, MAPE, MSE, and RMSE but omitted R2. "
            "R2 is recorded here only as a supplemental Phase 0 metric required by PROJECT_SPEC.md."
        ),
    }

    config = {
        "experiment_id": experiment_id,
        "phase": "Phase 0",
        "purpose": "faithful historical linear-regression baseline reproduction",
        "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)},
        "feature_definition": {
            "input": "300 past standardized Consumption observations, flattened",
            "target": "next-day Consumption",
            "past_only": True,
            "forecast_mode": "recursive multi-step",
            "historical_seed_window": "X_train[-1] (preserved exactly)",
        },
        "hyperparameters": {
            "test_fraction": test_fraction,
            "window_size_T_X": window_size,
            "univariate": True,
            "estimator": "sklearn.linear_model.LinearRegression",
            "estimator_parameters": model.get_params(deep=True),
            "scaler": "sklearn.preprocessing.StandardScaler",
        },
        "determinism": {
            "random_seed": None,
            "declaration": (
                "No random seed is used: LinearRegression and StandardScaler are deterministic "
                "for fixed input data, and this experiment performs no randomized operation."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "started_at_utc": started_at.isoformat(),
    }
    split_manifest = {
        "strategy": "chronological 90/10 holdout; no shuffle",
        "total_rows": len(frame),
        "train": {
            "row_count": len(train),
            "start_date": train["Date"].iloc[0].strftime("%Y-%m-%d"),
            "end_date": train["Date"].iloc[-1].strftime("%Y-%m-%d"),
            "positional_rows": [0, split_index - 1],
        },
        "test": {
            "row_count": len(test),
            "start_date": test["Date"].iloc[0].strftime("%Y-%m-%d"),
            "end_date": test["Date"].iloc[-1].strftime("%Y-%m-%d"),
            "positional_rows": [split_index, len(frame) - 1],
        },
        "preprocessing_fit_scope": "StandardScaler fitted on training rows only",
    }

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "split_manifest.json", split_manifest)
    _write_json(output_dir / "metrics.json", metrics)
    joblib.dump(model, output_dir / "linear_regression.joblib")
    joblib.dump(scaler, output_dir / "standard_scaler.joblib")

    fig, axis = plt.subplots(figsize=(12, 5))
    dates = pd.to_datetime(predictions["Date"])
    axis.plot(dates, predictions["actual"], label="Actual", linewidth=1.3)
    axis.plot(dates, predictions["forecast"], label="Historical recursive forecast", linewidth=1.3)
    axis.set(title="Phase 0: actual vs predicted", xlabel="Date", ylabel="Consumption")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    completed_at = datetime.now(timezone.utc)
    config["completed_at_utc"] = completed_at.isoformat()
    config["artifacts"] = sorted(path.name for path in output_dir.iterdir())
    _write_json(output_dir / "config.json", config)
    logger.info("Historical metrics: %s", json.dumps(historical_metrics, sort_keys=True))
    logger.info("Supplemental R2: %.15g", metrics["supplemental_phase0_metrics"]["R2"])
    logger.info("Saved fitted model and training-only scaler")
    logger.info("Completed Phase 0 reproduction with %d test predictions", len(predictions))
    handler.close()
    logger.handlers.clear()
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    output_dir = run_phase0_baseline(args.dataset, args.artifact_root, args.experiment_id)
    print(output_dir)


if __name__ == "__main__":
    main()
