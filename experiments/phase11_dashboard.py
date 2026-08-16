"""Artifact-backed data services for the Phase 11 Streamlit dashboard.

This module does not load or execute fitted models.  Historical evaluation is
read from preserved, validated outputs; future forecasts use transparent naive
methods that only receive observations available at the forecast origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_NAMES = ("MAE", "MSE", "RMSE", "R2")
FORECAST_METHODS = {
    "Last observation": 1,
    "Seasonal naive (7 days)": 7,
}


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    phase: str
    evaluation: str
    artifact_directory: str
    predictions_filename: str
    model_columns: tuple[str, ...]

    @property
    def predictions_path(self) -> Path:
        return self.predictions_path_for(PROJECT_ROOT)

    def predictions_path_for(self, project_root: Path) -> Path:
        """Resolve this preserved relative artifact within an explicit run root."""
        return project_root / self.artifact_directory / self.predictions_filename


# Only independently approved outputs are exposed. Invalid and failed attempt
# directories are intentionally absent from this allowlist.
EXPERIMENTS = (
    ExperimentSpec(
        "Phase 5 · chronological 80/20 holdout",
        "Phase 5",
        "Chronological holdout",
        "artifacts/phase5/phase5_attempt_2_20260815T230000Z",
        "predictions.csv",
        ("naive_lag_1", "linear_regression", "ridge", "lasso", "decision_tree", "knn", "svr", "random_forest", "bagging", "gradient_boosting", "voting"),
    ),
    ExperimentSpec(
        "Phase 6 · five-fold time-aware CV",
        "Phase 6",
        "Expanding-window cross-validation",
        "artifacts/phase6/phase6_attempt_1_20260815T233000Z",
        "predictions.csv",
        ("naive_lag_1", "linear_regression", "ridge", "lasso", "decision_tree", "knn", "svr", "random_forest", "bagging", "gradient_boosting", "voting"),
    ),
    ExperimentSpec(
        "Phase 7 · locked test evaluation",
        "Phase 7",
        "Locked chronological test",
        "artifacts/phase7/phase7_attempt_1_20260815T223000Z",
        "locked_test_predictions.csv",
        ("naive_lag_1", "linear_regression", "selected_model"),
    ),
    ExperimentSpec(
        "Phase 8 · nested time-aware CV",
        "Phase 8",
        "Nested expanding-window cross-validation",
        "artifacts/phase8/phase8_attempt_2_final_20260816T003000Z",
        "predictions.csv",
        ("naive_lag_1", "linear_regression", "selected_model"),
    ),
    ExperimentSpec(
        "Phase 9 · time-series baselines",
        "Phase 9",
        "Chronological holdout",
        "artifacts/phase9/phase9_attempt_1_final_20260816T012000Z",
        "predictions.csv",
        ("naive_last", "seasonal_naive_7", "arima", "sarimax", "prophet", "lstm", "gru", "cnn_1d"),
    ),
    ExperimentSpec(
        "Phase 10 · PatchTST evaluation",
        "Phase 10",
        "Chronological development validation",
        "artifacts/phase10/phase10_attempt_1_final_20260816T021000Z",
        "predictions.csv",
        ("naive_lag1", "patchtst"),
    ),
)


def experiment_by_label(label: str) -> ExperimentSpec:
    try:
        return next(item for item in EXPERIMENTS if item.label == label)
    except StopIteration as exc:
        raise ValueError(f"Unknown approved experiment: {label}") from exc


def load_predictions(spec: ExperimentSpec, project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Load and validate one preserved prediction artifact."""
    path = spec.predictions_path_for(project_root).resolve()
    artifact_root = (project_root / "artifacts").resolve()
    if artifact_root not in path.parents:
        raise ValueError("Prediction artifact must remain inside the artifact root")
    if not path.is_file():
        raise FileNotFoundError(f"Preserved predictions are unavailable: {path}")

    frame = pd.read_csv(path)
    required = {"Date", "actual", *spec.model_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Prediction artifact is missing columns: {missing}")
    frame = frame.loc[:, [column for column in frame.columns if column in required or column == "fold"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    numeric_columns = ["actual", *spec.model_columns]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    if frame.empty or frame[numeric_columns].isna().any().any():
        raise ValueError("Prediction artifact is empty or contains missing prediction values")
    sort_columns = ["Date"]
    if "fold" in frame:
        sort_columns.append("fold")
    return frame.sort_values(sort_columns).reset_index(drop=True)


def trailing_window(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    """Return rows from the last N unique dates in an evaluation artifact."""
    if days < 1:
        raise ValueError("Evaluation window must be at least one day")
    dates = frame["Date"].drop_duplicates().sort_values()
    selected_dates = dates.iloc[-min(days, len(dates)) :]
    return frame[frame["Date"].isin(selected_dates)].copy().reset_index(drop=True)


def causal_naive_forecast(
    history: pd.DataFrame,
    horizon_days: int,
    method: str = "Seasonal naive (7 days)",
    forecast_origin: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Forecast calendar days after the origin without reading future targets.

    The input is reduced to observations at or before the explicit origin (or
    its final date when no origin is supplied). Forecasts are generated
    recursively, so no target after the origin can enter a prediction. The
    returned schema intentionally has no ``actual`` column.
    """
    if horizon_days < 1:
        raise ValueError("Forecast horizon must be at least one day")
    if method not in FORECAST_METHODS:
        raise ValueError(f"Unknown forecast method: {method}")
    required = {"Date", "actual"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"Forecast history is missing columns: {missing}")

    observed = history.loc[:, ["Date", "actual"]].copy()
    observed["Date"] = pd.to_datetime(observed["Date"], errors="raise")
    observed["actual"] = pd.to_numeric(observed["actual"], errors="raise")
    origin = observed["Date"].max() if forecast_origin is None else pd.Timestamp(forecast_origin)
    observed = observed.loc[observed["Date"] <= origin]
    observed = observed.sort_values("Date").drop_duplicates("Date", keep="last")
    if observed.empty or not np.isfinite(observed["actual"].to_numpy(dtype=float)).all():
        raise ValueError("Forecast history must contain finite target observations")
    values = observed["actual"].astype(float).tolist()
    period = FORECAST_METHODS[method]
    if len(values) < period:
        raise ValueError(f"Forecast history needs at least {period} observations")

    forecasts: list[float] = []
    recursive_values = values.copy()
    for _ in range(horizon_days):
        forecast = recursive_values[-period]
        forecasts.append(forecast)
        recursive_values.append(forecast)
    dates = pd.date_range(origin + pd.offsets.Day(1), periods=horizon_days, freq="D")
    return pd.DataFrame({"Date": dates, "forecast": forecasts, "method": method})


def downloadable_forecast(frame: pd.DataFrame) -> bytes:
    """Export a future forecast without suggesting that future actuals exist."""
    required = ["Date", "forecast", "method"]
    if any(column not in frame for column in required):
        raise ValueError("Future forecast export has an invalid schema")
    output = frame.loc[:, required].copy()
    output["Date"] = pd.to_datetime(output["Date"], errors="raise").dt.strftime("%Y-%m-%d")
    return output.to_csv(index=False).encode("utf-8")


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    if len(actual_values) == 0 or len(actual_values) != len(predicted_values):
        raise ValueError("Actual and prediction vectors must be non-empty and equal length")
    residual = actual_values - predicted_values
    mse = float(np.mean(np.square(residual)))
    denominator = float(np.sum(np.square(actual_values - np.mean(actual_values))))
    r2 = float("nan") if denominator == 0.0 else 1.0 - float(np.sum(np.square(residual))) / denominator
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2,
    }


def metric_table(frame: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    rows = [{"model": model, **regression_metrics(frame["actual"], frame[model])} for model in models]
    return pd.DataFrame(rows)


def downloadable_predictions(frame: pd.DataFrame, model: str) -> bytes:
    output = frame.loc[:, ["Date", "actual", model]].rename(columns={model: "predicted"}).copy()
    output["residual"] = output["actual"] - output["predicted"]
    output["Date"] = output["Date"].dt.strftime("%Y-%m-%d")
    return output.to_csv(index=False).encode("utf-8")


def downloadable_workbook(frame: pd.DataFrame, models: tuple[str, ...]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        metric_table(frame, models).to_excel(writer, sheet_name="metrics", index=False)
        export = frame.copy()
        export["Date"] = export["Date"].dt.strftime("%Y-%m-%d")
        export.to_excel(writer, sheet_name="predictions", index=False)
    return buffer.getvalue()
