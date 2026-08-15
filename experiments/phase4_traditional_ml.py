"""Define and smoke-test the approved Phase 4 traditional ML model set.

The smoke comparison is deliberately confined to the Phase 3 training prefix.
Formal holdout and cross-validation experiments belong to later phases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PHASE2_DIR = Path("artifacts/phase2/phase2_attempt_1_20260815T204500Z")
DEFAULT_PHASE3_DIR = Path("artifacts/phase3/phase3_attempt_1_final_20260815T210000Z")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/phase4")
SEASON_CODES = {"winter": 0.0, "spring": 1.0, "summer": 2.0, "autumn": 3.0}
SCALED_MODELS = {"linear_regression", "ridge", "lasso", "knn", "svr"}


@dataclass(frozen=True)
class Phase4Config:
    """Deterministic configuration for the training-only model smoke check."""

    training_fraction: float = 0.70
    development_fit_fraction: float = 0.80
    random_seed: int = 42
    ridge_alpha: float = 1.0
    lasso_alpha: float = 0.01
    decision_tree_max_depth: int = 12
    knn_neighbors: int = 7
    svr_c: float = 10.0
    svr_epsilon: float = 0.1
    random_forest_estimators: int = 150
    bagging_estimators: int = 50
    gradient_boosting_estimators: int = 100

    def validate(self, row_count: int) -> tuple[int, int]:
        if not 0.0 < self.training_fraction < 1.0:
            raise ValueError("training_fraction must be strictly between zero and one")
        if not 0.0 < self.development_fit_fraction < 1.0:
            raise ValueError("development_fit_fraction must be strictly between zero and one")
        training_rows = int(np.floor(row_count * self.training_fraction))
        fit_rows = int(np.floor(training_rows * self.development_fit_fraction))
        if fit_rows < 2 or fit_rows >= training_rows:
            raise ValueError("training prefix is too short for a development split")
        return training_rows, fit_rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_selected_training_prefix(
    phase2_dir: Path, phase3_dir: Path, config: Phase4Config
) -> tuple[pd.DataFrame, list[str], int, int, dict[str, str]]:
    """Load only selected columns and only rows in the approved training prefix."""
    phase2_dir = phase2_dir.resolve()
    phase3_dir = phase3_dir.resolve()
    manifest = _read_json(phase2_dir / "feature_manifest.json")
    selection = _read_json(phase3_dir / "selection_report.json")
    selected = selection.get("selected_features")
    candidates = manifest.get("feature_columns")
    row_count = manifest.get("row_count")
    if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
        raise ValueError("Phase 3 selected feature list must be non-empty and unique")
    if not isinstance(candidates, list) or not set(selected).issubset(candidates):
        raise ValueError("Phase 3 selected features do not match the Phase 2 manifest")
    if not isinstance(row_count, int):
        raise ValueError("Phase 2 row count is invalid")
    training_rows, fit_rows = config.validate(row_count)
    if selection.get("training_rows") != training_rows:
        raise ValueError("Phase 3 training boundary does not match Phase 4 configuration")
    features_path = phase2_dir / "features.csv"
    expected_hash = selection.get("phase2_features_sha256")
    actual_hash = sha256_file(features_path)
    if expected_hash != actual_hash:
        raise ValueError("Phase 2 features do not match the Phase 3 selection input")
    auxiliary_columns = [] if "consumption_lag_1" in selected else ["consumption_lag_1"]
    if "consumption_lag_1" not in candidates:
        raise ValueError("Phase 2 manifest lacks the required naive lag-1 baseline")
    columns = ["Date", "Consumption", *selected, *auxiliary_columns]
    frame = pd.read_csv(features_path, nrows=training_rows, usecols=columns)
    if len(frame) != training_rows or set(frame.columns) != set(columns):
        raise ValueError("Phase 2 feature artifact does not satisfy the selected schema")
    frame = frame.loc[:, columns]
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if frame["Date"].duplicated().any() or not frame["Date"].is_monotonic_increasing:
        raise ValueError("training dates must be unique and strictly chronological")
    frame["Consumption"] = pd.to_numeric(frame["Consumption"], errors="raise")
    if frame["Consumption"].isna().any() or not np.isfinite(frame["Consumption"]).all():
        raise ValueError("training target must be finite and complete")
    for column in selected:
        if column == "season":
            frame[column] = frame[column].map(SEASON_CODES)
            if frame[column].isna().all():
                raise ValueError("season contains no recognized values")
        else:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame[selected] = frame[selected].replace([np.inf, -np.inf], np.nan).astype(float)
    return frame, selected, training_rows, fit_rows, {
        "phase2_features_sha256": actual_hash,
        "phase3_selection_report_sha256": sha256_file(phase3_dir / "selection_report.json"),
    }


def _pipeline(estimator: RegressorMixin, *, scale: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", RobustScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def build_model_registry(config: Phase4Config | None = None) -> dict[str, Pipeline]:
    """Construct exactly the approved traditional model families."""
    config = config or Phase4Config()
    seed = config.random_seed
    models: dict[str, Pipeline] = {
        "linear_regression": _pipeline(LinearRegression(), scale=True),
        "ridge": _pipeline(Ridge(alpha=config.ridge_alpha), scale=True),
        "lasso": _pipeline(
            Lasso(alpha=config.lasso_alpha, max_iter=20_000, random_state=seed), scale=True
        ),
        "decision_tree": _pipeline(
            DecisionTreeRegressor(max_depth=config.decision_tree_max_depth, random_state=seed),
            scale=False,
        ),
        "knn": _pipeline(KNeighborsRegressor(n_neighbors=config.knn_neighbors), scale=True),
        "svr": _pipeline(SVR(C=config.svr_c, epsilon=config.svr_epsilon), scale=True),
        "random_forest": _pipeline(
            RandomForestRegressor(
                n_estimators=config.random_forest_estimators,
                random_state=seed,
                n_jobs=1,
            ),
            scale=False,
        ),
        "bagging": _pipeline(
            BaggingRegressor(n_estimators=config.bagging_estimators, random_state=seed, n_jobs=1),
            scale=False,
        ),
        "gradient_boosting": _pipeline(
            GradientBoostingRegressor(
                n_estimators=config.gradient_boosting_estimators, random_state=seed
            ),
            scale=False,
        ),
    }
    voting = VotingRegressor(
        estimators=[
            (
                "ridge",
                Pipeline(
                    [("scaler", RobustScaler()), ("model", Ridge(alpha=config.ridge_alpha))]
                ),
            ),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=config.random_forest_estimators,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
            (
                "gradient_boosting",
                GradientBoostingRegressor(
                    n_estimators=config.gradient_boosting_estimators, random_state=seed
                ),
            ),
        ]
    )
    models["voting"] = _pipeline(voting, scale=False)
    return models


def regression_metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    mse = mean_squared_error(actual_values, predicted_values)
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual_values, predicted_values)),
    }


def _save_baseline_plot(predictions: pd.DataFrame, path: Path) -> None:
    """Plot only the simple baselines to avoid implying Phase 4 model selection."""
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(predictions["Date"], predictions["actual"], label="Actual", linewidth=1.5)
    axis.plot(
        predictions["Date"], predictions["naive_lag_1"], label="Naive lag-1", alpha=0.8
    )
    axis.plot(
        predictions["Date"], predictions["linear_regression"], label="Linear regression", alpha=0.8
    )
    axis.set(title="Phase 4 training-prefix smoke comparison", xlabel="Date", ylabel="Consumption")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_phase4_models(
    phase2_dir: Path = DEFAULT_PHASE2_DIR,
    phase3_dir: Path = DEFAULT_PHASE3_DIR,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    experiment_id: str | None = None,
    config: Phase4Config | None = None,
) -> Path:
    """Fit all model families on an earlier portion of the approved training prefix."""
    config = config or Phase4Config()
    if experiment_id is None:
        experiment_id = datetime.now(timezone.utc).strftime("phase4_%Y%m%dT%H%M%SZ")
    output_dir = artifact_root.resolve() / experiment_id
    output_dir.mkdir(parents=True, exist_ok=False)
    models_dir = output_dir / "models"
    models_dir.mkdir()
    started_at = datetime.now(timezone.utc)
    frame, selected, training_rows, fit_rows, hashes = load_selected_training_prefix(
        phase2_dir, phase3_dir, config
    )
    fit = frame.iloc[:fit_rows]
    assessment = frame.iloc[fit_rows:]
    x_fit, y_fit = fit[selected], fit["Consumption"]
    x_assess, y_assess = assessment[selected], assessment["Consumption"]

    prediction_table = pd.DataFrame({"Date": assessment["Date"], "actual": y_assess})
    naive_predictions = assessment["consumption_lag_1"].to_numpy(dtype=float)
    if not np.isfinite(naive_predictions).all():
        raise ValueError("naive lag-1 baseline is incomplete on the assessment interval")
    prediction_table["naive_lag_1"] = naive_predictions
    metric_rows = [{"model": "naive_lag_1", **regression_metrics(y_assess, naive_predictions)}]

    registry = build_model_registry(config)
    for name, model in registry.items():
        model.fit(x_fit, y_fit)
        predictions = model.predict(x_assess)
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"{name} produced non-finite predictions")
        prediction_table[name] = predictions
        metric_rows.append({"model": name, **regression_metrics(y_assess, predictions)})
        joblib.dump(model, models_dir / f"{name}.joblib")

    metrics = pd.DataFrame(metric_rows).sort_values(["mae", "model"]).reset_index(drop=True)
    prediction_table.to_csv(output_dir / "predictions.csv", index=False, date_format="%Y-%m-%d")
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    _save_baseline_plot(prediction_table, output_dir / "baseline_actual_vs_predicted.png")
    report = {
        "phase": "Phase 4",
        "purpose": "training-prefix-only model construction and smoke comparison",
        "formal_evaluation_deferred_to": ["Phase 5 chronological 80/20", "Phase 6 time-aware CV"],
        "model_names": list(registry),
        "simple_baselines": ["naive_lag_1", "linear_regression"],
        "boosting_choice": "GradientBoostingRegressor; one approved boosting family is sufficient",
        "selected_features": selected,
        "selected_feature_count": len(selected),
        "training_prefix_rows_loaded": training_rows,
        "reserved_rows_loaded": 0,
        "development_fit_rows": fit_rows,
        "development_assessment_rows": training_rows - fit_rows,
        "fit_end_date_inclusive": fit["Date"].iloc[-1].date().isoformat(),
        "assessment_start_date": assessment["Date"].iloc[0].date().isoformat(),
        "assessment_end_date_inclusive": assessment["Date"].iloc[-1].date().isoformat(),
        "chronological_order_preserved": True,
        "preprocessing": {
            "imputer": "median fitted by each Pipeline on development fit rows only",
            "scaled_models": sorted(SCALED_MODELS),
            "scaler": "RobustScaler fitted by each scaled-model Pipeline on development fit rows only",
            "tree_models_scaled": False,
        },
        "model_selection_claimed": False,
        "improvement_claimed": False,
        **hashes,
    }
    _write_json(output_dir / "model_report.json", report)
    (output_dir / "execution.log").write_text(
        "Phase 4 traditional-model smoke run completed successfully.\n"
        f"Loaded {training_rows} training-prefix rows and zero reserved rows.\n"
        f"Fitted {len(registry)} model families with random seed {config.random_seed}.\n"
        "No model selection or improvement claim was made.\n",
        encoding="utf-8",
    )
    _write_json(
        output_dir / "config.json",
        {
            "phase": "Phase 4",
            "experiment_id": experiment_id,
            "phase2_directory": str(Path(phase2_dir).resolve()),
            "phase3_directory": str(Path(phase3_dir).resolve()),
            "configuration": asdict(config),
            "random_seed": config.random_seed,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": [
                "config.json",
                "baseline_actual_vs_predicted.png",
                "execution.log",
                "metrics.csv",
                "model_report.json",
                "predictions.csv",
                "models/*.joblib",
            ],
        },
    )
    return output_dir


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2_DIR)
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-id")
    args = parser.parse_args(argv)
    print(
        run_phase4_models(
            args.phase2_dir, args.phase3_dir, args.artifact_root, args.experiment_id
        )
    )


if __name__ == "__main__":
    main()
