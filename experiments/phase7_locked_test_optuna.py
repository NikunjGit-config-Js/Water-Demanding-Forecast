"""Run Phase 7: train/validation Optuna tuning followed by one locked-test evaluation."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd
import sklearn
from optuna.samplers import TPESampler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from experiments.phase4_traditional_ml import (
    DEFAULT_PHASE2_DIR,
    DEFAULT_PHASE3_DIR,
    regression_metrics,
)
from experiments.phase5_chronological_holdout import Phase5Config, load_phase5_frame

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ARTIFACT_ROOT = Path("artifacts/phase7")
TUNED_MODELS = ("ridge", "random_forest", "gradient_boosting")


@dataclass(frozen=True)
class Phase7Config:
    """Configuration fixed before the locked test is observed."""

    training_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    trials_per_model: int = 20
    random_seed: int = 42

    def boundaries(self, row_count: int) -> tuple[int, int]:
        fractions = (self.training_fraction, self.validation_fraction, self.test_fraction)
        if not np.isclose(sum(fractions), 1.0) or any(value <= 0 for value in fractions):
            raise ValueError("Phase 7 fractions must be positive and sum to one")
        if self.trials_per_model < 1:
            raise ValueError("trials_per_model must be positive")
        train_end = int(np.floor(row_count * self.training_fraction))
        validation_end = int(np.floor(row_count * (self.training_fraction + self.validation_fraction)))
        if train_end < 2 or validation_end <= train_end or validation_end >= row_count:
            raise ValueError("dataset is too short for the Phase 7 split")
        return train_end, validation_end


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def chronological_partitions(
    frame: pd.DataFrame, config: Phase7Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return exact, non-overlapping 70/15/15 chronological partitions."""
    train_end, validation_end = config.boundaries(len(frame))
    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    locked_test = frame.iloc[validation_end:].copy()
    if not (train["Date"].iloc[-1] < validation["Date"].iloc[0] <= validation["Date"].iloc[-1] < locked_test["Date"].iloc[0]):
        raise RuntimeError("Phase 7 partitions are not strictly chronological")
    return train, validation, locked_test


def _suggested_pipeline(trial: optuna.Trial, model_name: str, seed: int) -> Pipeline:
    if model_name == "ridge":
        estimator = Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100.0, log=True))
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler()), ("model", estimator)])
    if model_name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 75, 250, step=25),
            max_depth=trial.suggest_int("max_depth", 4, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_float("max_features", 0.5, 1.0),
            random_state=seed,
            n_jobs=1,
        )
    elif model_name == "gradient_boosting":
        estimator = GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 250, step=25),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            random_state=seed,
        )
    else:
        raise ValueError(f"unsupported tuned model: {model_name}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def _fixed_pipeline(model_name: str, params: dict[str, Any], seed: int) -> Pipeline:
    trial = optuna.trial.FixedTrial(params)
    return _suggested_pipeline(trial, model_name, seed)


def _save_plot(predictions: pd.DataFrame, selected_model: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(predictions["Date"], predictions["actual"], label="Actual", linewidth=1.5)
    axis.plot(predictions["Date"], predictions["naive_lag_1"], label="Naive lag-1", alpha=0.75)
    axis.plot(predictions["Date"], predictions["selected_model"], label=selected_model, alpha=0.85)
    axis.set(title="Phase 7 locked-test evaluation", xlabel="Date", ylabel="Consumption")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_locked_test_diagnostics(
    predictions: pd.DataFrame,
    selected_model: str,
    output: Path,
    highest_error_count: int = 20,
) -> None:
    """Create descriptive diagnostics from already-frozen test predictions only."""
    required = {"Date", "actual", "selected_model"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"locked-test predictions are missing columns: {sorted(missing)}")
    if highest_error_count < 1:
        raise ValueError("highest_error_count must be positive")

    diagnostics = predictions[["Date", "actual", "selected_model"]].copy()
    diagnostics["residual"] = diagnostics["actual"] - diagnostics["selected_model"]
    diagnostics["absolute_error"] = diagnostics["residual"].abs()
    diagnostics.to_csv(output / "locked_test_diagnostics.csv", index=False, date_format="%Y-%m-%d")
    diagnostics.nlargest(highest_error_count, "absolute_error").to_csv(
        output / "highest_error_dates.csv", index=False, date_format="%Y-%m-%d"
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(diagnostics["actual"], diagnostics["selected_model"], s=18, alpha=0.65)
    bounds = [
        min(diagnostics["actual"].min(), diagnostics["selected_model"].min()),
        max(diagnostics["actual"].max(), diagnostics["selected_model"].max()),
    ]
    axis.plot(bounds, bounds, linestyle="--", color="black", linewidth=1, label="Ideal")
    axis.set(title=f"Locked-test actual vs predicted: {selected_model}", xlabel="Actual", ylabel="Predicted")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "actual_vs_predicted_scatter.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(diagnostics["selected_model"], diagnostics["residual"], s=18, alpha=0.65)
    axis.axhline(0, linestyle="--", color="black", linewidth=1)
    axis.set(title=f"Locked-test residuals: {selected_model}", xlabel="Predicted", ylabel="Residual (actual - predicted)")
    figure.tight_layout()
    figure.savefig(output / "residual_vs_predicted.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    residual_values = diagnostics["residual"].to_numpy(dtype=float)
    residual_span = float(np.ptp(residual_values))
    residual_scale = max(float(np.max(np.abs(residual_values))), 1.0)
    if residual_span <= np.finfo(float).eps * residual_scale:
        center = float(np.mean(residual_values))
        half_width = max(abs(center) * 1e-6, 1.0)
        histogram_bins: int | np.ndarray = np.array([center - half_width, center + half_width])
    else:
        histogram_bins = 30
    axis.hist(residual_values, bins=histogram_bins)
    axis.axvline(0, linestyle="--", color="black", linewidth=1)
    axis.set(title=f"Locked-test residual distribution: {selected_model}", xlabel="Residual (actual - predicted)", ylabel="Count")
    figure.tight_layout()
    figure.savefig(output / "residual_distribution.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(diagnostics["Date"], diagnostics["residual"], linewidth=1)
    axis.axhline(0, linestyle="--", color="black", linewidth=1)
    axis.set(title=f"Locked-test error over time: {selected_model}", xlabel="Date", ylabel="Error (actual - predicted)")
    figure.tight_layout()
    figure.savefig(output / "error_over_time.png", dpi=150)
    plt.close(figure)


def run_phase7_optuna(
    phase2_dir: Path = DEFAULT_PHASE2_DIR,
    phase3_dir: Path = DEFAULT_PHASE3_DIR,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    experiment_id: str | None = None,
    config: Phase7Config | None = None,
) -> Path:
    """Tune on train/validation only, freeze selection, then evaluate locked test once."""
    config = config or Phase7Config()
    experiment_id = experiment_id or datetime.now(timezone.utc).strftime("phase7_%Y%m%dT%H%M%SZ")
    output = artifact_root.resolve() / experiment_id
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)

    frame, selected_features, _, hashes = load_phase5_frame(phase2_dir, phase3_dir, Phase5Config())
    train, validation, locked_test = chronological_partitions(frame, config)
    x_train, y_train = train[selected_features], train["Consumption"]
    x_validation, y_validation = validation[selected_features], validation["Consumption"]

    trial_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    best_pipelines: dict[str, Pipeline] = {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for model_offset, model_name in enumerate(TUNED_MODELS):
        sampler = TPESampler(seed=config.random_seed + model_offset)
        study = optuna.create_study(direction="minimize", sampler=sampler, study_name=f"phase7_{model_name}")

        def objective(trial: optuna.Trial) -> float:
            pipeline = _suggested_pipeline(trial, model_name, config.random_seed)
            pipeline.fit(x_train, y_train)
            return regression_metrics(y_validation, pipeline.predict(x_validation))["mae"]

        study.optimize(objective, n_trials=config.trials_per_model, n_jobs=1)
        for trial in study.trials:
            trial_rows.append({"model": model_name, "trial_number": trial.number, "validation_mae": trial.value, "state": trial.state.name, "parameters_json": json.dumps(trial.params, sort_keys=True)})
        best = _fixed_pipeline(model_name, study.best_params, config.random_seed)
        best.fit(x_train, y_train)
        best_pipelines[model_name] = best
        validation_rows.append({"model": model_name, **regression_metrics(y_validation, best.predict(x_validation)), "best_parameters_json": json.dumps(study.best_params, sort_keys=True)})

    # Prespecified simple baselines are compared before selection; only tuned models
    # are eligible for the Optuna model-selection decision.
    naive_validation = validation["consumption_lag_1"].to_numpy(dtype=float)
    linear = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler()), ("model", LinearRegression())])
    linear.fit(x_train, y_train)
    validation_rows.extend([
        {"model": "naive_lag_1", **regression_metrics(y_validation, naive_validation), "best_parameters_json": "{}"},
        {"model": "linear_regression", **regression_metrics(y_validation, linear.predict(x_validation)), "best_parameters_json": "{}"},
    ])
    validation_metrics = pd.DataFrame(validation_rows).sort_values(["mae", "model"]).reset_index(drop=True)
    tuned_validation = validation_metrics[validation_metrics["model"].isin(TUNED_MODELS)]
    selected_model = str(tuned_validation.iloc[0]["model"])
    selected_pipeline = best_pipelines[selected_model]

    # Selection is frozen above. This is the first and only use of locked-test
    # targets in computation; no refit or preprocessing fit occurs afterwards.
    x_test, y_test = locked_test[selected_features], locked_test["Consumption"]
    predictions = pd.DataFrame({
        "Date": locked_test["Date"], "actual": y_test,
        "naive_lag_1": locked_test["consumption_lag_1"].to_numpy(dtype=float),
        "linear_regression": linear.predict(x_test),
        "selected_model": selected_pipeline.predict(x_test),
    })
    test_metrics = pd.DataFrame([
        {"model": "naive_lag_1", **regression_metrics(y_test, predictions["naive_lag_1"])},
        {"model": "linear_regression", **regression_metrics(y_test, predictions["linear_regression"])},
        {"model": selected_model, **regression_metrics(y_test, predictions["selected_model"])},
    ]).sort_values(["mae", "model"]).reset_index(drop=True)

    pd.DataFrame(trial_rows).to_csv(output / "optuna_trials.csv", index=False)
    validation_metrics.to_csv(output / "validation_metrics.csv", index=False)
    test_metrics.to_csv(output / "locked_test_metrics.csv", index=False)
    predictions.to_csv(output / "locked_test_predictions.csv", index=False, date_format="%Y-%m-%d")
    joblib.dump(selected_pipeline, output / "selected_model.joblib")
    joblib.dump(linear, output / "linear_baseline.joblib")
    _save_plot(predictions, selected_model, output / "locked_test_actual_vs_predicted.png")
    save_locked_test_diagnostics(predictions, selected_model, output)

    report = {
        "phase": "Phase 7", "experiment": "Test 3 chronological 70/15/15 with Optuna",
        "row_count": len(frame), "training_rows": len(train), "validation_rows": len(validation), "locked_test_rows": len(locked_test),
        "training_end_date_inclusive": train["Date"].iloc[-1].date().isoformat(),
        "validation_start_date": validation["Date"].iloc[0].date().isoformat(),
        "validation_end_date_inclusive": validation["Date"].iloc[-1].date().isoformat(),
        "locked_test_start_date": locked_test["Date"].iloc[0].date().isoformat(),
        "locked_test_end_date_inclusive": locked_test["Date"].iloc[-1].date().isoformat(),
        "chronological_order_preserved": True, "shuffle_used": False,
        "selected_features": selected_features, "feature_selection_source": "Phase 3 training-only 70% prefix",
        "tuned_models": list(TUNED_MODELS), "simple_baselines": ["naive_lag_1", "linear_regression"],
        "selection_metric": "validation MAE", "selected_model": selected_model,
        "preprocessing_fit_scope": "training partition only; validation and locked-test rows are transform/predict only",
        "locked_test_used_for_tuning_selection_or_refit": False,
        "locked_test_evaluations_after_selection": 1,
        "test_metrics_descriptive_only": True, "improvement_claimed": False,
        **hashes,
    }
    _write_json(output / "split_and_selection_report.json", report)
    _write_json(output / "config.json", {
        "phase": "Phase 7", "experiment_id": experiment_id, "configuration": asdict(config),
        "random_seed": config.random_seed, "optuna_sampler": "TPESampler",
        "phase2_directory": str(Path(phase2_dir).resolve()), "phase3_directory": str(Path(phase3_dir).resolve()),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__, "optuna": optuna.__version__},
        "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    (output / "execution.log").write_text(
        f"Phase 7 completed with seed {config.random_seed}.\n"
        f"Optuna ran {config.trials_per_model} train/validation trials for each of {len(TUNED_MODELS)} model families.\n"
        f"Selection was frozen as {selected_model} before one locked-test evaluation.\n"
        "No preprocessing was fit on validation/test data and no improvement claim was made.\n",
        encoding="utf-8",
    )
    return output


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2_DIR)
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-id")
    parser.add_argument("--trials-per-model", type=int, default=Phase7Config.trials_per_model)
    args = parser.parse_args(argv)
    config = Phase7Config(trials_per_model=args.trials_per_model)
    print(run_phase7_optuna(args.phase2_dir, args.phase3_dir, args.artifact_root, args.experiment_id, config))


if __name__ == "__main__":
    main()
