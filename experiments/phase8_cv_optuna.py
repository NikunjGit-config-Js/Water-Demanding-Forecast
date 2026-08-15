"""Run Phase 8: time-aware cross-validation with Optuna optimization."""

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
import optuna
import pandas as pd
import sklearn
from optuna.samplers import TPESampler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from experiments.phase4_traditional_ml import DEFAULT_PHASE2_DIR, DEFAULT_PHASE3_DIR, regression_metrics
from experiments.phase3_feature_selection import (
    SelectionConfig,
    _numeric_features,
    _validate_phase2_manifest,
    select_features,
)
from experiments.phase4_traditional_ml import sha256_file
from experiments.phase7_locked_test_optuna import TUNED_MODELS, _fixed_pipeline, _suggested_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ARTIFACT_ROOT = Path("artifacts/phase8")


@dataclass(frozen=True)
class Phase8Config:
    """Reproducible configuration for approved Test 4 methodology."""

    n_splits: int = 5
    trials_per_model: int = 20
    random_seed: int = 42

    def validate(self, row_count: int) -> None:
        if self.n_splits != 5:
            raise ValueError("Phase 8 requires exactly five time-aware folds")
        if self.trials_per_model < 1:
            raise ValueError("trials_per_model must be positive")
        if row_count <= self.n_splits:
            raise ValueError("dataset is too short for five-fold time-series cross-validation")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _load_candidate_frame(phase2_dir: Path, phase3_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    """Load the full Phase 2 candidate schema; Phase 3 is provenance only."""
    phase2_dir, phase3_dir = phase2_dir.resolve(), phase3_dir.resolve()
    manifest = _read_json(phase2_dir / "feature_manifest.json")
    _, target, candidates, row_count = _validate_phase2_manifest(manifest)
    features_path = phase2_dir / "features.csv"
    features_hash = sha256_file(features_path)
    phase3_report = _read_json(phase3_dir / "selection_report.json")
    if phase3_report.get("phase2_features_sha256") != features_hash:
        raise ValueError("Phase 2 features do not match the approved Phase 3 input")
    columns = ["Date", target, *candidates]
    frame = pd.read_csv(features_path, usecols=columns)
    if len(frame) != row_count or set(frame.columns) != set(columns):
        raise ValueError("Phase 2 feature artifact does not satisfy its manifest")
    frame = frame.loc[:, columns]
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if frame["Date"].duplicated().any() or not frame["Date"].is_monotonic_increasing:
        raise ValueError("dates must be unique and strictly chronological")
    frame[target] = pd.to_numeric(frame[target], errors="raise")
    if frame[target].isna().any() or not np.isfinite(frame[target]).all():
        raise ValueError("target must be finite and complete")
    if "consumption_lag_1" not in candidates:
        raise ValueError("naive lag-1 baseline is unavailable")
    return frame, manifest, {
        "phase2_features_sha256": features_hash,
        "phase3_selection_report_sha256": sha256_file(phase3_dir / "selection_report.json"),
    }


def _artifact_hashes(output: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file()):
        if path.name != "artifact_hashes.json":
            hashes[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def expanding_splits(row_count: int, config: Phase8Config) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return strictly chronological expanding-window folds used by every trial."""
    config.validate(row_count)
    splits = list(TimeSeriesSplit(n_splits=config.n_splits).split(np.arange(row_count)))
    for train_index, validation_index in splits:
        if train_index[-1] >= validation_index[0] or not np.array_equal(
            train_index, np.arange(train_index[-1] + 1)
        ):
            raise RuntimeError("cross-validation folds are not expanding and chronological")
    return splits


def _linear_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("model", LinearRegression()),
    ])


def _save_diagnostics(predictions: pd.DataFrame, selected_model: str, output: Path) -> None:
    ordered = predictions.sort_values(["fold", "Date"])
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(ordered["Date"], ordered["actual"], label="Actual", color="black", linewidth=1)
    axis.plot(ordered["Date"], ordered["selected_model"], label=selected_model, linewidth=1)
    axis.plot(ordered["Date"], ordered["naive_lag_1"], label="Naive lag-1", alpha=0.65)
    axis.set(title="Phase 8 out-of-fold predictions", xlabel="Date", ylabel="Consumption")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "actual_vs_predicted.png", dpi=150)
    plt.close(figure)

    residual = ordered["actual"] - ordered["selected_model"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(ordered["selected_model"], residual, s=10, alpha=0.6)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Predicted", ylabel="Residual", title=f"Residuals: {selected_model}")
    axes[1].hist(residual, bins=30)
    axes[1].set(xlabel="Residual", ylabel="Count", title="Residual distribution")
    figure.tight_layout()
    figure.savefig(output / "residual_diagnostics.png", dpi=150)
    plt.close(figure)


def run_phase8_cv_optuna(
    phase2_dir: Path = DEFAULT_PHASE2_DIR,
    phase3_dir: Path = DEFAULT_PHASE3_DIR,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    experiment_id: str | None = None,
    config: Phase8Config | None = None,
) -> Path:
    """Tune with fold-mean MAE, then compare frozen parameters on the same CV folds."""
    config = config or Phase8Config()
    experiment_id = experiment_id or datetime.now(timezone.utc).strftime("phase8_%Y%m%dT%H%M%SZ")
    output = artifact_root.resolve() / experiment_id
    output.mkdir(parents=True, exist_ok=False)
    models_dir = output / "models"
    models_dir.mkdir()
    started = datetime.now(timezone.utc)

    frame, manifest, hashes = _load_candidate_frame(phase2_dir, phase3_dir)
    splits = expanding_splits(len(frame), config)
    candidates = list(manifest["feature_columns"])
    y = frame["Consumption"]
    model_features = _numeric_features(frame, candidates)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Final selection is fitted independently on each outer-fold training prefix.
    # Tuning below uses separate selection fitted on every inner training prefix.
    selection_config = SelectionConfig(random_seed=config.random_seed)
    fold_features: dict[int, list[str]] = {}
    fold_selection_reports: dict[int, dict[str, Any]] = {}
    for fold_number, (train_index, _) in enumerate(splits, start=1):
        training_frame = frame.iloc[train_index].loc[:, ["Date", "Consumption", *candidates]].copy()
        ranking, selection_report = select_features(training_frame, manifest, selection_config)
        selected = list(selection_report["selected_features"])
        fold_features[fold_number] = selected
        fold_selection_reports[fold_number] = selection_report
        ranking.to_csv(output / f"fold_{fold_number}_feature_ranking.csv", index=False)
        _write_json(output / f"fold_{fold_number}_feature_selection.json", selection_report)

    trial_rows: list[dict[str, Any]] = []
    best_parameters: dict[str, dict[str, dict[str, Any]]] = {}
    best_trial_rows: list[dict[str, Any]] = []
    fold_selected_models: dict[int, str] = {}
    for outer_fold, (outer_train, _) in enumerate(splits, start=1):
        inner_splits = expanding_splits(len(outer_train), config)
        inner_features: dict[int, list[str]] = {}
        for inner_fold, (inner_train, _) in enumerate(inner_splits, start=1):
            inner_frame = frame.iloc[outer_train[inner_train]].loc[:, ["Date", "Consumption", *candidates]].copy()
            _, inner_report = select_features(inner_frame, manifest, selection_config)
            inner_features[inner_fold] = list(inner_report["selected_features"])

        fold_key = str(outer_fold)
        best_parameters[fold_key] = {}
        for model_offset, model_name in enumerate(TUNED_MODELS):
            study = optuna.create_study(
                direction="minimize",
                sampler=TPESampler(seed=config.random_seed + outer_fold * 100 + model_offset),
                study_name=f"phase8_outer_{outer_fold}_{model_name}",
            )

            def objective(trial: optuna.Trial) -> float:
                inner_maes: list[float] = []
                for inner_fold, (inner_train, inner_validation) in enumerate(inner_splits, start=1):
                    selected = inner_features[inner_fold]
                    fit_rows = outer_train[inner_train]
                    assess_rows = outer_train[inner_validation]
                    pipeline = _suggested_pipeline(trial, model_name, config.random_seed)
                    pipeline.fit(model_features.iloc[fit_rows][selected], y.iloc[fit_rows])
                    prediction = pipeline.predict(model_features.iloc[assess_rows][selected])
                    inner_maes.append(regression_metrics(y.iloc[assess_rows], prediction)["mae"])
                return float(np.mean(inner_maes))

            study.optimize(objective, n_trials=config.trials_per_model, n_jobs=1)
            best_parameters[fold_key][model_name] = study.best_params
            best_trial_rows.append({
                "outer_fold": outer_fold,
                "model": model_name,
                "selected_trial_number": study.best_trial.number,
                "inner_cv_mean_mae": study.best_value,
                "parameters_json": json.dumps(study.best_params, sort_keys=True),
            })
            for trial in study.trials:
                trial_rows.append({
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "trial_number": trial.number,
                    "inner_cv_mean_mae": trial.value,
                    "state": trial.state.name,
                    "parameters_json": json.dumps(trial.params, sort_keys=True),
                })
        fold_best = min(
            (row for row in best_trial_rows if row["outer_fold"] == outer_fold),
            key=lambda row: (row["inner_cv_mean_mae"], row["model"]),
        )
        fold_selected_models[outer_fold] = str(fold_best["model"])

    best_trial_metrics = pd.DataFrame(best_trial_rows).sort_values(
        ["outer_fold", "inner_cv_mean_mae", "model"]
    ).reset_index(drop=True)

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
        selected = fold_features[fold_number]
        fold_model_dir = models_dir / f"fold_{fold_number}"
        fold_model_dir.mkdir()
        fold_predictions = pd.DataFrame({
            "Date": frame.iloc[validation_index]["Date"].to_numpy(),
            "fold": fold_number,
            "actual": y.iloc[validation_index].to_numpy(),
        })
        naive = frame.iloc[validation_index]["consumption_lag_1"].to_numpy(dtype=float)
        fold_predictions["naive_lag_1"] = naive
        metric_rows.append({"fold": fold_number, "model": "naive_lag_1", **regression_metrics(y.iloc[validation_index], naive)})

        linear = _linear_pipeline()
        linear.fit(model_features.iloc[train_index][selected], y.iloc[train_index])
        linear_prediction = linear.predict(model_features.iloc[validation_index][selected])
        fold_predictions["linear_regression"] = linear_prediction
        metric_rows.append({"fold": fold_number, "model": "linear_regression", **regression_metrics(y.iloc[validation_index], linear_prediction)})
        joblib.dump(linear, fold_model_dir / "linear_regression.joblib")

        for model_name in TUNED_MODELS:
            pipeline = _fixed_pipeline(model_name, best_parameters[str(fold_number)][model_name], config.random_seed)
            pipeline.fit(model_features.iloc[train_index][selected], y.iloc[train_index])
            prediction = pipeline.predict(model_features.iloc[validation_index][selected])
            metric_rows.append({"fold": fold_number, "model": model_name, **regression_metrics(y.iloc[validation_index], prediction)})
            joblib.dump(pipeline, fold_model_dir / f"{model_name}.joblib")
            if model_name == fold_selected_models[fold_number]:
                fold_predictions["selected_model"] = prediction
        prediction_frames.append(fold_predictions)
        fold_reports.append({
            "fold": fold_number,
            "training_rows": len(train_index),
            "validation_rows": len(validation_index),
            "training_start_date": frame.iloc[train_index[0]]["Date"].date().isoformat(),
            "training_end_date_inclusive": frame.iloc[train_index[-1]]["Date"].date().isoformat(),
            "validation_start_date": frame.iloc[validation_index[0]]["Date"].date().isoformat(),
            "validation_end_date_inclusive": frame.iloc[validation_index[-1]]["Date"].date().isoformat(),
            "selected_features": selected,
            "selected_feature_count": len(selected),
            "feature_selection_rows": len(train_index),
            "feature_selection_end_row_inclusive": int(train_index[-1]),
            "feature_selection_used_validation_rows": False,
            "selected_tuned_model": fold_selected_models[fold_number],
            "selected_hyperparameters": best_parameters[str(fold_number)],
        })

    fold_metrics = pd.DataFrame(metric_rows).sort_values(["fold", "model"]).reset_index(drop=True)
    summary = fold_metrics.groupby("model")[["mae", "mse", "rmse", "r2"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values(["mae_mean", "model"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pd.DataFrame(trial_rows).to_csv(output / "optuna_trials.csv", index=False)
    best_trial_metrics.to_csv(output / "tuned_model_selection.csv", index=False)
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False, date_format="%Y-%m-%d")
    _save_diagnostics(predictions, "fold-local inner-CV winner", output)

    _write_json(output / "cv_selection_report.json", {
        "phase": "Phase 8",
        "experiment": "Test 4 TimeSeriesSplit plus Optuna",
        "row_count": len(frame),
        "n_splits": config.n_splits,
        "shuffle_used": False,
        "folds": fold_reports,
        "candidate_features": candidates,
        "feature_selection_source": "Phase 3 methodology refitted independently on each outer-fold training prefix",
        "phase3_fixed_feature_set_used_for_modeling": False,
        "fold_selected_features": {str(key): value for key, value in fold_features.items()},
        "tuned_models": list(TUNED_MODELS),
        "simple_baselines": ["naive_lag_1", "linear_regression"],
        "selection_metric": "mean inner-validation MAE within each outer-fold training prefix",
        "fold_selected_models": {str(key): value for key, value in fold_selected_models.items()},
        "best_parameters_by_outer_fold": best_parameters,
        "nested_tuning": True,
        "outer_validation_used_for_hyperparameter_selection": False,
        "feature_selection_fit_scope": "each outer-fold training prefix only; deterministic fold result reused across trials because it is hyperparameter-independent",
        "preprocessing_fit_scope": "each trial/final fold training prefix only; fold validation rows are transform/predict only",
        "validation_folds_used_for_preprocessing_fit": False,
        "external_holdout_or_locked_test_used": False,
        "full_data_refit_performed": False,
        "improvement_claimed": False,
        **hashes,
    })
    _write_json(output / "config.json", {
        "phase": "Phase 8",
        "experiment_id": experiment_id,
        "configuration": asdict(config),
        "random_seed": config.random_seed,
        "optuna_sampler": "TPESampler",
        "phase2_directory": str(Path(phase2_dir).resolve()),
        "phase3_directory": str(Path(phase3_dir).resolve()),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "joblib": joblib.__version__, "optuna": optuna.__version__,
        },
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    (output / "execution.log").write_text(
        f"Phase 8 completed with seed {config.random_seed}.\n"
        f"Optuna ran {config.trials_per_model} trials for each of {len(TUNED_MODELS)} model families over five expanding folds.\n"
        f"Each outer fold selected parameters and a tuned candidate using only nested chronological validation in its training prefix: {fold_selected_models}.\n"
        "Feature selection and every preprocessor were fit within each fold training prefix; no external holdout or locked test was used.\n"
        "Simple baselines are reported and no improvement claim was made.\n",
        encoding="utf-8",
    )
    _write_json(output / "artifact_hashes.json", _artifact_hashes(output))
    return output


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2_DIR)
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-id")
    parser.add_argument("--trials-per-model", type=int, default=Phase8Config.trials_per_model)
    args = parser.parse_args(argv)
    config = Phase8Config(trials_per_model=args.trials_per_model)
    print(run_phase8_cv_optuna(args.phase2_dir, args.phase3_dir, args.artifact_root, args.experiment_id, config))


if __name__ == "__main__":
    main()
