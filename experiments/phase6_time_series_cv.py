"""Run Phase 6 five-fold expanding-window time-series cross-validation."""

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
import pandas as pd
import sklearn
from sklearn.model_selection import TimeSeriesSplit

from experiments.phase4_traditional_ml import (
    DEFAULT_PHASE2_DIR,
    DEFAULT_PHASE3_DIR,
    Phase4Config,
    SCALED_MODELS,
    build_model_registry,
    regression_metrics,
)
from experiments.phase5_chronological_holdout import Phase5Config, load_phase5_frame

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ARTIFACT_ROOT = Path("artifacts/phase6")


@dataclass(frozen=True)
class Phase6Config:
    """Reproducible configuration for approved Test 2 methodology."""

    n_splits: int = 5
    random_seed: int = 42

    def validate(self) -> None:
        if self.n_splits != 5:
            raise ValueError("Phase 6 requires exactly five time-aware folds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expanding_splits(row_count: int, config: Phase6Config) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return five strictly chronological expanding-window split indices."""
    config.validate()
    splits = list(TimeSeriesSplit(n_splits=config.n_splits).split(np.arange(row_count)))
    for train_index, validation_index in splits:
        if train_index[-1] >= validation_index[0] or not np.array_equal(
            train_index, np.arange(train_index[-1] + 1)
        ):
            raise RuntimeError("cross-validation folds are not expanding and chronological")
    return splits


def _save_diagnostics(predictions: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    best_name = str(summary.loc[summary["model"] != "naive_lag_1"].iloc[0]["model"])
    ordered = predictions.sort_values("Date")
    figure, axis = plt.subplots(figsize=(12, 5))
    for fold, fold_rows in ordered.groupby("fold", sort=True):
        axis.plot(fold_rows["Date"], fold_rows["actual"], color="black", linewidth=1)
        axis.plot(fold_rows["Date"], fold_rows[best_name], linewidth=1, label=f"Fold {fold}")
    axis.set(title=f"Phase 6 expanding-window predictions: {best_name}", xlabel="Date", ylabel="Consumption")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output / "actual_vs_predicted.png", dpi=150)
    plt.close(figure)

    residual = ordered["actual"] - ordered[best_name]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(ordered[best_name], residual, s=10, alpha=0.6)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Predicted", ylabel="Residual", title=f"Residuals: {best_name}")
    axes[1].hist(residual, bins=30)
    axes[1].set(xlabel="Residual", ylabel="Count", title="Residual distribution")
    figure.tight_layout()
    figure.savefig(output / "residual_diagnostics.png", dpi=150)
    plt.close(figure)


def run_phase6_cv(
    phase2_dir: Path = DEFAULT_PHASE2_DIR,
    phase3_dir: Path = DEFAULT_PHASE3_DIR,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    experiment_id: str | None = None,
    config: Phase6Config | None = None,
    model_config: Phase4Config | None = None,
) -> Path:
    """Fit every model independently within each chronological fold and save evidence."""
    config = config or Phase6Config()
    config.validate()
    model_config = model_config or Phase4Config(random_seed=config.random_seed)
    if model_config.random_seed != config.random_seed:
        raise ValueError("cross-validation and model random seeds must match")
    experiment_id = experiment_id or datetime.now(timezone.utc).strftime("phase6_%Y%m%dT%H%M%SZ")
    output = artifact_root.resolve() / experiment_id
    output.mkdir(parents=True, exist_ok=False)
    models_dir = output / "models"
    models_dir.mkdir()
    started = datetime.now(timezone.utc)

    # Phase 6 evaluates the full chronology. Phase5Config is used only for its
    # already-validated upstream loader; its returned 80/20 boundary is ignored.
    frame, selected, _, hashes = load_phase5_frame(phase2_dir, phase3_dir, Phase5Config())
    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_manifest: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(expanding_splits(len(frame), config), start=1):
        train, validation = frame.iloc[train_index], frame.iloc[validation_index]
        x_train, y_train = train[selected], train["Consumption"]
        x_validation, y_validation = validation[selected], validation["Consumption"]
        fold_predictions = pd.DataFrame({
            "fold": fold, "Date": validation["Date"], "actual": y_validation,
            "naive_lag_1": validation["consumption_lag_1"].to_numpy(dtype=float),
        })
        if not np.isfinite(fold_predictions["naive_lag_1"]).all():
            raise ValueError(f"naive lag-1 baseline is incomplete in fold {fold}")
        fold_rows.append({"fold": fold, "model": "naive_lag_1", **regression_metrics(y_validation, fold_predictions["naive_lag_1"])})
        registry = build_model_registry(model_config)
        fold_model_dir = models_dir / f"fold_{fold}"
        fold_model_dir.mkdir()
        for name, model in registry.items():
            model.fit(x_train, y_train)
            predicted = model.predict(x_validation)
            if not np.isfinite(predicted).all():
                raise RuntimeError(f"{name} produced non-finite predictions in fold {fold}")
            fold_predictions[name] = predicted
            fold_rows.append({"fold": fold, "model": name, **regression_metrics(y_validation, predicted)})
            joblib.dump(model, fold_model_dir / f"{name}.joblib")
        prediction_frames.append(fold_predictions)
        fold_manifest.append({
            "fold": fold,
            "training_rows": len(train),
            "validation_rows": len(validation),
            "training_start_date": train["Date"].iloc[0].date().isoformat(),
            "training_end_date_inclusive": train["Date"].iloc[-1].date().isoformat(),
            "validation_start_date": validation["Date"].iloc[0].date().isoformat(),
            "validation_end_date_inclusive": validation["Date"].iloc[-1].date().isoformat(),
        })

    metrics = pd.DataFrame(fold_rows).sort_values(["fold", "mae", "model"]).reset_index(drop=True)
    metric_columns = ["mae", "mse", "rmse", "r2"]
    summary = metrics.groupby("model", as_index=False)[metric_columns].agg(["mean", "std"])
    summary.columns = ["model", *[f"{metric}_{stat}" for metric in metric_columns for stat in ("mean", "std")]]
    summary = summary.sort_values(["mae_mean", "model"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False, date_format="%Y-%m-%d")
    _save_diagnostics(predictions, summary, output)

    report = {
        "phase": "Phase 6",
        "experiment": "Test 2 five-fold expanding-window cross-validation",
        "row_count": len(frame),
        "n_splits": config.n_splits,
        "splitter": "sklearn.model_selection.TimeSeriesSplit",
        "expanding_window": True,
        "chronological_order_preserved": True,
        "shuffle_used": False,
        "folds": fold_manifest,
        "selected_features": selected,
        "simple_baselines": ["naive_lag_1", "linear_regression"],
        "model_names": list(build_model_registry(model_config)),
        "preprocessing_fit_scope": "each model Pipeline fitted independently on each fold's training rows only",
        "scaled_models": sorted(SCALED_MODELS),
        "validation_folds_used_for_fitting_or_tuning": False,
        "hyperparameter_optimization_performed": False,
        "improvement_claimed": False,
        "lowest_mean_cv_mae_model_descriptive_only": str(summary.iloc[0]["model"]),
        "metric_aggregation": "unweighted mean and sample standard deviation across five folds",
        **hashes,
    }
    _write_json(output / "cv_report.json", report)
    _write_json(output / "config.json", {
        "phase": "Phase 6", "experiment_id": experiment_id,
        "configuration": asdict(config), "model_configuration": asdict(model_config),
        "random_seed": config.random_seed,
        "phase2_directory": str(Path(phase2_dir).resolve()),
        "phase3_directory": str(Path(phase3_dir).resolve()),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    (output / "execution.log").write_text(
        f"Phase 6 five-fold expanding-window CV completed.\nRows: {len(frame)}; folds: {config.n_splits}.\n"
        f"Fitted {len(build_model_registry(model_config))} approved models independently per fold with seed {config.random_seed}.\n"
        "No tuning, locked Test 3 evaluation, or improvement claim was made.\n",
        encoding="utf-8",
    )
    return output


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2_DIR)
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-id")
    args = parser.parse_args(argv)
    print(run_phase6_cv(args.phase2_dir, args.phase3_dir, args.artifact_root, args.experiment_id))


if __name__ == "__main__":
    main()
