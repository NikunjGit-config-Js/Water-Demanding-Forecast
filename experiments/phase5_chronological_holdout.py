"""Run the formal Phase 5 chronological 80/20 traditional-ML holdout."""

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

from experiments.phase4_traditional_ml import (
    DEFAULT_PHASE2_DIR,
    DEFAULT_PHASE3_DIR,
    Phase4Config,
    SCALED_MODELS,
    build_model_registry,
    regression_metrics,
    sha256_file,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ARTIFACT_ROOT = Path("artifacts/phase5")
SEASON_CODES = {"winter": 0.0, "spring": 1.0, "summer": 2.0, "autumn": 3.0}


@dataclass(frozen=True)
class Phase5Config:
    """Reproducible configuration for the approved Test 1 methodology."""

    training_fraction: float = 0.80
    random_seed: int = 42

    def split_row(self, row_count: int) -> int:
        if not 0.0 < self.training_fraction < 1.0:
            raise ValueError("training_fraction must be strictly between zero and one")
        split = int(np.floor(row_count * self.training_fraction))
        if split < 2 or split >= row_count:
            raise ValueError("dataset is too short for the requested holdout")
        return split


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_phase5_frame(
    phase2_dir: Path, phase3_dir: Path, config: Phase5Config
) -> tuple[pd.DataFrame, list[str], int, dict[str, str]]:
    """Load the selected schema and verify it against approved upstream evidence."""
    phase2_dir, phase3_dir = phase2_dir.resolve(), phase3_dir.resolve()
    manifest = _read_json(phase2_dir / "feature_manifest.json")
    selection = _read_json(phase3_dir / "selection_report.json")
    candidates, selected, row_count = (
        manifest.get("feature_columns"),
        selection.get("selected_features"),
        manifest.get("row_count"),
    )
    if not isinstance(candidates, list) or not isinstance(selected, list) or not selected:
        raise ValueError("upstream feature manifests are invalid")
    if len(selected) != len(set(selected)) or not set(selected).issubset(candidates):
        raise ValueError("Phase 3 selected features do not match the Phase 2 manifest")
    if not isinstance(row_count, int):
        raise ValueError("Phase 2 row count is invalid")
    features_path = phase2_dir / "features.csv"
    features_hash = sha256_file(features_path)
    if selection.get("phase2_features_sha256") != features_hash:
        raise ValueError("Phase 2 features do not match the approved Phase 3 input")
    if "consumption_lag_1" not in candidates:
        raise ValueError("naive lag-1 baseline is unavailable")
    auxiliary = [] if "consumption_lag_1" in selected else ["consumption_lag_1"]
    columns = ["Date", "Consumption", *selected, *auxiliary]
    frame = pd.read_csv(features_path, usecols=columns)
    if len(frame) != row_count or set(frame.columns) != set(columns):
        raise ValueError("Phase 2 feature artifact does not satisfy its manifest")
    frame = frame.loc[:, columns]
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if frame["Date"].duplicated().any() or not frame["Date"].is_monotonic_increasing:
        raise ValueError("dates must be unique and strictly chronological")
    frame["Consumption"] = pd.to_numeric(frame["Consumption"], errors="raise")
    if frame["Consumption"].isna().any() or not np.isfinite(frame["Consumption"]).all():
        raise ValueError("target must be finite and complete")
    for column in selected:
        frame[column] = (
            frame[column].map(SEASON_CODES)
            if column == "season"
            else pd.to_numeric(frame[column], errors="raise")
        )
    frame[selected] = frame[selected].replace([np.inf, -np.inf], np.nan).astype(float)
    return frame, selected, config.split_row(row_count), {
        "phase2_features_sha256": features_hash,
        "phase3_selection_report_sha256": sha256_file(phase3_dir / "selection_report.json"),
    }


def _save_evaluation_artifacts(
    predictions: pd.DataFrame, metrics: pd.DataFrame, output: Path
) -> None:
    """Preserve the required holdout diagnostics for one clearly named model."""
    best_name = str(metrics.loc[metrics["model"] != "naive_lag_1"].iloc[0]["model"])
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(predictions["Date"], predictions["actual"], label="Actual", linewidth=1.5)
    axis.plot(predictions["Date"], predictions["naive_lag_1"], label="Naive lag-1", alpha=0.7)
    axis.plot(predictions["Date"], predictions["linear_regression"], label="Linear regression", alpha=0.7)
    if best_name != "linear_regression":
        axis.plot(predictions["Date"], predictions[best_name], label=f"Lowest MAE: {best_name}", alpha=0.8)
    axis.set(title="Phase 5 chronological 80/20 holdout", xlabel="Date", ylabel="Consumption")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "actual_vs_predicted.png", dpi=150)
    plt.close(figure)

    residual = predictions["actual"] - predictions[best_name]
    absolute_error = residual.abs()

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(predictions["actual"], predictions[best_name], s=12, alpha=0.65)
    lower = float(min(predictions["actual"].min(), predictions[best_name].min()))
    upper = float(max(predictions["actual"].max(), predictions[best_name].max()))
    axis.plot([lower, upper], [lower, upper], color="black", linewidth=1, linestyle="--")
    axis.set(
        xlabel="Actual consumption",
        ylabel="Predicted consumption",
        title=f"Actual versus predicted: {best_name}",
    )
    figure.tight_layout()
    figure.savefig(output / "actual_vs_predicted_scatter.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(predictions[best_name], residual, s=10, alpha=0.6)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Predicted", ylabel="Residual", title=f"Residuals: {best_name}")
    axes[1].hist(residual, bins=30)
    axes[1].set(xlabel="Residual", ylabel="Count", title="Residual distribution")
    figure.tight_layout()
    figure.savefig(output / "residual_diagnostics.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(predictions["Date"], residual, label="Residual (actual - prediction)", alpha=0.75)
    axis.plot(predictions["Date"], absolute_error, label="Absolute error", linewidth=1.5)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Date", ylabel="Error", title=f"Dated holdout error: {best_name}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "error_over_time.png", dpi=150)
    plt.close(figure)

    highest_errors = pd.DataFrame({
        "date": predictions["Date"],
        "model": best_name,
        "actual": predictions["actual"],
        "prediction": predictions[best_name],
        "residual": residual,
        "absolute_error": absolute_error,
    }).sort_values(["absolute_error", "date"], ascending=[False, True])
    highest_errors.head(min(20, len(highest_errors))).to_csv(
        output / "highest_error_dates.csv", index=False, date_format="%Y-%m-%d"
    )


def run_phase5_holdout(
    phase2_dir: Path = DEFAULT_PHASE2_DIR,
    phase3_dir: Path = DEFAULT_PHASE3_DIR,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    experiment_id: str | None = None,
    config: Phase5Config | None = None,
    model_config: Phase4Config | None = None,
) -> Path:
    config = config or Phase5Config()
    model_config = model_config or Phase4Config(random_seed=config.random_seed)
    if model_config.random_seed != config.random_seed:
        raise ValueError("holdout and model random seeds must match")
    experiment_id = experiment_id or datetime.now(timezone.utc).strftime("phase5_%Y%m%dT%H%M%SZ")
    output = artifact_root.resolve() / experiment_id
    output.mkdir(parents=True, exist_ok=False)
    models_dir = output / "models"
    models_dir.mkdir()
    started = datetime.now(timezone.utc)
    frame, selected, split, hashes = load_phase5_frame(phase2_dir, phase3_dir, config)
    train, test = frame.iloc[:split], frame.iloc[split:]
    x_train, y_train = train[selected], train["Consumption"]
    x_test, y_test = test[selected], test["Consumption"]
    predictions = pd.DataFrame({"Date": test["Date"], "actual": y_test})
    naive = test["consumption_lag_1"].to_numpy(dtype=float)
    if not np.isfinite(naive).all():
        raise ValueError("naive lag-1 baseline is incomplete on the holdout")
    predictions["naive_lag_1"] = naive
    metric_rows = [{"model": "naive_lag_1", **regression_metrics(y_test, naive)}]
    registry = build_model_registry(model_config)
    for name, model in registry.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        if not np.isfinite(predicted).all():
            raise RuntimeError(f"{name} produced non-finite predictions")
        predictions[name] = predicted
        metric_rows.append({"model": name, **regression_metrics(y_test, predicted)})
        joblib.dump(model, models_dir / f"{name}.joblib")
    metrics = pd.DataFrame(metric_rows).sort_values(["mae", "model"]).reset_index(drop=True)
    predictions.to_csv(output / "predictions.csv", index=False, date_format="%Y-%m-%d")
    metrics.to_csv(output / "metrics.csv", index=False)
    _save_evaluation_artifacts(predictions, metrics, output)
    report = {
        "phase": "Phase 5",
        "experiment": "Test 1 chronological 80/20 holdout",
        "row_count": len(frame),
        "training_rows": len(train),
        "holdout_rows": len(test),
        "training_start_date": train["Date"].iloc[0].date().isoformat(),
        "training_end_date_inclusive": train["Date"].iloc[-1].date().isoformat(),
        "holdout_start_date": test["Date"].iloc[0].date().isoformat(),
        "holdout_end_date_inclusive": test["Date"].iloc[-1].date().isoformat(),
        "chronological_order_preserved": True,
        "shuffle_used": False,
        "selected_features": selected,
        "simple_baselines": ["naive_lag_1", "linear_regression"],
        "model_names": list(registry),
        "preprocessing_fit_scope": "each Pipeline fitted on chronological training rows only",
        "scaled_models": sorted(SCALED_MODELS),
        "holdout_used_for_fitting_or_tuning": False,
        "hyperparameter_optimization_performed": False,
        "improvement_claimed": False,
        "lowest_holdout_mae_model_descriptive_only": str(metrics.iloc[0]["model"]),
        **hashes,
    }
    _write_json(output / "holdout_report.json", report)
    _write_json(output / "config.json", {
        "phase": "Phase 5", "experiment_id": experiment_id,
        "configuration": asdict(config), "model_configuration": asdict(model_config),
        "random_seed": config.random_seed,
        "phase2_directory": str(Path(phase2_dir).resolve()),
        "phase3_directory": str(Path(phase3_dir).resolve()),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    (output / "execution.log").write_text(
        f"Phase 5 chronological 80/20 holdout completed.\nTraining rows: {len(train)}; holdout rows: {len(test)}.\n"
        f"Fitted {len(registry)} approved models with seed {config.random_seed}.\nNo tuning or improvement claim was made.\n",
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
    print(run_phase5_holdout(args.phase2_dir, args.phase3_dir, args.artifact_root, args.experiment_id))


if __name__ == "__main__":
    main()
