"""Phase 10: evaluate a fixed PatchTST candidate without opening locked test data."""

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
from typing import Any, Iterable

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET = Path("data/preprocessed/all/preprocessed_data.csv")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/phase10")


@dataclass(frozen=True)
class Phase10Config:
    """Predeclared PatchTST configuration evaluated on validation, never test."""

    expected_total_rows: int = 3800
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    context_length: int = 56
    patch_length: int = 7
    patch_stride: int = 7
    d_model: int = 32
    num_heads: int = 4
    encoder_layers: int = 2
    feedforward_dim: int = 64
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 40
    batch_size: int = 32
    patience: int = 6
    random_seed: int = 42

    def split_rows(self) -> tuple[int, int, int]:
        train_end = int(np.floor(self.expected_total_rows * self.train_fraction))
        validation_end = train_end + int(np.floor(self.expected_total_rows * self.validation_fraction))
        if not (self.context_length < train_end < validation_end < self.expected_total_rows):
            raise ValueError("invalid chronological split")
        if self.context_length < self.patch_length or self.patch_stride < 1:
            raise ValueError("invalid patch geometry")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        return train_end, validation_end, self.expected_total_rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_development_prefix(dataset: Path, config: Phase10Config) -> pd.DataFrame:
    """Read train+validation only; the locked-test rows are never parsed."""
    _, validation_end, _ = config.split_rows()
    frame = pd.read_csv(dataset, usecols=["Date", "Consumption"], nrows=validation_end)
    if len(frame) != validation_end:
        raise ValueError(f"expected at least {validation_end} development rows, found {len(frame)}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame["Consumption"] = pd.to_numeric(frame["Consumption"], errors="raise")
    if frame["Date"].duplicated().any() or not frame["Date"].is_monotonic_increasing:
        raise ValueError("development dates must be unique and chronological")
    if frame["Consumption"].isna().any() or not np.isfinite(frame["Consumption"]).all():
        raise ValueError("development targets must be finite and complete")
    return frame


def make_one_step_windows(values: np.ndarray, context_length: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) <= context_length:
        raise ValueError("not enough observations for the requested context")
    x = np.stack([values[i - context_length:i] for i in range(context_length, len(values))])
    return x[..., np.newaxis], values[context_length:]


def configure_tensorflow(seed: int) -> tuple[Any, str]:
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
    return tf, "GPU" if gpus else "CPU"


def build_patchtst(config: Phase10Config, tf: Any) -> Any:
    """Build a compact channel-independent PatchTST-style one-step model."""
    inputs = tf.keras.Input((config.context_length, 1), name="past_target")
    patches = tf.keras.layers.Conv1D(
        config.d_model, config.patch_length, strides=config.patch_stride,
        padding="valid", name="patch_projection",
    )(inputs)
    patch_count = (config.context_length - config.patch_length) // config.patch_stride + 1
    positions = tf.keras.layers.Embedding(patch_count, config.d_model, name="positional_embedding")(
        tf.range(patch_count)
    )
    encoded = patches + positions
    for index in range(config.encoder_layers):
        attention_input = tf.keras.layers.LayerNormalization(name=f"attention_norm_{index}")(encoded)
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=config.num_heads, key_dim=config.d_model // config.num_heads,
            dropout=config.dropout, name=f"self_attention_{index}",
        )(attention_input, attention_input)
        encoded = encoded + tf.keras.layers.Dropout(config.dropout)(attention)
        feedforward_input = tf.keras.layers.LayerNormalization(name=f"ffn_norm_{index}")(encoded)
        feedforward = tf.keras.layers.Dense(config.feedforward_dim, activation="gelu")(feedforward_input)
        feedforward = tf.keras.layers.Dropout(config.dropout)(feedforward)
        feedforward = tf.keras.layers.Dense(config.d_model)(feedforward)
        encoded = encoded + tf.keras.layers.Dropout(config.dropout)(feedforward)
    encoded = tf.keras.layers.LayerNormalization(name="output_norm")(encoded)
    outputs = tf.keras.layers.Dense(1, name="forecast_head")(tf.keras.layers.Flatten()(encoded))
    model = tf.keras.Model(inputs, outputs, name="patchtst_univariate")
    model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate), loss="mse")
    return model


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(actual, predicted))
    return {"mae": float(mean_absolute_error(actual, predicted)), "mse": mse,
            "rmse": float(np.sqrt(mse)), "r2": float(r2_score(actual, predicted))}


def _save_plots(predictions: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(predictions["Date"], predictions["actual"], label="Actual", color="black", linewidth=1.2)
    axis.plot(predictions["Date"], predictions["patchtst"], label="PatchTST", alpha=0.8)
    axis.plot(predictions["Date"], predictions["naive_lag1"], label="Naive lag-1", alpha=0.65)
    axis.set(title="Phase 10 validation forecasts", xlabel="Date", ylabel="Consumption")
    axis.legend(); figure.tight_layout(); figure.savefig(output / "actual_vs_predicted.png", dpi=150); plt.close(figure)
    residual = predictions["actual"] - predictions["patchtst"]
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(predictions["actual"], predictions["patchtst"], s=12, alpha=0.65)
    lower = float(min(predictions["actual"].min(), predictions["patchtst"].min()))
    upper = float(max(predictions["actual"].max(), predictions["patchtst"].max()))
    axis.plot([lower, upper], [lower, upper], color="black", linestyle="--")
    axis.set(xlabel="Actual", ylabel="Predicted", title="PatchTST actual versus predicted")
    figure.tight_layout(); figure.savefig(output / "actual_vs_predicted_scatter.png", dpi=150); plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(predictions["patchtst"], residual, s=10, alpha=0.6); axes[0].axhline(0, color="black")
    axes[0].set(xlabel="Predicted", ylabel="Residual", title="PatchTST residuals")
    axes[1].hist(residual, bins=30); axes[1].set(xlabel="Residual", ylabel="Count", title="Residual distribution")
    figure.tight_layout(); figure.savefig(output / "residual_diagnostics.png", dpi=150); plt.close(figure)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(predictions["Date"], residual, label="Residual", alpha=0.75)
    axis.plot(predictions["Date"], residual.abs(), label="Absolute error", linewidth=1.2)
    axis.axhline(0, color="black", linewidth=1); axis.legend()
    axis.set(xlabel="Date", ylabel="Error", title="PatchTST validation error over time")
    figure.tight_layout(); figure.savefig(output / "error_over_time.png", dpi=150); plt.close(figure)


def run_phase10(dataset: Path = DEFAULT_DATASET, artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
                experiment_id: str | None = None, config: Phase10Config | None = None) -> Path:
    config = config or Phase10Config()
    train_end, validation_end, total_rows = config.split_rows()
    frame = load_development_prefix(dataset.resolve(), config)
    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    experiment_id = experiment_id or datetime.now(timezone.utc).strftime("phase10_%Y%m%dT%H%M%SZ")
    output = artifact_root.resolve() / experiment_id
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)

    tf, device = configure_tensorflow(config.random_seed)
    scaler = RobustScaler().fit(train[["Consumption"]])
    scaled_development = scaler.transform(frame[["Consumption"]]).reshape(-1)
    x_train, y_train = make_one_step_windows(scaled_development[:train_end], config.context_length)
    validation_context = scaled_development[train_end - config.context_length:validation_end]
    x_validation, y_validation = make_one_step_windows(validation_context, config.context_length)
    model = build_patchtst(config, tf)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config.patience, restore_best_weights=True,
    )
    history = model.fit(
        x_train, y_train, validation_data=(x_validation, y_validation), epochs=config.epochs,
        batch_size=config.batch_size, shuffle=False, verbose=0, callbacks=[callback],
    )
    predicted_scaled = model.predict(x_validation, batch_size=config.batch_size, verbose=0).reshape(-1)
    forecast = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).reshape(-1)
    actual = validation["Consumption"].to_numpy(dtype=float)
    naive = frame["Consumption"].iloc[train_end - 1:validation_end - 1].to_numpy(dtype=float)
    predictions = pd.DataFrame({"Date": validation["Date"].to_numpy(), "actual": actual,
                                "patchtst": forecast, "naive_lag1": naive})
    metrics = pd.DataFrame([
        {"model": "patchtst", **regression_metrics(actual, forecast)},
        {"model": "naive_lag1", **regression_metrics(actual, naive)},
    ]).sort_values(["mae", "model"]).reset_index(drop=True)
    predictions.to_csv(output / "predictions.csv", index=False, date_format="%Y-%m-%d")
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(history.history).rename_axis("epoch_zero_based").to_csv(output / "training_history.csv")
    model.save(output / "patchtst.keras")
    joblib.dump(scaler, output / "robust_scaler.joblib")
    _save_plots(predictions, output)
    errors = predictions.assign(residual=actual - forecast, absolute_error=np.abs(actual - forecast))
    errors.sort_values("absolute_error", ascending=False).head(20).to_csv(
        output / "highest_error_dates.csv", index=False, date_format="%Y-%m-%d")
    patch_mae = float(metrics.loc[metrics["model"] == "patchtst", "mae"].iloc[0])
    naive_mae = float(metrics.loc[metrics["model"] == "naive_lag1", "mae"].iloc[0])
    report = {
        "phase": "Phase 10", "candidate": "PatchTST", "candidate_count": 1,
        "train_rows": train_end, "validation_rows": validation_end - train_end,
        "locked_test_rows": total_rows - validation_end,
        "train_end_date_inclusive": train["Date"].iloc[-1].date().isoformat(),
        "validation_start_date": validation["Date"].iloc[0].date().isoformat(),
        "validation_end_date_inclusive": validation["Date"].iloc[-1].date().isoformat(),
        "locked_test_start_row_zero_based": validation_end,
        "locked_test_loaded": False, "locked_test_used_for_any_purpose": False,
        "scaler_fit_rows": train_end, "shuffle_used": False,
        "evaluation_protocol": "causal rolling one-step validation; each window uses only observations earlier than its target",
        "selection_or_hyperparameter_tuning_performed": False,
        "early_stopping_scope": "validation only; locked test remains unread",
        "simple_baseline": "naive_lag1", "improvement_claimed": patch_mae < naive_mae,
        "comparison_statement": ("PatchTST had lower validation MAE than naive lag-1."
                                 if patch_mae < naive_mae else
                                 "PatchTST did not beat naive lag-1 on validation MAE."),
        "epochs_completed": len(history.history["loss"]),
        "best_validation_loss_scaled": float(min(history.history["val_loss"])), "device": device,
        "development_prefix_sha256": hashlib.sha256(
            frame.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
        ).hexdigest(),
    }
    _write_json(output / "forecast_report.json", report)
    _write_json(output / "config.json", {
        "phase": "Phase 10", "experiment_id": experiment_id, "configuration": asdict(config),
        "random_seed": config.random_seed, "dataset": str(dataset.resolve()),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "tensorflow": tf.__version__},
        "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    (output / "execution.log").write_text(
        f"Phase 10 completed on {device}; seed={config.random_seed}; epochs={len(history.history['loss'])}.\n"
        f"Read rows [0,{validation_end}); locked rows [{validation_end},{total_rows}) were not loaded. "
        f"{report['comparison_statement']} No model selection or hyperparameter search was performed.\n",
        encoding="utf-8",
    )
    hashes = {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*"))
              if p.is_file() and p.name != "artifact_hashes.json"}
    _write_json(output / "artifact_hashes.json", hashes)
    return output


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-id")
    args = parser.parse_args(argv)
    print(run_phase10(args.dataset, args.artifact_root, args.experiment_id))


if __name__ == "__main__":
    main()
