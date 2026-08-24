#!/usr/bin/env python3
"""End-to-end water-demand forecasting pipeline.

Implements a reproducible forecasting workflow for the Siemens/KIT hourly water-demand dataset.
The workflow performs deterministic data preparation, model development, tuning, evaluation, and reporting.
then uses Codex once at the end as a read-only audit.

Command-line usage:
    python3 /mnt/c/Users/ASUS/Downloads/deadline_water_autopilot.py \
      . /mnt/c/Users/ASUS/Downloads/11045013.zip --push

Workflow:
    DATA PREPARATION -> BASELINE MODELING -> ENSEMBLE MODELING -> HYPERPARAMETER TUNING -> FINAL EVALUATION -> REPORTING

Methodological rules:
- chronological split only
- no future target leakage
- train-only preprocessing/tuning
- OHE for nominal, ordinal only for truly ordered feature, binary 0/1
- Yeo-Johnson only in scale-sensitive numerical pipelines
- 5-fold TimeSeriesSplit reporting
- Optuna on shortlisted ML model(s)
- Keras Tuner on DL model
- locked test used only after choices are frozen
- all metrics/plots/README/notebook are generated from actual executed work
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning)

REQUIRED_FILES = {
    "waterconsumption_rawdata.csv",
    "waterconsumption_noNaN.csv",
    "waterconsumption_noNaN_fixedTimechange.csv",
}

SEED = 42
np = None
pd = None


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None, check: bool = True):
    log("+ " + " ".join(map(str, cmd)))
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=False, check=False, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def maybe_reexec_project_venv(repo: Path) -> None:
    """Reuse the environment created by the earlier bootstrap instead of reinstalling packages."""
    if os.environ.get("WATER_FORECAST_VENV_REEXEC") == "1":
        return
    candidates = [repo / ".venv" / "bin" / "python", repo / ".venv-fast" / "bin" / "python"]
    for py in candidates:
        if py.exists():
            try:
                same = py.resolve() == Path(sys.executable).resolve()
            except Exception:
                same = False
            if not same:
                env = os.environ.copy()
                env["WATER_FORECAST_VENV_REEXEC"] = "1"
                log(f"Re-executing inside project environment: {py}")
                os.execve(str(py), [str(py), str(Path(__file__).resolve()), *sys.argv[1:]], env)
            return

def ensure_imports(repo: Path) -> None:
    """Install only missing lightweight packages; TF/Keras Tuner are expected from prior bootstrap."""
    global np, pd
    import importlib.util

    package_map = {
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "optuna": "optuna",
        "matplotlib": "matplotlib",
        "nbformat": "nbformat",
        "nbclient": "nbclient",
        "tensorflow": "tensorflow",
        "keras_tuner": "keras-tuner",
    }
    missing = [pip_name for mod, pip_name in package_map.items() if importlib.util.find_spec(mod) is None]
    if missing:
        log(f"Installing missing packages only: {missing}")
        run([sys.executable, "-m", "pip", "install", *missing], cwd=repo)

    import numpy as _np
    import pandas as _pd
    np, pd = _np, _pd


def ensure_dataset(repo: Path, archive: Path) -> tuple[Path, Path]:
    source = repo / "data" / "siemens_hourly" / "source"
    source.mkdir(parents=True, exist_ok=True)
    canonical = source / "waterconsumption_noNaN_fixedTimechange.csv"
    raw = source / "waterconsumption_rawdata.csv"
    if not canonical.exists() or not raw.exists():
        if not archive.exists():
            raise FileNotFoundError(f"Dataset archive not found: {archive}")
        with zipfile.ZipFile(archive) as zf:
            by_name = {Path(n).name: n for n in zf.namelist()}
            missing = REQUIRED_FILES - set(by_name)
            if missing:
                raise RuntimeError(f"Dataset archive missing {sorted(missing)}")
            for name in REQUIRED_FILES:
                with zf.open(by_name[name]) as src, (source / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    prov = {
        "source_record": "Zenodo 11045013",
        "doi": "10.5281/zenodo.11045013",
        "synthetic_data": False,
        "canonical_modeling_file": canonical.name,
        "canonical_sha256": sha256(canonical),
        "raw_audit_file": raw.name,
        "raw_sha256": sha256(raw),
    }
    (repo / "data" / "siemens_hourly" / "dataset_metadata.json").write_text(
        json.dumps(prov, indent=2) + "\n", encoding="utf-8"
    )
    return canonical, raw


def parse_timestamp(frame):
    start_hour = frame["Time"].astype(str).str.strip().str.extract(r"^(\d{1,2}):")[0].astype(int)
    dt = pd.to_datetime(frame["Date"], errors="raise") + pd.to_timedelta(start_hour, unit="h")
    return dt


def season_from_month(month):
    return np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["winter", "spring", "summer"],
        default="autumn",
    )


def hour_period(hour):
    return np.select(
        [hour.between(0, 5), hour.between(6, 11), hour.between(12, 17)],
        ["night", "morning", "afternoon"],
        default="evening",
    )


def rain_band(r):
    # Ordered auxiliary feature; numeric rain is also retained via lagged continuous feature.
    return pd.cut(
        r,
        bins=[-np.inf, 0.0, 0.5, 2.0, np.inf],
        labels=["none", "light", "moderate", "heavy"],
        ordered=True,
    )


def engineer(canonical: Path, raw: Path, smoke_rows: int | None = None):
    model = pd.read_csv(canonical)
    raw_df = pd.read_csv(raw)
    if smoke_rows:
        model = model.tail(smoke_rows).copy()
    model["timestamp"] = parse_timestamp(model)
    model = model.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    audit = {
        "raw_rows": int(len(raw_df)),
        "raw_missing": {k: int(v) for k, v in raw_df.isna().sum().items()},
        "canonical_rows": int(len(model)),
        "canonical_missing": {k: int(v) for k, v in model.isna().sum().items()},
        "timestamp_duplicates_after_parse": int(model["timestamp"].duplicated().sum()),
        "start": model["timestamp"].min().isoformat(),
        "end": model["timestamp"].max().isoformat(),
    }

    # Known-at-forecast-time calendar features.
    ts = model["timestamp"]
    model["hour"] = ts.dt.hour
    model["dow"] = ts.dt.dayofweek
    model["day_of_week_name"] = ts.dt.day_name()
    model["month"] = ts.dt.month
    model["day_of_year"] = ts.dt.dayofyear
    model["is_weekend"] = (model["dow"] >= 5).astype(int)
    model["season"] = season_from_month(model["month"])
    model["hour_period"] = hour_period(model["hour"])

    # Cyclical encodings: preserve closeness of 23:00 and 00:00 etc.
    model["hour_sin"] = np.sin(2 * np.pi * model["hour"] / 24)
    model["hour_cos"] = np.cos(2 * np.pi * model["hour"] / 24)
    model["dow_sin"] = np.sin(2 * np.pi * model["dow"] / 7)
    model["dow_cos"] = np.cos(2 * np.pi * model["dow"] / 7)
    model["month_sin"] = np.sin(2 * np.pi * (model["month"] - 1) / 12)
    model["month_cos"] = np.cos(2 * np.pi * (model["month"] - 1) / 12)

    # Demand history: every predictor references observations strictly earlier than t.
    y = model["Consumption"].astype(float)
    shifted = y.shift(1)
    for lag in (1, 2, 3, 24, 48, 72, 168):
        model[f"consumption_lag_{lag}"] = y.shift(lag)
    for w in (3, 24, 168):
        roll = shifted.rolling(w, min_periods=w)
        model[f"consumption_roll_mean_{w}"] = roll.mean()
        model[f"consumption_roll_std_{w}"] = roll.std()
        model[f"consumption_roll_min_{w}"] = roll.min()
        model[f"consumption_roll_max_{w}"] = roll.max()
    model["consumption_ewm_6"] = shifted.ewm(span=6, adjust=False).mean()
    model["consumption_ewm_24"] = shifted.ewm(span=24, adjust=False).mean()
    model["consumption_prev_change"] = y.shift(1) - y.shift(2)

    # Weather: use lagged measurements, not target-hour observed weather.
    for col in ("Temperature", "Rain", "Sun"):
        model[f"{col.lower()}_lag_1"] = model[col].shift(1)
        model[f"{col.lower()}_lag_24"] = model[col].shift(24)
    model["rain_roll_mean_24"] = model["Rain"].shift(1).rolling(24, min_periods=24).mean()
    model["sun_roll_mean_24"] = model["Sun"].shift(1).rolling(24, min_periods=24).mean()
    model["temperature_roll_mean_24"] = model["Temperature"].shift(1).rolling(24, min_periods=24).mean()
    model["rain_intensity"] = rain_band(model["rain_lag_1"])

    # Warm-up rows with unavailable target history are removed. Weather NaNs remain for train-only imputation.
    required_target_history = ["consumption_lag_168", "consumption_roll_mean_168", "consumption_roll_std_168"]
    model = model.dropna(subset=["Consumption", *required_target_history]).reset_index(drop=True)
    return model, audit


NOMINAL = ["day_of_week_name", "season", "hour_period"]
ORDINAL = ["rain_intensity"]
BINARY = ["is_weekend"]
CYCLICAL = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
CONTINUOUS = [
    "consumption_lag_1", "consumption_lag_2", "consumption_lag_3",
    "consumption_lag_24", "consumption_lag_48", "consumption_lag_72", "consumption_lag_168",
    "consumption_roll_mean_3", "consumption_roll_std_3", "consumption_roll_min_3", "consumption_roll_max_3",
    "consumption_roll_mean_24", "consumption_roll_std_24", "consumption_roll_min_24", "consumption_roll_max_24",
    "consumption_roll_mean_168", "consumption_roll_std_168", "consumption_roll_min_168", "consumption_roll_max_168",
    "consumption_ewm_6", "consumption_ewm_24", "consumption_prev_change",
    "temperature_lag_1", "temperature_lag_24", "temperature_roll_mean_24",
    "rain_lag_1", "rain_lag_24", "rain_roll_mean_24",
    "sun_lag_1", "sun_lag_24", "sun_roll_mean_24",
]
FEATURES = CONTINUOUS + CYCLICAL + BINARY + NOMINAL + ORDINAL


@dataclass
class SplitInfo:
    rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


def split_frame(frame):
    n = len(frame)
    i1 = int(n * 0.70)
    i2 = int(n * 0.85)
    train = frame.iloc[:i1].copy()
    val = frame.iloc[i1:i2].copy()
    test = frame.iloc[i2:].copy()
    info = SplitInfo(
        rows=n, train_rows=len(train), validation_rows=len(val), test_rows=len(test),
        train_end=train["timestamp"].iloc[-1].isoformat(),
        validation_start=val["timestamp"].iloc[0].isoformat(),
        validation_end=val["timestamp"].iloc[-1].isoformat(),
        test_start=test["timestamp"].iloc[0].isoformat(),
        test_end=test["timestamp"].iloc[-1].isoformat(),
    )
    return train, val, test, info


def regression_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(np.where(denom == 0, 0.0, 200.0 * np.abs(y_true - y_pred) / denom)))
    return {"mae": mae, "rmse": rmse, "r2": r2, "smape_pct": smape}


def make_preprocessors():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, RobustScaler

    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ordinal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=[["none", "light", "moderate", "heavy"]],
            handle_unknown="use_encoded_value", unknown_value=-1,
        )),
    ])
    # Yeo-Johnson belongs to scale-sensitive continuous predictors only.
    yj_cont = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("robust", RobustScaler()),
    ])
    bounded = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("robust", RobustScaler()),
    ])
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])

    scale_sensitive = ColumnTransformer([
        ("continuous_yj", yj_cont, CONTINUOUS),
        ("cyclical", bounded, CYCLICAL),
        ("binary", binary, BINARY),
        ("nominal", nominal, NOMINAL),
        ("ordinal", ordinal, ORDINAL),
    ], remainder="drop", sparse_threshold=0.0)

    # Tree branch deliberately omits Yeo-Johnson/scaling: not required for tree split ordering.
    numeric_tree = Pipeline([("impute", SimpleImputer(strategy="median"))])
    tree = ColumnTransformer([
        ("continuous", numeric_tree, CONTINUOUS + CYCLICAL),
        ("binary", binary, BINARY),
        ("nominal", nominal, NOMINAL),
        ("ordinal", ordinal, ORDINAL),
    ], remainder="drop", sparse_threshold=0.0)
    return scale_sensitive, tree


def save_plot_actual_pred(path: Path, ts, actual, pred, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.5))
    n = min(len(actual), 24 * 14)  # two weeks for readability
    ax.plot(ts.iloc[-n:], np.asarray(actual)[-n:], label="Actual", linewidth=1.5)
    ax.plot(ts.iloc[-n:], np.asarray(pred)[-n:], label="Predicted", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Water consumption")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def baseline_stage(repo: Path, train, val):
    from sklearn.base import clone
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline

    out = repo / "artifacts" / "deadline" / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    preds = pd.DataFrame({"timestamp": val["timestamp"], "actual": val["Consumption"].to_numpy()})
    for lag in (1, 24, 168):
        p = val[f"consumption_lag_{lag}"].to_numpy(dtype=float)
        m = regression_metrics(val["Consumption"], p)
        rows.append({"model": f"naive_lag_{lag}", **m})
        preds[f"naive_lag_{lag}"] = p

    scale_pre, tree_pre = make_preprocessors()
    ridge = Pipeline([("pre", clone(scale_pre)), ("model", Ridge(alpha=2.0))])
    ridge.fit(train[FEATURES], train["Consumption"])
    p = ridge.predict(val[FEATURES])
    rows.append({"model": "ridge_yj_robust", **regression_metrics(val["Consumption"], p)})
    preds["ridge_yj_robust"] = p

    rf = Pipeline([
        ("pre", clone(tree_pre)),
        ("model", RandomForestRegressor(
            n_estimators=100, max_depth=18, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1, max_features=0.8,
        )),
    ])
    rf.fit(train[FEATURES], train["Consumption"])
    p = rf.predict(val[FEATURES])
    rows.append({"model": "random_forest_baseline", **regression_metrics(val["Consumption"], p)})
    preds["random_forest_baseline"] = p

    metrics = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    preds.to_csv(out / "predictions.csv", index=False)
    best = metrics.iloc[0]["model"]
    save_plot_actual_pred(out / "best_validation_forecast.png", val["timestamp"], val["Consumption"], preds[best], f"Baseline validation: {best}")
    return metrics, preds


def xgb_model(params: dict[str, Any] | None = None, use_gpu: bool = True):
    from xgboost import XGBRegressor
    base = dict(
        objective="reg:squarederror",
        n_estimators=350,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=SEED,
        tree_method="hist",
        n_jobs=4,
        verbosity=0,
    )
    if use_gpu:
        base["device"] = "cuda"
    if params:
        base.update(params)
    return XGBRegressor(**base)


def detect_xgb_gpu(X_small, y_small) -> bool:
    try:
        m = xgb_model({"n_estimators": 5, "max_depth": 3}, use_gpu=True)
        m.fit(X_small, y_small)
        return True
    except Exception as exc:
        log(f"XGBoost CUDA probe failed; CPU fallback will be recorded: {exc}")
        return False


def time_series_cv_models(trainval, model_specs, n_splits: int = 5):
    from sklearn.base import clone
    from sklearn.model_selection import TimeSeriesSplit
    rows = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for model_name, preprocessor, estimator in model_specs:
        fold_mae = []
        fold_rmse = []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(trainval), 1):
            tr = trainval.iloc[tr_idx]
            va = trainval.iloc[va_idx]
            pre = clone(preprocessor)
            Xtr = pre.fit_transform(tr[FEATURES], tr["Consumption"])
            Xva = pre.transform(va[FEATURES])
            est = clone(estimator)
            est.fit(Xtr, tr["Consumption"])
            pred = est.predict(Xva)
            met = regression_metrics(va["Consumption"], pred)
            fold_mae.append(met["mae"])
            fold_rmse.append(met["rmse"])
            rows.append({"model": model_name, "fold": fold, **met})
        log(f"CV {model_name}: MAE={np.mean(fold_mae):.3f} ± {np.std(fold_mae):.3f}")
    detailed = pd.DataFrame(rows)
    summary = detailed.groupby("model", as_index=False).agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        r2_mean=("r2", "mean"), smape_mean=("smape_pct", "mean"),
    ).sort_values("mae_mean")
    return detailed, summary


def better_stage(repo: Path, train, val, xgb_gpu: bool, smoke: bool = False):
    from sklearn.base import clone
    from sklearn.ensemble import BaggingRegressor, HistGradientBoostingRegressor, VotingRegressor
    from sklearn.tree import ExtraTreeRegressor

    out = repo / "artifacts" / "deadline" / "better"
    out.mkdir(parents=True, exist_ok=True)
    _, tree_pre = make_preprocessors()
    pre = clone(tree_pre)
    Xtr = pre.fit_transform(train[FEATURES], train["Consumption"])
    Xv = pre.transform(val[FEATURES])
    ytr = train["Consumption"].to_numpy()
    yv = val["Consumption"].to_numpy()

    bag = BaggingRegressor(
        estimator=ExtraTreeRegressor(max_depth=20, min_samples_leaf=2, random_state=SEED),
        n_estimators=40 if smoke else 70, max_samples=0.85, max_features=0.9,
        bootstrap=True, random_state=SEED, n_jobs=-1,
    )
    hgb = HistGradientBoostingRegressor(
        learning_rate=0.07, max_iter=80 if smoke else 180, max_leaf_nodes=31,
        l2_regularization=0.5, min_samples_leaf=20, random_state=SEED,
    )
    xgb = xgb_model({"n_estimators": 80 if smoke else 350}, use_gpu=xgb_gpu)

    models = {"bagging_extra_tree": bag, "hist_gradient_boosting": hgb, "xgboost": xgb}
    rows = []
    preds = pd.DataFrame({"timestamp": val["timestamp"], "actual": yv})
    fitted = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        p = model.predict(Xv)
        fitted[name] = model
        preds[name] = p
        rows.append({"model": name, "fit_seconds": time.time() - t0, **regression_metrics(yv, p)})
        log(f"BETTER {name}: MAE={rows[-1]['mae']:.3f}")

    voting = VotingRegressor(
        estimators=[("bag", clone(bag)), ("hgb", clone(hgb)), ("xgb", clone(xgb))],
        weights=[1.0, 1.0, 2.0], n_jobs=None,
    )
    t0 = time.time()
    voting.fit(Xtr, ytr)
    p = voting.predict(Xv)
    fitted["voting_regressor"] = voting
    preds["voting_regressor"] = p
    rows.append({"model": "voting_regressor", "fit_seconds": time.time() - t0, **regression_metrics(yv, p)})

    metrics = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    preds.to_csv(out / "predictions.csv", index=False)
    (out / "xgboost_device.json").write_text(json.dumps({"gpu_used": bool(xgb_gpu), "device": "cuda" if xgb_gpu else "cpu"}, indent=2) + "\n")

    # Genuine 5-fold time-aware CV on representative ensemble families. Keep Voting out of CV to avoid duplicate triple fitting.
    trainval = pd.concat([train, val], ignore_index=True)
    model_specs = [
        ("bagging_extra_tree", tree_pre, clone(bag)),
        ("xgboost", tree_pre, clone(xgb)),
    ]
    if not smoke:
        model_specs.insert(1, ("hist_gradient_boosting", tree_pre, clone(hgb)))
    detailed, summary = time_series_cv_models(trainval, model_specs, n_splits=3 if smoke else 5)
    detailed.to_csv(out / "cv_folds.csv", index=False)
    summary.to_csv(out / "cv_summary.csv", index=False)

    best = metrics.iloc[0]["model"]
    save_plot_actual_pred(out / "best_validation_forecast.png", val["timestamp"], yv, preds[best], f"Better validation: {best}")
    return metrics, preds, pre, fitted, summary


def optuna_xgb(repo: Path, train, val, xgb_gpu: bool, smoke: bool = False):
    import optuna
    from sklearn.base import clone
    from sklearn.model_selection import TimeSeriesSplit

    out = repo / "artifacts" / "deadline" / "optimal"
    out.mkdir(parents=True, exist_ok=True)
    _, tree_pre = make_preprocessors()
    dev = pd.concat([train, val], ignore_index=True)
    tscv = TimeSeriesSplit(n_splits=2 if smoke else 3)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 180, 500, step=80),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        maes = []
        for tr_idx, va_idx in tscv.split(dev):
            tr = dev.iloc[tr_idx]
            va = dev.iloc[va_idx]
            pre = clone(tree_pre)
            Xtr = pre.fit_transform(tr[FEATURES], tr["Consumption"])
            Xva = pre.transform(va[FEATURES])
            model = xgb_model(params, use_gpu=xgb_gpu)
            model.fit(Xtr, tr["Consumption"])
            pred = model.predict(Xva)
            maes.append(regression_metrics(va["Consumption"], pred)["mae"])
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=2 if smoke else 8, timeout=120 if smoke else 12 * 60, show_progress_bar=False)
    payload = {"best_value_cv_mae": study.best_value, "best_params": study.best_params, "trials_completed": len(study.trials)}
    (out / "optuna_xgb.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([
        {"number": t.number, "value": t.value, "state": str(t.state), **t.params}
        for t in study.trials
    ]).to_csv(out / "optuna_trials.csv", index=False)

    # Validation score with frozen tuned parameters; still no test access.
    pre = clone(tree_pre)
    Xtr = pre.fit_transform(train[FEATURES], train["Consumption"])
    Xv = pre.transform(val[FEATURES])
    model = xgb_model(study.best_params, use_gpu=xgb_gpu)
    model.fit(Xtr, train["Consumption"])
    pred = model.predict(Xv)
    met = regression_metrics(val["Consumption"], pred)
    (out / "tuned_xgb_validation.json").write_text(json.dumps(met, indent=2) + "\n")
    log(f"OPTIMAL tuned XGBoost validation MAE={met['mae']:.3f}")
    return study.best_params, met


def tensorflow_gpu_report(repo: Path):
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        with contextlib.suppress(Exception):
            tf.config.experimental.set_memory_growth(gpu, True)
    report = {"tensorflow_version": tf.__version__, "gpu_devices": [g.name for g in gpus], "gpu_used": bool(gpus)}
    d = repo / "artifacts" / "deadline" / "environment"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tensorflow_gpu.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def keras_tune_mlp(repo: Path, train, val, smoke: bool = False):
    import tensorflow as tf
    import keras_tuner as kt
    from sklearn.base import clone

    out = repo / "artifacts" / "deadline" / "optimal"
    out.mkdir(parents=True, exist_ok=True)
    scale_pre, _ = make_preprocessors()
    pre = clone(scale_pre)
    Xtr = np.asarray(pre.fit_transform(train[FEATURES], train["Consumption"]), dtype=np.float32)
    Xv = np.asarray(pre.transform(val[FEATURES]), dtype=np.float32)
    ytr = train["Consumption"].to_numpy(dtype=np.float32)
    yv = val["Consumption"].to_numpy(dtype=np.float32)

    # Normalize target using training statistics only for neural optimization stability.
    y_mean = float(ytr.mean())
    y_std = float(ytr.std() or 1.0)
    ytr_n = (ytr - y_mean) / y_std
    yv_n = (yv - y_mean) / y_std

    def build_model(hp):
        inputs = tf.keras.Input(shape=(Xtr.shape[1],))
        x = inputs
        layers = hp.Int("hidden_layers", 1, 3)
        for i in range(layers):
            units = hp.Choice(f"units_{i}", [64, 128, 256])
            x = tf.keras.layers.Dense(units, activation="relu", kernel_initializer="he_normal")(x)
            if hp.Boolean(f"batchnorm_{i}"):
                x = tf.keras.layers.BatchNormalization()(x)
            dropout = hp.Choice(f"dropout_{i}", [0.0, 0.1, 0.2, 0.3])
            if dropout:
                x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)
        model = tf.keras.Model(inputs, outputs)
        lr = hp.Float("learning_rate", 1e-4, 3e-3, sampling="log")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
        return model

    tuner_dir = out / "keras_tuner"
    if tuner_dir.exists():
        shutil.rmtree(tuner_dir)
    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=1 if smoke else 4,
        overwrite=True,
        directory=str(tuner_dir),
        project_name="mlp",
        seed=SEED,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    tuner.search(
        Xtr, ytr_n,
        validation_data=(Xv, yv_n),
        epochs=3 if smoke else 12,
        batch_size=512,
        callbacks=callbacks,
        verbose=0,
    )
    best_hp = tuner.get_best_hyperparameters(1)[0]
    model = tuner.hypermodel.build(best_hp)
    hist = model.fit(
        Xtr, ytr_n, validation_data=(Xv, yv_n),
        epochs=3 if smoke else 20,
        batch_size=512, callbacks=callbacks, verbose=0,
    )
    pred_n = model.predict(Xv, batch_size=1024, verbose=0).reshape(-1)
    pred = pred_n * y_std + y_mean
    met = regression_metrics(yv, pred)
    best_epoch = int(np.argmin(hist.history["val_loss"]) + 1)
    payload = {
        "best_hyperparameters": best_hp.values,
        "validation_metrics": met,
        "best_epoch": best_epoch,
        "target_train_mean": y_mean,
        "target_train_std": y_std,
    }
    (out / "keras_tuner_mlp.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    log(f"OPTIMAL tuned MLP validation MAE={met['mae']:.3f}; best_epoch={best_epoch}")
    return best_hp.values, met, best_epoch


def choose_final(validation_table):
    # All candidates compared on validation only; locked test remains untouched.
    return validation_table.sort_values("mae").iloc[0].to_dict()


def final_locked_test(repo: Path, train, val, test, xgb_gpu: bool, tuned_xgb_params, mlp_hp, mlp_epoch, selected_name: str):
    from sklearn.base import clone
    out = repo / "artifacts" / "deadline" / "final"
    out.mkdir(parents=True, exist_ok=True)
    dev = pd.concat([train, val], ignore_index=True)
    ytest = test["Consumption"].to_numpy(dtype=float)
    rows = []
    preds = pd.DataFrame({"timestamp": test["timestamp"], "actual": ytest})

    # Predetermined persistence baselines are always allowed for honest context.
    for lag in (1, 24, 168):
        p = test[f"consumption_lag_{lag}"].to_numpy(dtype=float)
        rows.append({"model": f"naive_lag_{lag}", "role": "baseline", **regression_metrics(ytest, p)})
        preds[f"naive_lag_{lag}"] = p

    if selected_name.startswith("naive_lag_"):
        lag = int(selected_name.rsplit("_", 1)[1])
        p = test[f"consumption_lag_{lag}"].to_numpy(dtype=float)
    elif selected_name == "tuned_mlp":
        import tensorflow as tf
        scale_pre, _ = make_preprocessors()
        pre = clone(scale_pre)
        Xdev = np.asarray(pre.fit_transform(dev[FEATURES], dev["Consumption"]), dtype=np.float32)
        Xt = np.asarray(pre.transform(test[FEATURES]), dtype=np.float32)
        ydev = dev["Consumption"].to_numpy(dtype=np.float32)
        mean, std = float(ydev.mean()), float(ydev.std() or 1.0)
        ydevn = (ydev - mean) / std
        inputs = tf.keras.Input(shape=(Xdev.shape[1],))
        x = inputs
        for i in range(int(mlp_hp.get("hidden_layers", 1))):
            units = int(mlp_hp.get(f"units_{i}", 128))
            x = tf.keras.layers.Dense(units, activation="relu", kernel_initializer="he_normal")(x)
            if bool(mlp_hp.get(f"batchnorm_{i}", False)):
                x = tf.keras.layers.BatchNormalization()(x)
            d = float(mlp_hp.get(f"dropout_{i}", 0.0))
            if d:
                x = tf.keras.layers.Dropout(d)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(float(mlp_hp.get("learning_rate", 1e-3))), loss="mse")
        model.fit(Xdev, ydevn, epochs=max(1, mlp_epoch), batch_size=512, verbose=0)
        p = model.predict(Xt, batch_size=1024, verbose=0).reshape(-1) * std + mean
    elif selected_name == "ridge_yj_robust":
        from sklearn.linear_model import Ridge
        scale_pre, _ = make_preprocessors()
        pre = clone(scale_pre)
        Xdev = pre.fit_transform(dev[FEATURES], dev["Consumption"])
        Xt = pre.transform(test[FEATURES])
        model = Ridge(alpha=2.0)
        model.fit(Xdev, dev["Consumption"])
        p = model.predict(Xt)
    else:
        from sklearn.ensemble import BaggingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
        from sklearn.tree import ExtraTreeRegressor
        _, tree_pre = make_preprocessors()
        pre = clone(tree_pre)
        Xdev = pre.fit_transform(dev[FEATURES], dev["Consumption"])
        Xt = pre.transform(test[FEATURES])
        if selected_name == "random_forest_baseline":
            model = RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_leaf=2, random_state=SEED, n_jobs=-1, max_features=0.8)
        elif selected_name == "bagging_extra_tree":
            model = BaggingRegressor(estimator=ExtraTreeRegressor(max_depth=20, min_samples_leaf=2, random_state=SEED), n_estimators=70, max_samples=0.85, max_features=0.9, bootstrap=True, random_state=SEED, n_jobs=-1)
        elif selected_name == "hist_gradient_boosting":
            model = HistGradientBoostingRegressor(learning_rate=0.07, max_iter=180, max_leaf_nodes=31, l2_regularization=0.5, min_samples_leaf=20, random_state=SEED)
        elif selected_name == "xgboost":
            model = xgb_model({"n_estimators": 350}, use_gpu=xgb_gpu)
        elif selected_name == "tuned_xgboost":
            model = xgb_model(tuned_xgb_params, use_gpu=xgb_gpu)
        elif selected_name == "voting_regressor":
            bag = BaggingRegressor(estimator=ExtraTreeRegressor(max_depth=20, min_samples_leaf=2, random_state=SEED), n_estimators=70, max_samples=0.85, max_features=0.9, bootstrap=True, random_state=SEED, n_jobs=-1)
            hgb = HistGradientBoostingRegressor(learning_rate=0.07, max_iter=180, max_leaf_nodes=31, l2_regularization=0.5, min_samples_leaf=20, random_state=SEED)
            xgb = xgb_model({"n_estimators": 350}, use_gpu=xgb_gpu)
            model = VotingRegressor(estimators=[("bag", bag), ("hgb", hgb), ("xgb", xgb)], weights=[1.0, 1.0, 2.0])
        else:
            raise ValueError(f"Unsupported selected model: {selected_name}")
        model.fit(Xdev, dev["Consumption"])
        p = model.predict(Xt)

    preds[selected_name] = p
    met = regression_metrics(ytest, p)
    rows.append({"model": selected_name, "role": "selected_final_model", **met})
    metrics = pd.DataFrame(rows).sort_values(["role", "mae"])
    metrics.to_csv(out / "locked_test_metrics.csv", index=False)
    preds.to_csv(out / "locked_test_predictions.csv", index=False)
    (out / "selection.json").write_text(json.dumps({
        "selected_before_test": selected_name,
        "locked_test_used_for_selection": False,
        "locked_test_metrics": met,
    }, indent=2) + "\n")
    save_plot_actual_pred(out / "locked_test_forecast.png", test["timestamp"], ytest, p, f"Locked test: {selected_name}")
    return metrics, preds, met


def generate_summary(repo: Path, audit: dict, split_info: SplitInfo, baseline_metrics, better_metrics, cv_summary, tuned_xgb_met, mlp_met, selected, final_metrics, gpu_report, elapsed):
    out = repo / "artifacts" / "deadline"
    val_candidates = pd.DataFrame([
        {"model": "tuned_xgboost", **tuned_xgb_met},
        {"model": "tuned_mlp", **mlp_met},
    ])
    summary = {
        "dataset": {
            "source": "Hourly Water Demand of a Mixed District Recorded by Supplier",
            "zenodo_record": "11045013",
            "doi": "10.5281/zenodo.11045013",
            "synthetic": False,
            "audit": audit,
        },
        "split": asdict(split_info),
        "feature_policy": {
            "nominal_encoding": "OneHotEncoder",
            "ordinal_encoding": "OrdinalEncoder only for rain_intensity: none < light < moderate < heavy",
            "binary_encoding": "0/1",
            "scale_sensitive_numeric": "MedianImputer -> YeoJohnson -> RobustScaler",
            "trees": "No unnecessary scaling/Yeo-Johnson requirement; train-only imputation + categorical encoding",
            "target_history": "all lags/rolls are past-only; rolling uses target.shift(1)",
        },
        "baseline_validation": baseline_metrics.to_dict(orient="records"),
        "better_validation": better_metrics.to_dict(orient="records"),
        "better_cv": cv_summary.to_dict(orient="records"),
        "optimal_validation": val_candidates.to_dict(orient="records"),
        "selected_before_locked_test": selected,
        "locked_test": final_metrics.to_dict(orient="records"),
        "gpu": gpu_report,
        "elapsed_minutes": elapsed / 60.0,
    }
    (out / "experiment_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def generate_readme(repo: Path, summary: dict):
    final = [r for r in summary["locked_test"] if r["role"] == "selected_final_model"][0]
    selected = summary["selected_before_locked_test"]["model"]
    cv = summary["better_cv"]
    best_cv = min(cv, key=lambda x: x["mae_mean"]) if cv else None
    old = repo / "README.md"
    backup = repo / "docs" / "README_before_siemens.md"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if old.exists() and not backup.exists():
        shutil.copy2(old, backup)
    text = f"""# Water Demand Forecasting — Siemens/KIT Hourly Dataset

A reproducible, leakage-safe water-demand forecasting project built on **real hourly operational data**, not synthetic data.

## Problem
Forecast one-hour-ahead aggregate water consumption for a mixed residential/commercial/industrial district so a water utility can support pump scheduling and capacity planning.

## Data
- Source: *Hourly Water Demand of a Mixed District Recorded by Supplier*
- Zenodo record: `11045013`
- DOI: `10.5281/zenodo.11045013`
- Canonical series: hourly data from 2016–2021 with Consumption, Temperature, Rain and Sun.
- Synthetic data used: **No**

## Methodology
`Data Preparation -> Baseline Modeling -> Ensemble Evaluation -> Hyperparameter Tuning -> Locked-Test Evaluation`

- Chronological 70/15/15 train/validation/test split.
- Test is locked until preprocessing, model family and hyperparameters are frozen.
- Demand lag features: 1/2/3/24/48/72/168 hours.
- Past-only rolling statistics: 3h, 24h and 168h.
- Weather features are lagged to avoid using target-hour observed weather as if it were a forecast.
- Nominal categories: OneHotEncoder.
- Ordered rain-intensity auxiliary feature: OrdinalEncoder.
- Binary indicators: 0/1.
- Scale-sensitive numeric branch: median imputation -> Yeo-Johnson -> RobustScaler.
- Tree/boosting branch does not claim scaling is necessary.
- Ensembles: Bagging, boosting/XGBoost and VotingRegressor.
- ML validation: TimeSeriesSplit cross-validation.
- ML tuning: Optuna on the shortlisted XGBoost family.
- DL tuning: Keras Tuner on a GPU-backed MLP with EarlyStopping.

## Final result
The model was selected **before** viewing the locked test based on validation performance.

- Selected model: **{selected}**
- Locked-test MAE: **{final['mae']:.3f}**
- Locked-test RMSE: **{final['rmse']:.3f}**
- Locked-test R²: **{final['r2']:.4f}**
- Locked-test sMAPE: **{final['smape_pct']:.3f}%**
"""
    if best_cv:
        text += f"- Best reported ensemble-family CV MAE: **{best_cv['mae_mean']:.3f} ± {best_cv['mae_std']:.3f}** ({best_cv['model']})\n"
    text += """

## Reproducibility
- `src/forecasting_pipeline.py` — actual experiment implementation.
- `artifacts/results/` — metrics, predictions, split metadata, GPU report and tuning history.
- `notebooks/model_evaluation_report.ipynb` — executed interview/Colab notebook.
- `artifacts/results/methodology_audit.txt` — final read-only methodology audit when Codex is available.

## Important limitation
This project predicts hourly demand from historical demand/calendar and lagged observed weather. In production, future weather should come from a weather-forecast feed. Results are experimental and are not presented as a deployed utility system.
"""
    old.write_text(text, encoding="utf-8")


def make_notebook(repo: Path, summary: dict):
    import nbformat as nbf
    from nbclient import NotebookClient

    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"}
    final = [r for r in summary["locked_test"] if r["role"] == "selected_final_model"][0]
    selected = summary["selected_before_locked_test"]["model"]
    nb.cells = [
        nbf.v4.new_markdown_cell(f"""# Water Demand Forecasting — Baseline → Better → Optimal

**Real dataset:** Siemens/KIT hourly water-demand data (Zenodo DOI `10.5281/zenodo.11045013`).  
**Selected before locked test:** `{selected}`  
**Locked-test MAE:** `{final['mae']:.3f}` | **RMSE:** `{final['rmse']:.3f}` | **R²:** `{final['r2']:.4f}`

This notebook is generated only after the actual experiments finish. No metric below is fabricated.
"""),
        nbf.v4.new_markdown_cell("""## Methodology rules

- Chronological train/validation/test split; no shuffled forecasting CV.
- OHE only for nominal features; ordinal encoding only for a genuinely ordered rain-intensity feature; binary indicators remain 0/1.
- Yeo-Johnson is applied inside the training-fitted scale-sensitive numerical pipeline, followed by RobustScaler.
- Tree/boosting models do not receive scaling merely for appearance.
- Target rolling features are computed after `shift(1)`.
- Optuna and Keras Tuner never see the locked test.
"""),
        nbf.v4.new_code_cell("""# Colab/local setup: clone the repository first if running outside it.
from pathlib import Path
import json, pandas as pd
ROOT = Path.cwd()
if not (ROOT / 'artifacts/results/experiment_summary.json').exists():
    print('Run this notebook from the repository root after cloning the project.')
summary = json.loads((ROOT / 'artifacts/results/experiment_summary.json').read_text())
summary['dataset']['source'], summary['split']
"""),
        nbf.v4.new_markdown_cell("## Dataset audit"),
        nbf.v4.new_code_cell("""pd.DataFrame({
    'raw_missing': summary['dataset']['audit']['raw_missing'],
    'canonical_missing': summary['dataset']['audit']['canonical_missing'],
}).fillna(0)
"""),
        nbf.v4.new_markdown_cell("## Feature engineering and preprocessing\nThe complete executable implementation is in `src/forecasting_pipeline.py`. The code below shows the actual feature policy used."),
        nbf.v4.new_code_cell("""summary['feature_policy']
"""),
        nbf.v4.new_markdown_cell("## Baseline validation"),
        nbf.v4.new_code_cell("""baseline = pd.read_csv(ROOT / 'artifacts/results/baseline_models/metrics.csv')
baseline
"""),
        nbf.v4.new_markdown_cell("## Better: Bagging + Boosting + Voting + time-aware CV"),
        nbf.v4.new_code_cell("""better = pd.read_csv(ROOT / 'artifacts/results/ensemble_models/metrics.csv')
cv = pd.read_csv(ROOT / 'artifacts/results/ensemble_models/cv_summary.csv')
display(better)
display(cv)
"""),
        nbf.v4.new_markdown_cell("## Optimal: Optuna + Keras Tuner"),
        nbf.v4.new_code_cell("""optuna_result = json.loads((ROOT / 'artifacts/results/hyperparameter_tuning/optuna_xgb.json').read_text())
keras_result = json.loads((ROOT / 'artifacts/results/hyperparameter_tuning/keras_tuner_mlp.json').read_text())
print('Optuna:', optuna_result)
print('Keras Tuner validation:', keras_result['validation_metrics'])
"""),
        nbf.v4.new_markdown_cell("## Locked test — viewed only after model choice was frozen"),
        nbf.v4.new_code_cell("""locked = pd.read_csv(ROOT / 'artifacts/results/final_evaluation/locked_test_metrics.csv')
locked
"""),
        nbf.v4.new_code_cell("""from IPython.display import Image, display
p = ROOT / 'artifacts/results/final_evaluation/locked_test_forecast.png'
display(Image(filename=str(p)))
"""),
        nbf.v4.new_markdown_cell("""## Interview explanation

**Problem:** one-hour-ahead operational water-demand forecasting.  
**Baseline:** persistence + simple ML.  
**Better:** leakage-safe engineered lags/calendar/weather with Bagging, boosting/XGBoost and Voting.  
**Optimal:** time-aware CV, Optuna for shortlisted ML, Keras Tuner for DL, then one locked-test evaluation.  

**Limitation:** target-hour weather was not treated as known. Only lagged observed weather was used; a production system would consume forecast weather.
"""),
    ]
    path = repo / "notebooks" / "Water_Demand_Forecasting_Final.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Execute lightweight reporting cells against real artifacts; expensive training is already complete.
    client = NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(repo)}})
    try:
        client.execute()
    except Exception as exc:
        # Preserve the notebook but clearly record execution failure rather than faking outputs.
        nb.cells.insert(1, nbf.v4.new_markdown_cell(f"**Local notebook execution warning:** `{type(exc).__name__}: {exc}`"))
    nbf.write(nb, path)
    return path


def copy_workflow_into_repo(repo: Path):
    target = repo / "fast_pipeline" / "deadline_water_workflow.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve()
    if src != target.resolve():
        shutil.copy2(src, target)
    init = target.parent / "__init__.py"
    init.touch(exist_ok=True)


def codex_audit(repo: Path) -> str:
    out = repo / "artifacts" / "deadline" / "methodology_audit.txt"
    codex = shutil.which("codex")
    if not codex:
        text = "AUDIT_NOT_RUN\nCodex CLI not found; deterministic gates completed."
        out.write_text(text + "\n")
        return text
    prompt = """
Read README.md, src/forecasting_pipeline.py, data/dataset_metadata.json,
and all files under artifacts/results that are necessary to verify methodology.

This is a READ-ONLY final audit. Verify ONLY these high-value rules:
1) real Siemens/KIT dataset, no synthetic-data claim;
2) chronological split and locked test not used for tuning/model choice;
3) target lags/rolls are past-only;
4) OHE for nominal, ordinal only for genuinely ordered rain intensity, binary 0/1;
5) Yeo-Johnson fitted training-only in scale-sensitive branch, not falsely claimed necessary for trees;
6) Bagging, boosting/XGBoost, Voting, TimeSeriesSplit CV, Optuna, Keras Tuner are evidenced by code/artifacts;
7) README metrics match actual artifact metrics and contain no invented business impact.

Return first line exactly PASS or FAIL. If FAIL, list concrete methodological errors only; do not fail for cosmetic style.
""".strip()
    with tempfile.TemporaryDirectory(prefix="water-codex-audit-") as td:
        tmp = Path(td) / "audit.txt"
        try:
            p = subprocess.run([
                codex, "exec", "-C", str(repo), "-s", "read-only",
                "--output-last-message", str(tmp), prompt,
            ], text=True, capture_output=True, timeout=300, check=False)
            if p.returncode == 0 and tmp.exists():
                text = tmp.read_text(encoding="utf-8").strip()
            else:
                text = f"AUDIT_ERROR\nreturncode={p.returncode}\n{p.stderr[-2000:]}"
        except subprocess.TimeoutExpired:
            text = "AUDIT_TIMEOUT\nCodex audit exceeded 300 seconds; deterministic validation artifacts remain available."
    out.write_text(text + "\n", encoding="utf-8")
    return text


def git_finalize(repo: Path, push: bool):
    if not (repo / ".git").exists():
        return
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=repo, text=True, capture_output=True).stdout.strip()
    if branch in {"main", "master", ""}:
        run(["git", "checkout", "-b", "feature/water-demand-forecasting"], cwd=repo)
    paths = [
        "README.md", "docs/README_before_siemens.md",
        "src/forecasting_pipeline.py",
        "data/dataset_metadata.json",
        "artifacts/results", "notebooks/model_evaluation_report.ipynb",
    ]
    existing = [p for p in paths if (repo / p).exists()]
    run(["git", "add", *existing], cwd=repo)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo)
    if diff.returncode != 0:
        run(["git", "commit", "-m", "feat: complete Siemens KIT water demand forecasting workflow"], cwd=repo)
    if push:
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=repo, text=True, capture_output=True).stdout.strip()
        run(["git", "push", "-u", "origin", branch], cwd=repo, check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", type=Path)
    ap.add_argument("dataset_zip", type=Path)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="quick code-path check on a smaller tail sample")
    ap.add_argument("--skip-codex-audit", action="store_true")
    args = ap.parse_args()
    repo = args.repo.expanduser().resolve()
    archive = args.dataset_zip.expanduser().resolve()
    maybe_reexec_project_venv(repo)
    start = time.time()
    log("\n=== WATER DEMAND FORECASTING PIPELINE ===")
    ensure_imports(repo)
    copy_workflow_into_repo(repo)
    canonical, raw = ensure_dataset(repo, archive)
    frame, audit = engineer(canonical, raw, smoke_rows=6000 if args.smoke else None)
    train, val, test, split_info = split_frame(frame)
    out = repo / "artifacts" / "deadline"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_quality_report.json").write_text(json.dumps(audit, indent=2) + "\n")
    (out / "data_split.json").write_text(json.dumps(asdict(split_info), indent=2) + "\n")
    (out / "feature_manifest.json").write_text(json.dumps({
        "features": FEATURES, "continuous_yj": CONTINUOUS, "cyclical": CYCLICAL,
        "binary": BINARY, "nominal_ohe": NOMINAL, "ordinal": ORDINAL,
    }, indent=2) + "\n")

    log(f"Rows after causal feature warm-up: {len(frame):,}")
    log(f"Train/val/test: {len(train):,}/{len(val):,}/{len(test):,}")

    log("\n=== 1/4 BASELINE MODELING ===")
    baseline_metrics, _ = baseline_stage(repo, train, val)
    log(baseline_metrics.to_string(index=False))

    # Pre-fit small tree encoding for CUDA capability probe.
    _, tree_pre = make_preprocessors()
    probe_pre = tree_pre.fit(train[FEATURES].iloc[: min(2000, len(train))], train["Consumption"].iloc[: min(2000, len(train))])
    Xprobe = probe_pre.transform(train[FEATURES].iloc[: min(2000, len(train))])
    yprobe = train["Consumption"].iloc[: min(2000, len(train))]
    xgb_gpu = detect_xgb_gpu(Xprobe, yprobe)

    log("\n=== 2/4 ENSEMBLE MODELING AND TIME-SERIES CROSS-VALIDATION ===")
    better_metrics, _, _, _, cv_summary = better_stage(repo, train, val, xgb_gpu, smoke=args.smoke)
    log(better_metrics.to_string(index=False))
    log("CV summary:\n" + cv_summary.to_string(index=False))

    log("\n=== 3/4 HYPERPARAMETER TUNING ===")
    tuned_xgb_params, tuned_xgb_met = optuna_xgb(repo, train, val, xgb_gpu, smoke=args.smoke)
    gpu_report = tensorflow_gpu_report(repo)
    log("TensorFlow GPU report: " + json.dumps(gpu_report))
    mlp_hp, mlp_met, mlp_epoch = keras_tune_mlp(repo, train, val, smoke=args.smoke)

    validation_table = pd.concat([
        baseline_metrics.drop(columns=[c for c in ["fit_seconds"] if c in baseline_metrics.columns]),
        better_metrics.drop(columns=[c for c in ["fit_seconds"] if c in better_metrics.columns]),
        pd.DataFrame([
            {"model": "tuned_xgboost", **tuned_xgb_met},
            {"model": "tuned_mlp", **mlp_met},
        ]),
    ], ignore_index=True, sort=False)
    selected = choose_final(validation_table)
    log("Model selected before locked-test evaluation: " + json.dumps(selected, default=float))
    (out / "model_selection.json").write_text(json.dumps(selected, indent=2, default=float) + "\n")

    final_metrics, _, final_met = final_locked_test(
        repo, train, val, test, xgb_gpu, tuned_xgb_params, mlp_hp, mlp_epoch, selected["model"]
    )
    log("LOCKED-TEST EVALUATION:\n" + final_metrics.to_string(index=False))

    elapsed = time.time() - start
    summary = generate_summary(
        repo, audit, split_info, baseline_metrics, better_metrics, cv_summary,
        tuned_xgb_met, mlp_met, selected, final_metrics, gpu_report, elapsed,
    )
    generate_readme(repo, summary)

    log("\n=== 4/4 REPORT GENERATION ===")
    nb = make_notebook(repo, summary)
    log(f"Notebook: {nb}")

    if not args.skip_codex_audit:
        log("\n=== METHODOLOGY AUDIT ===")
        audit_text = codex_audit(repo)
        log(audit_text)
    else:
        audit_text = "AUDIT_SKIPPED_BY_USER"

    # Final reproducibility checks for required artifacts and finite evaluation metrics.
    required = [
        repo / "artifacts/results/baseline_models/metrics.csv",
        repo / "artifacts/results/ensemble_models/cv_summary.csv",
        repo / "artifacts/results/hyperparameter_tuning/optuna_xgb.json",
        repo / "artifacts/results/hyperparameter_tuning/keras_tuner_mlp.json",
        repo / "artifacts/results/final_evaluation/locked_test_metrics.csv",
        nb,
        repo / "README.md",
    ]
    missing = [str(p) for p in required if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError("Final gate failed; missing artifacts: " + ", ".join(missing))
    # Make sure the final selected row exists and all numbers are finite.
    if not all(np.isfinite([final_met["mae"], final_met["rmse"], final_met["r2"], final_met["smape_pct"]])):
        raise RuntimeError("Final gate failed; non-finite locked-test metrics")

    git_finalize(repo, args.push)
    elapsed = time.time() - start
    log("\nFORECASTING_PIPELINE_COMPLETE")
    log(f"Elapsed: {elapsed/60:.1f} minutes")
    log(f"Final notebook: {nb}")
    log(f"README: {repo/'README.md'}")
    log(f"Locked-test metrics: {repo/'artifacts/results/final_evaluation/locked_test_metrics.csv'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nExecution interrupted; completed artifacts remain preserved.")
        raise SystemExit(130)
    except Exception as exc:
        log("\nFORECASTING_PIPELINE_FAILED")
        log(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise SystemExit(2)
