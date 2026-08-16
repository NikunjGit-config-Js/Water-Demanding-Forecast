"""POST-HOC ADVANCED MODEL BENCHMARK.

Tests XGBoost, ExtraTrees, and HistGradientBoosting against validated baselines
on London and Delhi. This is a separate post-hoc experiment and does NOT modify
the original Phase 0-13 results.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    city: str
    protocol: str
    mae: float
    mse: float
    rmse: float
    r2: float
    runtime_seconds: float


def load_city_data(city: str) -> tuple[pd.DataFrame, list[str], int]:
    """Load Phase 2 features and Phase 3 selected features for a city."""
    from orchestration.context import RunContext
    ctx = RunContext.for_city(city)

    phase2_dirs = sorted((ctx.artifact_root / "phase2").iterdir()) if (ctx.artifact_root / "phase2").exists() else []
    phase3_dirs = sorted((ctx.artifact_root / "phase3").iterdir()) if (ctx.artifact_root / "phase3").exists() else []
    if not phase2_dirs or not phase3_dirs:
        raise FileNotFoundError(f"Phase 2/3 artifacts not found for {city}")

    phase2_dir = phase2_dirs[-1]
    phase3_dir = phase3_dirs[-1]

    manifest = json.loads((phase2_dir / "feature_manifest.json").read_text())
    selection = json.loads((phase3_dir / "selection_report.json").read_text())

    features_df = pd.read_csv(phase2_dir / "features.csv")
    selected = selection["selected_features"]
    row_count = manifest["row_count"]

    return features_df, selected, row_count


def prepare_xy(features_df: pd.DataFrame, selected: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X and y from the features dataframe, dropping NaN rows from lag/rolling features."""
    df = features_df.copy()
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)
    target_col = [c for c in df.columns if c.lower() == "consumption"]
    if not target_col:
        raise ValueError(f"No 'consumption' column found in {df.columns.tolist()}")
    y = df[target_col[0]].copy()
    X = df[selected].copy()
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    return X, y


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def naive_lag1_baseline(y: pd.Series, n_train: int) -> np.ndarray:
    """Naive lag-1 prediction: use previous actual value."""
    predictions = np.empty(len(y) - n_train)
    for i in range(n_train, len(y)):
        predictions[i - n_train] = y.iloc[i - 1]
    return predictions


def run_holdout_benchmark(city: str, features_df: pd.DataFrame, selected: list[str]) -> list[BenchmarkResult]:
    """Phase-5-style chronological 80/20 holdout benchmark."""
    X, y = prepare_xy(features_df, selected)
    n = len(X)
    split = int(np.floor(n * 0.8))

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "naive_lag_1": None,
        "ridge": Ridge(alpha=1.0),
        "xgboost": XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0, n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, max_depth=6, random_state=42,
        ),
    }

    results = []
    for name, model in models.items():
        start = time.time()
        if name == "naive_lag_1":
            preds = naive_lag1_baseline(y, split)
        else:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
        runtime = time.time() - start

        metrics = evaluate(y_test.values, preds)
        results.append(BenchmarkResult(
            model=name, city=city, protocol="holdout_80_20",
            mae=metrics["mae"], mse=metrics["mse"],
            rmse=metrics["rmse"], r2=metrics["r2"],
            runtime_seconds=round(runtime, 3),
        ))

    return results


def run_cv_benchmark(city: str, features_df: pd.DataFrame, selected: list[str]) -> list[BenchmarkResult]:
    """Phase-6-style 5-fold expanding-window TimeSeriesSplit benchmark."""
    X, y = prepare_xy(features_df, selected)
    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        "naive_lag_1": None,
        "ridge": Ridge(alpha=1.0),
        "xgboost": XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0, n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, max_depth=6, random_state=42,
        ),
    }

    all_results = {name: [] for name in models}
    runtimes = {name: 0.0 for name in models}

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for name, model in models.items():
            start = time.time()
            if name == "naive_lag_1":
                preds = np.asarray([y.iloc[i - 1] for i in test_idx])
            else:
                model.fit(X_train_s, y_train)
                preds = model.predict(X_test_s)
            runtimes[name] += time.time() - start

            metrics = evaluate(y_test.values, preds)
            all_results[name].append(metrics)

    results = []
    for name in models:
        avg = {
            "mae": np.mean([r["mae"] for r in all_results[name]]),
            "mse": np.mean([r["mse"] for r in all_results[name]]),
            "rmse": np.mean([r["rmse"] for r in all_results[name]]),
            "r2": np.mean([r["r2"] for r in all_results[name]]),
        }
        results.append(BenchmarkResult(
            model=name, city=city, protocol="cv_5fold_tscv",
            mae=round(float(avg["mae"]), 6),
            mse=round(float(avg["mse"]), 6),
            rmse=round(float(avg["rmse"]), 6),
            r2=round(float(avg["r2"]), 6),
            runtime_seconds=round(runtimes[name] / 5, 3),
        ))

    return results


def run_benchmark(city: str) -> list[BenchmarkResult]:
    """Run full benchmark for a city."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {city.upper()}")
    print(f"{'='*60}")

    features_df, selected, row_count = load_city_data(city)
    print(f"Rows: {row_count}, Selected features: {len(selected)}")

    holdout_results = run_holdout_benchmark(city, features_df, selected)
    cv_results = run_cv_benchmark(city, features_df, selected)

    return holdout_results + cv_results


def main():
    output_dir = PROJECT_ROOT / "reports" / "advanced_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for city in ["london", "delhi"]:
        try:
            results = run_benchmark(city)
            all_results.extend(results)
        except Exception as e:
            print(f"ERROR benchmarking {city}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    results_json = [asdict(r) for r in all_results]
    (output_dir / "advanced_benchmark_results.json").write_text(
        json.dumps(results_json, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Print summary
    print(f"\n{'='*60}")
    print("ADVANCED MODEL BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'City':<10} {'Protocol':<20} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 80)
    for r in all_results:
        print(f"{r.model:<25} {r.city:<10} {r.protocol:<20} {r.mae:>8.2f} {r.rmse:>8.2f} {r.r2:>8.4f}")

    # Identify best per city/protocol
    for city in ["london", "delhi"]:
        for protocol in ["holdout_80_20", "cv_5fold_tscv"]:
            city_proto = [r for r in all_results if r.city == city and r.protocol == protocol]
            if city_proto:
                best = min(city_proto, key=lambda r: r.mae)
                print(f"\nBest for {city} ({protocol}): {best.model} (MAE={best.mae:.2f})")


if __name__ == "__main__":
    main()
