"""Build deterministic, past-only Phase 2 water-demand features."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import holidays
import numpy as np
import pandas as pd


DEFAULT_DATASET = Path("data/preprocessed/all/preprocessed_data.csv")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/phase2")
LAGS = (1, 7, 14, 30, 365)
ROLLING_WINDOWS = (7, 30)
EMA_SPANS = (7, 30)
FOURIER_EPOCH = pd.Timestamp("1970-01-01")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != ["Date", "Consumption"]:
        raise ValueError("dataset columns must be exactly Date and Consumption")
    parsed = frame.copy()
    parsed["Date"] = pd.to_datetime(parsed["Date"], errors="raise")
    parsed["Consumption"] = pd.to_numeric(parsed["Consumption"], errors="raise")
    if parsed.isna().any().any() or not np.isfinite(parsed["Consumption"]).all():
        raise ValueError("dataset must not contain missing or non-finite values")
    if not parsed["Date"].is_monotonic_increasing or parsed["Date"].duplicated().any():
        raise ValueError("dates must be unique and strictly chronological")
    return parsed


def _rolling_slope(values: np.ndarray) -> float:
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def _add_fourier_terms(result: pd.DataFrame, elapsed_days: np.ndarray, period: float, prefix: str) -> None:
    for harmonic in (1, 2):
        angle = 2.0 * np.pi * harmonic * elapsed_days / period
        result[f"{prefix}_sin_{harmonic}"] = np.sin(angle)
        result[f"{prefix}_cos_{harmonic}"] = np.cos(angle)


def build_past_only_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return target plus features available immediately before each row's target.

    Observation lags deliberately refer to earlier observed rows. The explicitly
    calendar-based previous-week/year features instead require an exact date
    match, which prevents source-data gaps from changing their meaning.
    """
    parsed = _validate_frame(frame)
    dates = parsed["Date"]
    target = parsed["Consumption"].astype(float)
    past_target = target.shift(1)
    result = parsed.copy()

    for lag in LAGS:
        result[f"consumption_lag_{lag}"] = target.shift(lag)

    for window in ROLLING_WINDOWS:
        rolling = past_target.rolling(window=window, min_periods=window)
        result[f"rolling_mean_{window}"] = rolling.mean()
        result[f"rolling_median_{window}"] = rolling.median()
        result[f"rolling_std_{window}"] = rolling.std(ddof=1)
        result[f"rolling_min_{window}"] = rolling.min()
        result[f"rolling_max_{window}"] = rolling.max()
        result[f"rolling_slope_{window}"] = rolling.apply(_rolling_slope, raw=True)

    for span in EMA_SPANS:
        result[f"ema_{span}"] = past_target.ewm(span=span, adjust=False, min_periods=span).mean()

    result["day_over_day_change"] = target.shift(1) - target.shift(2)
    denominator = target.shift(8).replace(0.0, np.nan)
    result["growth_7d"] = target.shift(1).sub(target.shift(8)).div(denominator)

    result["day_of_week"] = dates.dt.dayofweek.astype("int8")
    result["month"] = dates.dt.month.astype("int8")
    result["is_weekend"] = dates.dt.dayofweek.ge(5).astype("int8")
    result["season"] = pd.cut(
        dates.dt.month,
        bins=[0, 2, 5, 8, 11, 12],
        labels=["winter", "spring", "summer", "autumn", "winter"],
        ordered=False,
    ).astype("string")
    result["day_of_year_sin"] = np.sin(2 * np.pi * (dates.dt.dayofyear - 1) / 365.2425)
    result["day_of_year_cos"] = np.cos(2 * np.pi * (dates.dt.dayofyear - 1) / 365.2425)
    result["day_of_week_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
    result["day_of_week_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)

    elapsed_days = (dates - FOURIER_EPOCH).dt.days.to_numpy(dtype=float)
    _add_fourier_terms(result, elapsed_days, 7.0, "fourier_weekly")
    _add_fourier_terms(result, elapsed_days, 365.2425, "fourier_yearly")

    canada_holidays = holidays.CA(
        years=range(int(dates.dt.year.min()), int(dates.dt.year.max()) + 1), subdiv="ON"
    )
    result["is_canada_ontario_holiday"] = dates.dt.date.map(
        lambda value: int(value in canada_holidays)
    ).astype("int8")

    by_date = pd.Series(target.to_numpy(), index=pd.DatetimeIndex(dates))
    result["same_weekday_previous_week"] = (dates - pd.Timedelta(days=7)).map(by_date)
    result["same_period_previous_year"] = dates.map(
        lambda value: by_date.get(value - pd.DateOffset(years=1), np.nan)
    )
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_safe) + "\n",
        encoding="utf-8",
    )


def _feature_manifest(features: pd.DataFrame) -> dict[str, Any]:
    feature_columns = [column for column in features if column not in {"Date", "Consumption"}]
    return {
        "target_column": "Consumption",
        "date_column": "Date",
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "row_count": len(features),
        "rows_complete_for_all_features": int(features[feature_columns].notna().all(axis=1).sum()),
        "missing_by_feature": {
            column: int(features[column].isna().sum()) for column in feature_columns
        },
        "causality": {
            "target_derived_features": "computed only from Consumption values at timestamps strictly before the prediction timestamp",
            "rolling_policy": "shift target by one observation before every rolling/EMA calculation",
            "calendar_offset_policy": "previous-week/year values require an exact source date match",
            "calendar_features": "derived only from the known prediction date",
        },
        "feature_selection": {
            "performed": False,
            "reason": "Feature selection is Phase 3 and must be fitted on training data only.",
            "validation_or_test_accessed": False,
        },
    }


def run_phase2_features(
    dataset_path: Path = DEFAULT_DATASET,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    experiment_id: str | None = None,
) -> Path:
    dataset_path = dataset_path.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if experiment_id is None:
        experiment_id = datetime.now(timezone.utc).strftime("phase2_%Y%m%dT%H%M%SZ")
    output_dir = artifact_root.resolve() / experiment_id
    output_dir.mkdir(parents=True, exist_ok=False)
    source_hash_before = sha256_file(dataset_path)
    started_at = datetime.now(timezone.utc)

    features = build_past_only_features(pd.read_csv(dataset_path))
    features.to_csv(output_dir / "features.csv", index=False, date_format="%Y-%m-%d")
    manifest = _feature_manifest(features)
    _write_json(output_dir / "feature_manifest.json", manifest)

    source_hash_after = sha256_file(dataset_path)
    if source_hash_after != source_hash_before:
        raise RuntimeError("source dataset changed during Phase 2 feature engineering")
    _write_json(
        output_dir / "config.json",
        {
            "phase": "Phase 2",
            "experiment_id": experiment_id,
            "dataset": {"path": str(dataset_path), "sha256": source_hash_before},
            "source_unchanged": True,
            "lags": list(LAGS),
            "rolling_windows": list(ROLLING_WINDOWS),
            "ema_spans": list(EMA_SPANS),
            "holiday_calendar": "Canada, Ontario subdivision",
            "fourier": {"weekly_period_days": 7.0, "yearly_period_days": 365.2425, "harmonics": [1, 2]},
            "missing_feature_policy": "preserve warm-up and calendar-gap NaNs; do not impute or drop rows",
            "random_seed": None,
            "random_seed_note": "No randomized operations are used.",
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "holidays": holidays.__version__,
            },
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": ["config.json", "feature_manifest.json", "features.csv"],
        },
    )
    return output_dir


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-id")
    args = parser.parse_args(argv)
    print(run_phase2_features(args.dataset, args.artifact_root, args.experiment_id))


if __name__ == "__main__":
    main()
