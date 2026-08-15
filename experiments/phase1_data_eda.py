"""Validate the supplied daily water-demand data and preserve Phase 1 EDA artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


DEFAULT_DATASET = Path("data/preprocessed/all/preprocessed_data.csv")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/phase1")
EXPECTED_COLUMNS = ["Date", "Consumption"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )


def validate_daily_consumption(dataset_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse without reordering and return the frame plus a complete quality report."""
    raw = pd.read_csv(dataset_path, dtype=str)
    if list(raw.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"dataset columns must be exactly {EXPECTED_COLUMNS}")

    dates = pd.to_datetime(raw["Date"], errors="coerce")
    consumption = pd.to_numeric(raw["Consumption"], errors="coerce")
    invalid_date_mask = dates.isna() & raw["Date"].notna()
    invalid_consumption_mask = consumption.isna() & raw["Consumption"].notna()
    frame = pd.DataFrame({"Date": dates, "Consumption": consumption})

    valid_dates = dates.dropna()
    date_differences = valid_dates.diff()
    non_increasing_mask = date_differences <= pd.Timedelta(0)
    positive_differences = date_differences[date_differences > pd.Timedelta(0)]
    gap_mask = positive_differences > pd.Timedelta(days=1)
    gaps = []
    for row_index, difference in positive_differences[gap_mask].items():
        current_position = valid_dates.index.get_loc(row_index)
        gaps.append(
            {
                "previous_date": valid_dates.iloc[current_position - 1].strftime("%Y-%m-%d"),
                "next_date": valid_dates.loc[row_index].strftime("%Y-%m-%d"),
                "elapsed_days": int(difference / pd.Timedelta(days=1)),
                "missing_calendar_days": int(difference / pd.Timedelta(days=1)) - 1,
            }
        )

    finite_consumption = consumption[np.isfinite(consumption)]
    q1 = finite_consumption.quantile(0.25) if not finite_consumption.empty else np.nan
    q3 = finite_consumption.quantile(0.75) if not finite_consumption.empty else np.nan
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_mask = (consumption < lower_fence) | (consumption > upper_fence)

    report = {
        "shape": {"rows": len(raw), "columns": len(raw.columns)},
        "columns": EXPECTED_COLUMNS,
        "parsed_dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "date_range": {
            "start": valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else None,
            "end": valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else None,
        },
        "missing_values": {name: int(value) for name, value in raw.isna().sum().items()},
        "invalid_values": {
            "invalid_date_count": int(invalid_date_mask.sum()),
            "invalid_date_rows": raw.index[invalid_date_mask].tolist(),
            "non_numeric_consumption_count": int(invalid_consumption_mask.sum()),
            "non_numeric_consumption_rows": raw.index[invalid_consumption_mask].tolist(),
            "non_finite_consumption_count": int((consumption.notna() & ~np.isfinite(consumption)).sum()),
            "non_positive_consumption_count": int((consumption <= 0).sum()),
        },
        "duplicates": {
            "duplicate_row_count": int(raw.duplicated().sum()),
            "duplicate_date_count": int((dates.notna() & dates.duplicated(keep=False)).sum()),
        },
        "chronology": {
            "strictly_increasing": bool(
                not valid_dates.empty and not invalid_date_mask.any() and not non_increasing_mask.any()
            ),
            "non_increasing_transition_count": int(non_increasing_mask.sum()),
            "observed_frequency": pd.infer_freq(valid_dates) if len(valid_dates) >= 3 else None,
            "gap_count": len(gaps),
            "missing_calendar_day_count": sum(gap["missing_calendar_days"] for gap in gaps),
            "gaps": gaps,
        },
        "suspicious_observations": {
            "method": "Tukey 1.5*IQR inspection candidates; values are not removed or called anomalies",
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "candidate_count": int(outlier_mask.sum()),
            "rows_with_repeated_consumption_value": int(
                (consumption.notna() & consumption.duplicated(keep=False)).sum()
            ),
            "repeated_value_note": (
                "Repeated target values are reported for inspection but are not duplicate rows."
            ),
        },
    }
    return frame, report


def _save_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    dates = frame["Date"]
    target = frame["Consumption"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(target.dropna(), bins=40, edgecolor="white")
    axes[0].set(title="Consumption distribution", xlabel="Consumption", ylabel="Count")
    axes[1].boxplot(target.dropna(), orientation="vertical")
    axes[1].set(title="Consumption box plot", ylabel="Consumption", xticks=[])
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution_boxplot.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 5))
    axis.plot(dates, target, linewidth=0.8, alpha=0.7, label="Daily consumption")
    axis.plot(dates, target.rolling(30, min_periods=30).mean(), linewidth=1.4, label="30-observation trailing mean")
    axis.plot(dates, target.rolling(365, min_periods=365).mean(), linewidth=1.7, label="365-observation trailing mean")
    axis.set(title="Daily water consumption and trailing trends", xlabel="Date", ylabel="Consumption")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "time_series_trend.png", dpi=150)
    plt.close(fig)

    calendar = frame.assign(
        month=dates.dt.month,
        weekday=dates.dt.day_name(),
        year=dates.dt.year,
    )
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    calendar.groupby("month", observed=True)["Consumption"].mean().plot(ax=axes[0], marker="o")
    calendar.groupby("weekday", observed=True)["Consumption"].mean().reindex(weekday_order).plot(ax=axes[1], marker="o")
    axes[0].set(title="Mean consumption by month", xlabel="Month", ylabel="Mean consumption", xticks=range(1, 13))
    axes[1].set(title="Mean consumption by weekday", xlabel="Weekday", ylabel="Mean consumption")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "calendar_seasonality.png", dpi=150)
    plt.close(fig)


def run_phase1_eda(
    dataset_path: Path = DEFAULT_DATASET,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    experiment_id: str | None = None,
) -> Path:
    """Run non-mutating validation/EDA and return the new artifact directory."""
    dataset_path = dataset_path.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if experiment_id is None:
        experiment_id = datetime.now(timezone.utc).strftime("phase1_%Y%m%dT%H%M%SZ")
    output_dir = artifact_root.resolve() / experiment_id
    output_dir.mkdir(parents=True, exist_ok=False)
    source_hash_before = sha256_file(dataset_path)
    started = datetime.now(timezone.utc)

    frame, quality = validate_daily_consumption(dataset_path)
    descriptive = frame["Consumption"].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    descriptive.rename_axis("statistic").to_csv(output_dir / "descriptive_statistics.csv", header=["Consumption"])

    lower = quality["suspicious_observations"]["lower_fence"]
    upper = quality["suspicious_observations"]["upper_fence"]
    candidates = frame.loc[(frame["Consumption"] < lower) | (frame["Consumption"] > upper)].copy()
    candidates["reason"] = np.where(candidates["Consumption"] < lower, "below_IQR_fence", "above_IQR_fence")
    candidates.to_csv(output_dir / "suspicious_observations.csv", index=False, date_format="%Y-%m-%d")

    calendar_summary = pd.concat(
        {
            "month": frame.groupby(frame["Date"].dt.month)["Consumption"].agg(["count", "mean", "median", "std"]),
            "weekday": frame.groupby(frame["Date"].dt.day_name())["Consumption"].agg(["count", "mean", "median", "std"]),
            "year": frame.groupby(frame["Date"].dt.year)["Consumption"].agg(["count", "mean", "median", "std"]),
        },
        names=["grouping", "value"],
    )
    calendar_summary.to_csv(output_dir / "calendar_summary.csv")
    _save_plots(frame, output_dir)

    source_hash_after = sha256_file(dataset_path)
    if source_hash_after != source_hash_before:
        raise RuntimeError("source dataset changed during Phase 1 analysis")
    quality["source_preservation"] = {
        "sha256_before": source_hash_before,
        "sha256_after": source_hash_after,
        "unchanged": True,
    }
    quality["correlation_analysis"] = {
        "performed": False,
        "reason": "Consumption is the only numeric variable; self-correlation is not informative.",
    }
    _write_json(output_dir / "data_quality_report.json", quality)
    _write_json(
        output_dir / "config.json",
        {
            "phase": "Phase 1",
            "experiment_id": experiment_id,
            "dataset": {"path": str(dataset_path), "sha256": source_hash_before},
            "analysis_scope": "non-mutating validation and univariate exploratory data analysis",
            "outlier_policy": "report Tukey 1.5*IQR candidates; do not remove or transform values",
            "chronology_policy": "preserve input row order; do not sort or impute",
            "random_seed": None,
            "random_seed_note": "No randomized operations are used.",
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            "started_at_utc": started.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": sorted(path.name for path in output_dir.iterdir()) + ["config.json"],
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    print(run_phase1_eda(args.dataset, args.artifact_root, args.experiment_id))


if __name__ == "__main__":
    main()
