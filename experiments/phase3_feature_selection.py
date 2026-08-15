"""Select Phase 2 features using only a chronological training prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit


DEFAULT_PHASE2_DIR = Path("artifacts/phase2/phase2_attempt_1_20260815T204500Z")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/phase3")
SEASON_CODES = {"winter": 0.0, "spring": 1.0, "summer": 2.0, "autumn": 3.0}


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for deterministic training-only feature selection."""

    training_fraction: float = 0.70
    n_splits: int = 3
    n_estimators: int = 200
    permutation_repeats: int = 5
    redundancy_threshold: float = 0.98
    max_features: int = 20
    stability_top_k: int = 15
    random_seed: int = 42

    def validate(self, row_count: int, feature_count: int) -> int:
        if not 0.0 < self.training_fraction < 1.0:
            raise ValueError("training_fraction must be strictly between zero and one")
        if self.n_splits < 2 or self.n_estimators < 1 or self.permutation_repeats < 1:
            raise ValueError("split, estimator, and permutation counts must be positive")
        if not 0.0 < self.redundancy_threshold <= 1.0:
            raise ValueError("redundancy_threshold must be in (0, 1]")
        if not 1 <= self.max_features <= feature_count:
            raise ValueError("max_features must be between one and the candidate count")
        if not 1 <= self.stability_top_k <= feature_count:
            raise ValueError("stability_top_k must be between one and the candidate count")
        training_rows = int(np.floor(row_count * self.training_fraction))
        if training_rows <= self.n_splits:
            raise ValueError("training prefix is too short for the requested time splits")
        return training_rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_phase2_manifest(manifest: dict[str, Any]) -> tuple[str, str, list[str], int]:
    date_column = manifest.get("date_column")
    target_column = manifest.get("target_column")
    features = manifest.get("feature_columns")
    row_count = manifest.get("row_count")
    selection = manifest.get("feature_selection", {})
    causality = manifest.get("causality", {})
    if date_column != "Date" or target_column != "Consumption":
        raise ValueError("unexpected Phase 2 date or target column")
    if not isinstance(features, list) or not features or len(features) != len(set(features)):
        raise ValueError("Phase 2 feature list must be non-empty and unique")
    if not isinstance(row_count, int) or row_count < 1:
        raise ValueError("Phase 2 row count is invalid")
    if selection.get("performed") is not False:
        raise ValueError("input must be the unselected Phase 2 candidate set")
    required_causality = {
        "target_derived_features",
        "rolling_policy",
        "calendar_offset_policy",
        "calendar_features",
    }
    if not required_causality.issubset(causality):
        raise ValueError("Phase 2 causality evidence is incomplete")
    return date_column, target_column, features, row_count


def load_training_prefix(
    phase2_dir: Path, config: SelectionConfig
) -> tuple[pd.DataFrame, dict[str, Any], int, str]:
    """Load no rows beyond the declared chronological training boundary."""
    phase2_dir = phase2_dir.resolve()
    manifest_path = phase2_dir / "feature_manifest.json"
    features_path = phase2_dir / "features.csv"
    manifest = _read_json(manifest_path)
    date_column, target_column, feature_columns, row_count = _validate_phase2_manifest(manifest)
    training_rows = config.validate(row_count, len(feature_columns))
    expected_columns = [date_column, target_column, *feature_columns]
    frame = pd.read_csv(features_path, nrows=training_rows, usecols=expected_columns)
    if len(frame) != training_rows or list(frame.columns) != expected_columns:
        raise ValueError("Phase 2 feature artifact does not satisfy its manifest")
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
    if frame[date_column].duplicated().any() or not frame[date_column].is_monotonic_increasing:
        raise ValueError("training dates must be unique and strictly chronological")
    target = pd.to_numeric(frame[target_column], errors="raise")
    if target.isna().any() or not np.isfinite(target).all():
        raise ValueError("training target must be finite and complete")
    return frame, manifest, training_rows, sha256_file(features_path)


def _numeric_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in feature_columns:
        if column == "season":
            converted = frame[column].map(SEASON_CODES)
            if converted.isna().any():
                raise ValueError("season contains an unknown category")
            result[column] = converted
        else:
            result[column] = pd.to_numeric(frame[column], errors="raise")
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.astype(float)


def leakage_screen(
    frame: pd.DataFrame, feature_columns: list[str], target_column: str
) -> dict[str, Any]:
    """Screen candidate names and values without examining non-training rows."""
    forbidden_names = {target_column, "Date", "target", "future_consumption", "lead_consumption"}
    forbidden_candidates = sorted(forbidden_names.intersection(feature_columns))
    exact_target_matches: list[str] = []
    target = frame[target_column].to_numpy(dtype=float)
    numeric = _numeric_features(frame, feature_columns)
    for column in feature_columns:
        values = numeric[column].to_numpy(dtype=float)
        observed = np.isfinite(values)
        if observed.all() and np.array_equal(values, target):
            exact_target_matches.append(column)
    if forbidden_candidates or exact_target_matches:
        raise ValueError(
            f"leakage screening failed: names={forbidden_candidates}, "
            f"exact_target_matches={exact_target_matches}"
        )
    return {
        "status": "passed",
        "forbidden_candidate_names": forbidden_candidates,
        "exact_target_matches": exact_target_matches,
        "phase2_causality_manifest_required": True,
        "rows_screened": len(frame),
    }


def _fit_imputation_values(frame: pd.DataFrame) -> pd.Series:
    medians = frame.median(axis=0, skipna=True)
    return medians.fillna(0.0)


def _importance_across_time(
    features: pd.DataFrame, target: pd.Series, config: SelectionConfig
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    names = list(features.columns)
    records: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []
    splitter = TimeSeriesSplit(n_splits=config.n_splits)
    for fold, (fit_indices, assess_indices) in enumerate(splitter.split(features), start=1):
        fit = features.iloc[fit_indices]
        assess = features.iloc[assess_indices]
        imputation_values = _fit_imputation_values(fit)
        fit_values = fit.fillna(imputation_values)
        assess_values = assess.fillna(imputation_values)
        model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            random_state=config.random_seed + fold,
            n_jobs=1,
        )
        model.fit(fit_values, target.iloc[fit_indices])
        permutation = permutation_importance(
            model,
            assess_values,
            target.iloc[assess_indices],
            scoring="neg_mean_absolute_error",
            n_repeats=config.permutation_repeats,
            random_state=config.random_seed + fold,
            n_jobs=1,
        )
        permutation_values = permutation.importances_mean
        top = set(np.argsort(permutation_values)[-config.stability_top_k :])
        for index, name in enumerate(names):
            records.append(
                {
                    "fold": fold,
                    "feature": name,
                    "model_importance": float(model.feature_importances_[index]),
                    "permutation_importance_mae": float(permutation_values[index]),
                    "in_fold_top_k": int(index in top),
                }
            )
        split_records.append(
            {
                "fold": fold,
                "fit_start_row": int(fit_indices[0]),
                "fit_end_row_inclusive": int(fit_indices[-1]),
                "assessment_start_row": int(assess_indices[0]),
                "assessment_end_row_inclusive": int(assess_indices[-1]),
                "imputation_fitted_on_fit_rows_only": True,
            }
        )
    details = pd.DataFrame(records)
    summary = details.groupby("feature", sort=False).agg(
        model_importance_mean=("model_importance", "mean"),
        permutation_importance_mae_mean=("permutation_importance_mae", "mean"),
        stability_fraction=("in_fold_top_k", "mean"),
    )
    model_scale = summary["model_importance_mean"].max()
    permutation_positive = summary["permutation_importance_mae_mean"].clip(lower=0.0)
    permutation_scale = permutation_positive.max()
    summary["combined_score"] = (
        summary["model_importance_mean"] / (model_scale if model_scale > 0 else 1.0)
        + permutation_positive / (permutation_scale if permutation_scale > 0 else 1.0)
        + summary["stability_fraction"]
    ) / 3.0
    return summary.reset_index(), split_records


def _redundancy_pairs(features: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    correlations = features.corr(method="pearson", min_periods=2).abs()
    names = list(features.columns)
    pairs: list[dict[str, Any]] = []
    for right_index, right in enumerate(names):
        for left in names[:right_index]:
            value = correlations.loc[left, right]
            if pd.notna(value) and value >= threshold:
                pairs.append({"feature_a": left, "feature_b": right, "absolute_correlation": float(value)})
    return pairs


def select_features(
    training_frame: pd.DataFrame,
    manifest: dict[str, Any],
    config: SelectionConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    date_column, target_column, candidates, _ = _validate_phase2_manifest(manifest)
    leakage = leakage_screen(training_frame, candidates, target_column)
    numeric = _numeric_features(training_frame, candidates)
    constant_features = [name for name in candidates if numeric[name].nunique(dropna=True) <= 1]
    eligible = [name for name in candidates if name not in constant_features]
    if not eligible:
        raise ValueError("no eligible features remain after domain sanity checks")
    importance, split_records = _importance_across_time(
        numeric[eligible], training_frame[target_column].astype(float), config
    )
    scores = importance.set_index("feature")["combined_score"]
    redundancy = _redundancy_pairs(numeric[eligible], config.redundancy_threshold)
    correlated: dict[str, set[str]] = {name: set() for name in eligible}
    for pair in redundancy:
        correlated[pair["feature_a"]].add(pair["feature_b"])
        correlated[pair["feature_b"]].add(pair["feature_a"])
    redundant_drops: dict[str, str] = {}
    retained_representatives: list[str] = []
    score_order = sorted(eligible, key=lambda name: (-scores[name], name))
    for name in score_order:
        representative = next(
            (retained for retained in retained_representatives if retained in correlated[name]), None
        )
        if representative is None:
            retained_representatives.append(name)
        else:
            redundant_drops[name] = representative
    importance["domain_sanity_passed"] = True
    importance["redundant_with"] = importance["feature"].map(redundant_drops).fillna("")
    importance["eligible_after_redundancy"] = ~importance["feature"].isin(redundant_drops)
    ranked = importance.loc[importance["eligible_after_redundancy"]].sort_values(
        ["combined_score", "feature"], ascending=[False, True]
    )
    selected = ranked.head(config.max_features)["feature"].tolist()
    if not selected:
        raise ValueError("feature selection produced an empty set")
    importance["selected"] = importance["feature"].isin(selected)
    importance = importance.sort_values(["selected", "combined_score"], ascending=[False, False])
    report = {
        "leakage_screening": leakage,
        "domain_sanity": {
            "constant_features_excluded": constant_features,
            "finite_values_required_after_training-fitted_imputation": True,
            "known_season_encoding": SEASON_CODES,
        },
        "redundancy": {
            "threshold": config.redundancy_threshold,
            "pairs_at_or_above_threshold": redundancy,
            "dropped_feature_to_retained_feature": redundant_drops,
        },
        "importance": {
            "model": "RandomForestRegressor",
            "model_importance": "mean impurity importance across expanding-window folds",
            "permutation_importance": "mean increase in MAE on each later in-training assessment fold",
            "stability": f"fraction of folds ranked in the top {config.stability_top_k}",
            "combined_score": "equal-weight mean of normalized model importance, non-negative normalized permutation importance, and stability",
        },
        "time_splits": split_records,
        "selected_features": selected,
        "selected_feature_count": len(selected),
        "candidate_feature_count": len(candidates),
    }
    return importance, report


def run_phase3_selection(
    phase2_dir: Path = DEFAULT_PHASE2_DIR,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    experiment_id: str | None = None,
    config: SelectionConfig | None = None,
) -> Path:
    config = config or SelectionConfig()
    if experiment_id is None:
        experiment_id = datetime.now(timezone.utc).strftime("phase3_%Y%m%dT%H%M%SZ")
    output_dir = artifact_root.resolve() / experiment_id
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    frame, manifest, training_rows, feature_hash_before = load_training_prefix(phase2_dir, config)
    importance, report = select_features(frame, manifest, config)
    feature_hash_after = sha256_file(Path(phase2_dir).resolve() / "features.csv")
    if feature_hash_after != feature_hash_before:
        raise RuntimeError("Phase 2 features changed during selection")

    importance.to_csv(output_dir / "feature_ranking.csv", index=False)
    report.update(
        {
            "phase": "Phase 3",
            "feature_selection_fitted_on": "chronological training prefix only",
            "training_rows": training_rows,
            "training_start_date": frame[manifest["date_column"]].iloc[0].date().isoformat(),
            "training_end_date_inclusive": frame[manifest["date_column"]].iloc[-1].date().isoformat(),
            "validation_rows_loaded": 0,
            "locked_test_rows_loaded": 0,
            "phase2_features_sha256": feature_hash_before,
            "phase2_source_unchanged": True,
        }
    )
    _write_json(output_dir / "selection_report.json", report)
    _write_json(
        output_dir / "config.json",
        {
            "phase": "Phase 3",
            "experiment_id": experiment_id,
            "phase2_directory": str(Path(phase2_dir).resolve()),
            "selection_config": asdict(config),
            "random_seed": config.random_seed,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": ["config.json", "feature_ranking.csv", "selection_report.json"],
        },
    )
    return output_dir


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-id")
    args = parser.parse_args(argv)
    print(run_phase3_selection(args.phase2_dir, args.artifact_root, args.experiment_id))


if __name__ == "__main__":
    main()
