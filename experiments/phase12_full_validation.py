"""Fail-closed, reproducible Phase 12 audit of approved forecasting artifacts.

The audit never fits or selects a model.  It anchors the source data and every
audited result to approved digests, then independently validates dates, folds,
source targets, and reported metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestration.flow import _read_persisted_checkpoint  # noqa: E402


DATASET = ROOT / "data/preprocessed/all/preprocessed_data.csv"
# Preserved in the approved Phase 0 configuration and independently repeated
# in Phases 1, 2, and 9.  A digest computed from the file under audit is never
# accepted as its own authority.
APPROVED_DATASET_SHA256 = "2d95a510176c4dad93eae25958c714ab713790512193f55cf5747b44153da6c9"


@dataclass(frozen=True)
class EvaluationSpec:
    predictions: Path
    metrics: Path
    protocol: Path
    cv: bool
    manifest: Path | None = None
    approved_manifest_sha256: str | None = None
    # Phases 5--7 predate artifact_hashes.json.  These digests are anchored to
    # their validator-approved commits (9e44179, 9277bab, and 1fbfa90).
    approved_hashes: tuple[tuple[str, str], ...] = ()


def _artifact_dir(phase: str, run: str) -> Path:
    return ROOT / "artifacts" / phase / run


_P5 = _artifact_dir("phase5", "phase5_attempt_2_20260815T230000Z")
_P6 = _artifact_dir("phase6", "phase6_attempt_1_20260815T233000Z")
_P7 = _artifact_dir("phase7", "phase7_attempt_1_20260815T223000Z")
_P8 = _artifact_dir("phase8", "phase8_attempt_2_final_20260816T003000Z")
_P9 = _artifact_dir("phase9", "phase9_attempt_1_final_20260816T012000Z")
_P10 = _artifact_dir("phase10", "phase10_attempt_1_final_20260816T021000Z")
EVALUATIONS = {
    "phase5": EvaluationSpec(_P5 / "predictions.csv", _P5 / "metrics.csv", _P5 / "holdout_report.json", False,
                             approved_hashes=(("predictions.csv", "2131a887ebdf4900bd53d5d02573042c789c72f970b127bfc111563e17921ef9"), ("metrics.csv", "8f0870ae83755fa77180b4d73a950101edbd2d3f1a12e32eb732de4e25992c02"), ("holdout_report.json", "11d7dab2bd6d948ab287d025a62251e5d254857b0005c5339cc37115b8cbc043"))),
    "phase6": EvaluationSpec(_P6 / "predictions.csv", _P6 / "metrics_summary.csv", _P6 / "cv_report.json", True,
                             approved_hashes=(("predictions.csv", "f34d86aaf21827c3a96763a84c500e489ed3a3772d519948253ebd0384df378e"), ("metrics_summary.csv", "6105242b742e5b2d08a9edc20b6f5380b1ff589d2297a5ecef2ec4a7ad449546"), ("cv_report.json", "2b90df125d15e8a973fba1009fb9c5553580c71e650193b12efa1dbcdfe34b06"))),
    "phase7": EvaluationSpec(_P7 / "locked_test_predictions.csv", _P7 / "locked_test_metrics.csv", _P7 / "split_and_selection_report.json", False,
                             approved_hashes=(("locked_test_predictions.csv", "17941ef9f4367f82fa83abffbec637922139678a4e0370868924dbbb95a1e78e"), ("locked_test_metrics.csv", "5615144d6fb72f857c8a40068cf9d6791b0a91c990fea0271c49b7b1ee078780"), ("split_and_selection_report.json", "fef0e44d1ae6eecf7654c684323f9d794280acc3f437d7ef9ec3fbf303b2506c"))),
    "phase8": EvaluationSpec(_P8 / "predictions.csv", _P8 / "metrics_summary.csv", _P8 / "cv_selection_report.json", True, _P8 / "artifact_hashes.json", "49e8036bbb3d9339ba1a3edd3b4c47e90486d854ecbb09b9c32dac28282cfe7b"),
    "phase9": EvaluationSpec(_P9 / "predictions.csv", _P9 / "metrics.csv", _P9 / "forecast_report.json", False, _P9 / "artifact_hashes.json", "f332d903da3e7fd9e40d11ae4cded28df078c844ee26a957787d75dd968d398a"),
    "phase10": EvaluationSpec(_P10 / "predictions.csv", _P10 / "metrics.csv", _P10 / "forecast_report.json", False, _P10 / "artifact_hashes.json", "c774646adde34a23eaf1805a87e46c10eef32d55a5e235e014726e22321be4c9"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    residual = actual.to_numpy(float) - predicted.to_numpy(float)
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((actual.to_numpy(float) - actual.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(residual))), "mse": mse,
            "rmse": float(np.sqrt(mse)), "r2": float(1.0 - np.sum(residual**2) / denominator)}


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-7, atol=1e-6):
        raise ValueError(f"Metric mismatch for {label}: {actual} != {expected}")


def _verify_hashes(name: str, spec: EvaluationSpec) -> dict[str, str]:
    if spec.manifest is not None:
        manifest_digest = _sha256(spec.manifest)
        if manifest_digest != spec.approved_manifest_sha256:
            raise ValueError(f"{name} approved manifest hash mismatch: {manifest_digest} != {spec.approved_manifest_sha256}")
        manifest = json.loads(spec.manifest.read_text(encoding="utf-8"))
        expected = {path.name: manifest.get(path.name) for path in (spec.predictions, spec.metrics, spec.protocol)}
        source = str(spec.manifest.relative_to(ROOT)) if spec.manifest.is_relative_to(ROOT) else str(spec.manifest)
    else:
        expected = dict(spec.approved_hashes)
        source = "approved validated-commit digest registry"
    calculated = {}
    if spec.manifest is not None:
        calculated["manifest_sha256"] = manifest_digest
    for path in (spec.predictions, spec.metrics, spec.protocol):
        digest = _sha256(path)
        approved = expected.get(path.name)
        if not isinstance(approved, str) or digest != approved:
            raise ValueError(f"{name} hash mismatch for {path.name}: {digest} != {approved}")
        calculated[path.name] = digest
    calculated["authority"] = source
    return calculated


def _source_date(source: pd.DataFrame, row: int) -> str:
    if not 0 <= row < len(source):
        raise ValueError(f"Protocol row {row} lies outside canonical source")
    return source["Date"].iloc[row].date().isoformat()


def _verify_protocol_splits(name: str, protocol: dict[str, Any], source: pd.DataFrame, cv: bool) -> None:
    if cv:
        previous_training_rows = previous_validation_rows = None
        for expected_fold, fold in enumerate(protocol.get("folds", []), 1):
            if int(fold.get("fold", -1)) != expected_fold:
                raise ValueError(f"{name} CV protocol fold numbering is malformed")
            training_rows = int(fold["training_rows"])
            validation_rows = int(fold["validation_rows"])
            if fold["training_start_date"] != _source_date(source, 0):
                raise ValueError(f"{name} fold {expected_fold} training start mismatch")
            if fold["training_end_date_inclusive"] != _source_date(source, training_rows - 1):
                raise ValueError(f"{name} fold {expected_fold} training end/rows mismatch")
            if fold["validation_start_date"] != _source_date(source, training_rows):
                raise ValueError(f"{name} fold {expected_fold} training-validation adjacency mismatch")
            if fold["validation_end_date_inclusive"] != _source_date(source, training_rows + validation_rows - 1):
                raise ValueError(f"{name} fold {expected_fold} validation end/rows mismatch")
            if previous_training_rows is not None and training_rows != previous_training_rows + previous_validation_rows:
                raise ValueError(f"{name} fold {expected_fold} is not an expanding adjacent window")
            previous_training_rows, previous_validation_rows = training_rows, validation_rows
        if not protocol.get("folds"):
            raise ValueError(f"{name} CV protocol has no folds")
        return

    bounds = _date_bounds(protocol, False)[0]
    _, start, end, rows = bounds
    source_dates = source["Date"].dt.date.astype(str).tolist()
    try:
        start_row = source_dates.index(start)
    except ValueError as exc:
        raise ValueError(f"{name} evaluation start is absent from canonical source") from exc
    if end != _source_date(source, start_row + rows - 1):
        raise ValueError(f"{name} evaluation end/rows mismatch canonical source")
    training_rows = int(protocol.get("training_rows", protocol.get("train_rows", 0)))
    if training_rows:
        training_start = protocol.get("training_start_date")
        if training_start is not None and training_start != _source_date(source, 0):
            raise ValueError(f"{name} training start mismatch")
        training_end = protocol.get("training_end_date_inclusive", protocol.get("train_end_date_inclusive"))
        if training_end != _source_date(source, training_rows - 1):
            raise ValueError(f"{name} training end/rows mismatch")
        is_locked_evaluation = protocol.get("locked_test_start_date") == start
        preceding_rows = training_rows + (int(protocol.get("validation_rows", 0)) if is_locked_evaluation else 0)
        if start_row != preceding_rows:
            raise ValueError(f"{name} non-CV split is not adjacent to its preceding partition")


def _date_bounds(protocol: dict[str, Any], cv: bool) -> list[tuple[int | None, str, str, int]]:
    if cv:
        return [(int(f["fold"]), f["validation_start_date"], f["validation_end_date_inclusive"], int(f["validation_rows"])) for f in protocol["folds"]]
    for prefix in ("holdout", "locked_test", "validation"):
        start = protocol.get(f"{prefix}_start_date")
        end = protocol.get(f"{prefix}_end_date_inclusive")
        rows = protocol.get(f"{prefix}_rows")
        if start is not None and rows is not None and end is None and prefix == "holdout":
            # Phase 9 records the canonical holdout start/count and source hash;
            # its end is necessarily the approved dataset's final date.
            end = "2020-09-02"
        if start is not None and end is not None and rows is not None:
            return [(None, start, end, int(rows))]
    raise ValueError("Protocol lacks complete evaluation boundaries")


def _verify_dates_actuals(name: str, predictions: pd.DataFrame, source: pd.DataFrame,
                           protocol: dict[str, Any], cv: bool) -> None:
    if "Date" not in predictions or "actual" not in predictions:
        raise ValueError(f"{name} predictions lack Date or actual")
    try:
        dates = pd.to_datetime(predictions["Date"], format="%Y-%m-%d", errors="raise")
    except Exception as exc:
        raise ValueError(f"{name} contains invalid prediction dates") from exc
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"{name} prediction dates must be unique and chronological")
    bounds = _date_bounds(protocol, cv)
    if len(predictions) != sum(item[3] for item in bounds):
        raise ValueError(f"{name} prediction row count does not match protocol")
    if cv:
        if "fold" not in predictions or predictions["fold"].isna().any():
            raise ValueError(f"{name} CV predictions lack fold assignments")
        numeric_folds = pd.to_numeric(predictions["fold"], errors="coerce")
        if numeric_folds.isna().any() or not np.equal(numeric_folds, np.floor(numeric_folds)).all():
            raise ValueError(f"{name} fold assignments must be integers")
        expected_ids = [item[0] for item in bounds]
        if expected_ids != list(range(1, len(bounds) + 1)) or list(dict.fromkeys(numeric_folds.astype(int))) != expected_ids:
            raise ValueError(f"{name} fold assignments are malformed or reordered")
        previous_end = None
        for fold_id, start, end, rows in bounds:
            fold = predictions.loc[numeric_folds.astype(int) == fold_id]
            fold_dates = pd.to_datetime(fold["Date"], format="%Y-%m-%d", errors="raise")
            if len(fold) != rows or fold_dates.iloc[0].date().isoformat() != start or fold_dates.iloc[-1].date().isoformat() != end:
                raise ValueError(f"{name} fold {fold_id} boundaries or row count mismatch")
            if previous_end is not None and fold_dates.iloc[0] <= previous_end:
                raise ValueError(f"{name} folds overlap or are not chronological")
            previous_end = fold_dates.iloc[-1]
    else:
        _, start, end, rows = bounds[0]
        if len(predictions) != rows or dates.iloc[0].date().isoformat() != start or dates.iloc[-1].date().isoformat() != end:
            raise ValueError(f"{name} evaluation boundaries mismatch")

    canonical = source.set_index("Date")["Consumption"]
    expected_actual = canonical.reindex(dates)
    if expected_actual.isna().any() or not np.allclose(predictions["actual"].to_numpy(float), expected_actual.to_numpy(float), rtol=0, atol=1e-9):
        raise ValueError(f"{name} actual values do not match canonical Consumption by date")


def _verify_evaluation(name: str, spec: EvaluationSpec, source: pd.DataFrame) -> dict[str, Any]:
    verified_hashes = _verify_hashes(name, spec)
    predictions = pd.read_csv(spec.predictions)
    reported = pd.read_csv(spec.metrics).set_index("model")
    protocol = json.loads(spec.protocol.read_text(encoding="utf-8"))
    _verify_protocol_splits(name, protocol, source, spec.cv)
    _verify_dates_actuals(name, predictions, source, protocol, spec.cv)
    models = [c for c in predictions if c not in {"Date", "actual", "fold"}]
    if not models or not np.isfinite(predictions[["actual", *models]].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} predictions are empty or non-finite")

    fold_metrics_path = spec.predictions.parent / "fold_metrics.csv"
    if name == "phase8":
        if spec.manifest is None:
            raise ValueError("phase8 requires its artifact hash manifest")
        manifest = json.loads(spec.manifest.read_text(encoding="utf-8"))
        if _sha256(fold_metrics_path) != manifest.get("fold_metrics.csv"):
            raise ValueError("phase8 hash mismatch for fold_metrics.csv")
    fold_reported = pd.read_csv(fold_metrics_path).set_index(["fold", "model"]) if name == "phase8" else None
    winners = {int(f["fold"]): f["selected_tuned_model"] for f in protocol.get("folds", [])} if name == "phase8" else {}
    for model in models:
        if spec.cv:
            fold_values = []
            for fold_id, fold in predictions.groupby("fold", sort=True):
                values = _metrics(fold["actual"], fold[model])
                fold_values.append(values)
                if model == "selected_model":
                    winner = winners.get(int(fold_id))
                    if winner is None or fold_reported is None or (int(fold_id), winner) not in fold_reported.index:
                        raise ValueError(f"{name} lacks preserved winner metrics for fold {fold_id}")
                    for metric, value in values.items():
                        _assert_close(value, float(fold_reported.loc[(int(fold_id), winner), metric]), f"{name}/selected_model/fold_{fold_id}/{metric}")
            if model == "selected_model":
                continue
            if model not in reported.index:
                raise ValueError(f"{name} metrics lack model {model}")
            for metric in ("mae", "mse", "rmse", "r2"):
                values = np.array([item[metric] for item in fold_values])
                _assert_close(float(values.mean()), float(reported.loc[model, f"{metric}_mean"]), f"{name}/{model}/{metric}_mean")
                _assert_close(float(values.std(ddof=1)), float(reported.loc[model, f"{metric}_std"]), f"{name}/{model}/{metric}_std")
        else:
            calculated = _metrics(predictions["actual"], predictions[model])
            report_model = "ridge" if name == "phase7" and model == "selected_model" else model
            if report_model not in reported.index:
                raise ValueError(f"{name} metrics lack model {report_model}")
            for metric, value in calculated.items():
                _assert_close(value, float(reported.loc[report_model, metric]), f"{name}/{model}/{metric}")
    return {"rows": len(predictions), "models": models, "hashes": verified_hashes,
            "start_date": predictions["Date"].iloc[0], "end_date": predictions["Date"].iloc[-1]}


def run_audit(
    dataset: Path | None = None,
    approved_dataset_sha256: str | None = None,
    evaluations: dict[str, EvaluationSpec] | None = None,
    expected_rows: int = 3800,
    checkpoint_reader: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """Audit an explicit artifact set; defaults remain the approved London set."""
    dataset = DATASET if dataset is None else dataset
    approved_dataset_sha256 = (
        APPROVED_DATASET_SHA256
        if approved_dataset_sha256 is None
        else approved_dataset_sha256
    )
    evaluations = EVALUATIONS if evaluations is None else evaluations
    checkpoint_reader = (
        _read_persisted_checkpoint if checkpoint_reader is None else checkpoint_reader
    )
    checkpoints = []
    for number in range(12):
        checkpoint = checkpoint_reader(number)
        checkpoints.append({"phase": checkpoint.phase_name, "validated_at_utc": checkpoint.validated_at_utc,
                            "test_returncode": checkpoint.test_evidence.returncode})

    digest = _sha256(dataset)
    if digest != approved_dataset_sha256:
        raise ValueError(f"Canonical dataset hash mismatch: {digest} != {approved_dataset_sha256}")
    data = pd.read_csv(dataset)
    if list(data.columns) != ["Date", "Consumption"]:
        raise ValueError("Unexpected source dataset columns")
    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d", errors="raise")
    if len(data) != expected_rows or data.isna().any().any() or data.duplicated().any():
        raise ValueError("Source dataset shape, null, or duplicate invariant failed")
    if not data["Date"].is_monotonic_increasing or data["Date"].duplicated().any():
        raise ValueError("Source dates are not unique chronological observations")

    evaluation_results = {
        name: _verify_evaluation(name, spec, data) for name, spec in evaluations.items()
    }
    try:
        dataset_display = str(dataset.relative_to(ROOT))
    except ValueError:
        dataset_display = str(dataset.resolve())
    return {"phase": "Phase 12", "scope": "pre-validator full validation audit",
            "independent_validator_invoked": False, "generated_at_utc": datetime.now(UTC).isoformat(),
            "status": "AUDIT_CHECKS_PASSED", "checkpoint_chain": checkpoints,
            "dataset": {"path": dataset_display, "rows": len(data),
                        "start_date": data["Date"].iloc[0].date().isoformat(),
                        "end_date": data["Date"].iloc[-1].date().isoformat(), "sha256": digest,
                        "approved_sha256": approved_dataset_sha256},
            "evaluations": evaluation_results,
            "limitations": ["This report is implementation evidence, not an independent validator verdict.",
                            "Phase 13 documentation and final artifacts are outside this audit's scope."]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{report['status']}: {args.output}")


if __name__ == "__main__":
    main()
