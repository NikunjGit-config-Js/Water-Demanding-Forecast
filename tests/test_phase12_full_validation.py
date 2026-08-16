import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import experiments.phase12_full_validation as audit


def test_full_validation_audit_recomputes_preserved_results() -> None:
    report = audit.run_audit()
    assert report["status"] == "AUDIT_CHECKS_PASSED"
    assert report["dataset"]["sha256"] == report["dataset"]["approved_sha256"]
    assert [item["phase"] for item in report["checkpoint_chain"]] == [f"Phase {n}" for n in range(12)]
    assert set(report["evaluations"]) == {"phase5", "phase6", "phase7", "phase8", "phase9", "phase10"}
    assert all("authority" in item["hashes"] for item in report["evaluations"].values())


def test_full_validation_report_is_json_serializable(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    destination.write_text(json.dumps(audit.run_audit()), encoding="utf-8")
    assert json.loads(destination.read_text(encoding="utf-8"))["phase"] == "Phase 12"


def _isolated_phase8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    source_spec = audit.EVALUATIONS["phase8"]
    artifact_dir = tmp_path / "phase8"
    artifact_dir.mkdir()
    for path in (source_spec.predictions, source_spec.metrics, source_spec.protocol,
                 source_spec.manifest, source_spec.predictions.parent / "fold_metrics.csv"):
        assert path is not None
        shutil.copy2(path, artifact_dir / path.name)
    dataset = tmp_path / "dataset.csv"
    shutil.copy2(audit.DATASET, dataset)
    spec = replace(source_spec, predictions=artifact_dir / source_spec.predictions.name,
                   metrics=artifact_dir / source_spec.metrics.name,
                   protocol=artifact_dir / source_spec.protocol.name,
                   manifest=artifact_dir / "artifact_hashes.json")
    monkeypatch.setattr(audit, "DATASET", dataset)
    monkeypatch.setattr(audit, "EVALUATIONS", {"phase8": spec})
    return dataset, artifact_dir


def _rehash_prediction(directory: Path) -> None:
    manifest_path = directory / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["predictions.csv"] = audit._sha256(directory / "predictions.csv")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _trust_test_manifest(directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = audit.EVALUATIONS["phase8"]
    monkeypatch.setattr(audit, "EVALUATIONS", {
        "phase8": replace(spec, approved_manifest_sha256=audit._sha256(directory / "artifact_hashes.json"))
    })


def test_audit_fails_on_dataset_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, _ = _isolated_phase8(tmp_path, monkeypatch)
    frame = pd.read_csv(dataset)
    frame.loc[0, "Consumption"] += 1
    frame.to_csv(dataset, index=False)
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        audit.run_audit()


@pytest.mark.parametrize("filename", ["predictions.csv", "metrics_summary.csv"])
def test_audit_fails_on_prediction_or_metric_tampering(filename: str, tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    _, directory = _isolated_phase8(tmp_path, monkeypatch)
    path = directory / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.run_audit()


def test_audit_fails_on_manifest_hash_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, directory = _isolated_phase8(tmp_path, monkeypatch)
    manifest_path = directory / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["predictions.csv"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.run_audit()


def test_audit_fails_on_incorrect_actual(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, directory = _isolated_phase8(tmp_path, monkeypatch)
    path = directory / "predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "actual"] += 1
    frame.to_csv(path, index=False)
    _rehash_prediction(directory)
    _trust_test_manifest(directory, monkeypatch)
    with pytest.raises(ValueError, match="actual values"):
        audit.run_audit()


@pytest.mark.parametrize("mutation", ["duplicate", "reordered", "invalid"])
def test_audit_fails_on_invalid_prediction_dates(mutation: str, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    _, directory = _isolated_phase8(tmp_path, monkeypatch)
    path = directory / "predictions.csv"
    frame = pd.read_csv(path)
    if mutation == "duplicate":
        frame.loc[1, "Date"] = frame.loc[0, "Date"]
    elif mutation == "reordered":
        frame.loc[[0, 1], "Date"] = frame.loc[[1, 0], "Date"].to_numpy()
    else:
        frame.loc[0, "Date"] = "not-a-date"
    frame.to_csv(path, index=False)
    _rehash_prediction(directory)
    _trust_test_manifest(directory, monkeypatch)
    with pytest.raises(ValueError, match="dates"):
        audit.run_audit()


def test_audit_fails_on_malformed_fold_assignments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, directory = _isolated_phase8(tmp_path, monkeypatch)
    path = directory / "predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "fold"] = 2
    frame.to_csv(path, index=False)
    _rehash_prediction(directory)
    _trust_test_manifest(directory, monkeypatch)
    with pytest.raises(ValueError, match="fold .*boundaries|fold assignments"):
        audit.run_audit()


def test_phase8_selected_model_is_checked_against_fold_winner(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    _, directory = _isolated_phase8(tmp_path, monkeypatch)
    path = directory / "predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "selected_model"] += 1
    frame.to_csv(path, index=False)
    _rehash_prediction(directory)
    _trust_test_manifest(directory, monkeypatch)
    with pytest.raises(ValueError, match="selected_model/fold_1"):
        audit.run_audit()


def test_audit_fails_when_fold_metrics_manifest_does_not_match(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    _, directory = _isolated_phase8(tmp_path, monkeypatch)
    path = directory / "fold_metrics.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="fold_metrics"):
        audit.run_audit()


def test_audit_fails_on_protocol_report_tampering(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    _, directory = _isolated_phase8(tmp_path, monkeypatch)
    path = directory / "cv_selection_report.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["folds"][0]["training_rows"] += 1
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.run_audit()


def test_audit_fails_on_coordinated_manifest_and_artifact_tampering(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, directory = _isolated_phase8(tmp_path, monkeypatch)
    path = directory / "predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "selected_model"] += 1
    frame.to_csv(path, index=False)
    _rehash_prediction(directory)
    with pytest.raises(ValueError, match="approved manifest hash mismatch"):
        audit.run_audit()


def test_audit_fails_on_malformed_cv_training_boundaries(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, directory = _isolated_phase8(tmp_path, monkeypatch)
    protocol_path = directory / "cv_selection_report.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["folds"][1]["training_rows"] += 1
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    manifest_path = directory / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cv_selection_report.json"] = audit._sha256(protocol_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _trust_test_manifest(directory, monkeypatch)
    with pytest.raises(ValueError, match="training end/rows|training-validation adjacency|expanding"):
        audit.run_audit()


def test_audit_accepts_city_scoped_dataset_and_checkpoint_reader(tmp_path: Path) -> None:
    dataset = tmp_path / "canonical.csv"
    pd.DataFrame(
        {"Date": ["2024-01-01", "2024-01-02"], "Consumption": [1.0, 2.0]}
    ).to_csv(dataset, index=False)

    report = audit.run_audit(
        dataset=dataset,
        approved_dataset_sha256=audit._sha256(dataset),
        evaluations={},
        expected_rows=2,
        checkpoint_reader=lambda number: SimpleNamespace(
            phase_name=f"Phase {number}",
            validated_at_utc="2026-01-01T00:00:00+00:00",
            test_evidence=SimpleNamespace(returncode=0),
        ),
    )

    assert report["dataset"]["path"] == str(dataset.resolve())
    assert len(report["checkpoint_chain"]) == 12
