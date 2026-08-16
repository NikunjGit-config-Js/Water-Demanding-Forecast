import json
from pathlib import Path

import pytest

from experiments.phase13_finalize import FINAL_EVIDENCE, build_manifest


REPOSITORY_ROOT = Path(__file__).parents[1]


def write_checkpoint(root: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "schema_version": 2,
        "phase_number": 0,
        "phase_name": "Phase 0",
        "validation_verdict": "PASS",
        "validated_at_utc": "2026-08-15T00:00:00+00:00",
        "test_evidence": {
            "command": ["python", "-m", "pytest", "-q"],
            "returncode": 0,
        },
        "validation_report": "PASS\nIndependent validation evidence.",
    }
    payload.update(overrides)
    checkpoint_dir = root / "orchestration/state/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "phase_0_passed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase13_manifest_indexes_approved_chain_and_final_evidence() -> None:
    manifest = build_manifest(REPOSITORY_ROOT)

    checkpoints = manifest["prerequisite_checkpoints"]
    assert [item["phase"] for item in checkpoints] == [f"Phase {i}" for i in range(13)]
    assert all(len(item["sha256"]) == 64 for item in checkpoints)
    assert [item["path"] for item in manifest["files"]] == list(FINAL_EVIDENCE)
    assert manifest["validator_invoked"] is False


def test_phase13_manifest_fails_closed_for_missing_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "orchestration/state/checkpoints").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="missing prerequisite checkpoint"):
        build_manifest(tmp_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"phase_number": 1},
        {"phase_name": "Phase 1"},
    ],
)
def test_phase13_manifest_rejects_mismatched_checkpoint_identity(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    write_checkpoint(tmp_path, **overrides)

    with pytest.raises(ValueError, match="does not match Phase 0"):
        build_manifest(tmp_path)


@pytest.mark.parametrize(
    "test_evidence",
    [
        {"command": ["python", "-m", "pytest", "-q"], "returncode": 1},
        {"command": ["python", "-m", "pytest", "-q"], "returncode": False},
        {"command": ["python", "-m", "pytest", "-q"], "returncode": "0"},
        {"command": ["python", "-m", "pytest", "-q"], "returncode": 0.0},
        {"command": ["pytest", "-q"], "returncode": 0},
        {"command": ["python", "-m", "pytest", "-q", "extra"], "returncode": 0},
        {"command": ["python", "-m", "pytest", "-q"]},
        {
            "command": ["python", "-m", "pytest", "-q"],
            "returncode": 0,
            "unexpected": True,
        },
        None,
        "tests passed",
    ],
)
def test_phase13_manifest_rejects_unsafe_or_incomplete_test_evidence(
    tmp_path: Path, test_evidence: object
) -> None:
    write_checkpoint(tmp_path, test_evidence=test_evidence)

    with pytest.raises(ValueError, match="not a schema-v2 PASS"):
        build_manifest(tmp_path)


@pytest.mark.parametrize(
    "timestamp",
    [
        "not-a-timestamp",
        "2026-08-15T00:00:00",
        "2026-08-15T01:00:00+01:00",
    ],
)
def test_phase13_manifest_rejects_invalid_or_non_utc_timestamp(
    tmp_path: Path, timestamp: str
) -> None:
    write_checkpoint(tmp_path, validated_at_utc=timestamp)

    with pytest.raises(ValueError, match="not a schema-v2 PASS"):
        build_manifest(tmp_path)


def test_phase13_manifest_rejects_unknown_checkpoint_field(tmp_path: Path) -> None:
    write_checkpoint(tmp_path, unexpected="field")

    with pytest.raises(ValueError, match="not a schema-v2 PASS"):
        build_manifest(tmp_path)


@pytest.mark.parametrize(
    "missing_field",
    [
        "schema_version",
        "phase_number",
        "phase_name",
        "validation_verdict",
        "validated_at_utc",
        "test_evidence",
        "validation_report",
    ],
)
def test_phase13_manifest_rejects_missing_checkpoint_field(
    tmp_path: Path, missing_field: str
) -> None:
    path = write_checkpoint(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload[missing_field]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not a schema-v2 PASS"):
        build_manifest(tmp_path)


@pytest.mark.parametrize(
    ("verdict", "report"),
    [
        ("PASS", "FAIL\nREQUIRED_CORRECTIONS:\nFix it."),
        ("FAIL", "PASS\nContradictory verdict."),
        ("PASS", "pass\nWrong case."),
        ("PASS", "PASSING\nNot an exact verdict."),
        ("PASS", ""),
    ],
)
def test_phase13_manifest_rejects_contradictory_or_malformed_pass_report(
    tmp_path: Path, verdict: str, report: str
) -> None:
    write_checkpoint(
        tmp_path, validation_verdict=verdict, validation_report=report
    )

    with pytest.raises(ValueError, match="not a schema-v2 PASS"):
        build_manifest(tmp_path)
