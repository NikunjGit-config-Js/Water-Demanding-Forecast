from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

import pytest

from orchestration.agents.validator import ValidationResult
from orchestration import flow as flow_module
from orchestration.flow import (
    OrchestrationState,
    PhaseCheckpoint,
    WaterForecastFlow,
    specialist_for_phase,
)
from orchestration.tools.test_tool import CheckpointTestEvidence
from orchestration.tools.git_tool import GitSnapshot
from orchestration.tools.test_tool import CommandResult, TestReport as OrchestrationTestReport


UNCHANGED = GitSnapshot(status="", diff="")


def passing_tests() -> OrchestrationTestReport:
    return OrchestrationTestReport(
        (CommandResult(("python", "-m", "pytest", "-q"), 0, "passed", ""),)
    )


def build_flow(
    *,
    phases: list[str] | None = None,
    max_attempts: int = 3,
    implementation: Callable[[str, str], str] | None = None,
    validator: Callable[[str, str], ValidationResult] | None = None,
    test_executor: Callable[[], OrchestrationTestReport] = passing_tests,
    checkpoints: list[object] | None = None,
    snapshot_reader: Callable[[], GitSnapshot] = lambda: UNCHANGED,
) -> WaterForecastFlow:
    saved = checkpoints if checkpoints is not None else []
    return WaterForecastFlow(
        phases=phases if phases is not None else ["Phase 0"],
        max_attempts=max_attempts,
        implementation_executor=implementation or (lambda prompt, name: "implemented"),
        test_executor=test_executor,
        validation_executor=validator or (
            lambda phase, summary: ValidationResult("PASS", "PASS\nEvidence")
        ),
        checkpoint_writer=lambda state: saved.append(state),
        snapshot_reader=snapshot_reader,
    )


def test_pass_routes_to_checkpoint_and_completion() -> None:
    checkpoints: list[object] = []
    flow = build_flow(checkpoints=checkpoints)

    flow.orchestrate()

    assert flow.state.status == "completed"
    assert flow.state.completed_phases == ["Phase 0"]
    assert flow.state.specialist_role == "supervisor"
    assert flow.state.attempt == 1
    assert len(checkpoints) == 1


def test_fail_repair_pass_includes_exact_validator_report() -> None:
    prompts: list[str] = []
    failure = "FAIL\nDetails exactly.\nREQUIRED_CORRECTIONS:\nFix exact item A."
    verdicts = iter(
        [ValidationResult("FAIL", failure), ValidationResult("PASS", "PASS\nFixed")]
    )

    flow = build_flow(
        implementation=lambda prompt, name: prompts.append(prompt) or "summary",
        validator=lambda phase, summary: next(verdicts),
    )
    flow.orchestrate()

    assert flow.state.status == "completed"
    assert flow.state.attempt == 2
    assert len(prompts) == 2
    assert failure in prompts[1]


def test_resumed_flow_uses_exact_repair_report_and_remaining_budget() -> None:
    prompts: list[str] = []
    exact = "FAIL\nREQUIRED_CORRECTIONS:\nPreserve this exact correction."
    flow = WaterForecastFlow(
        phases=["Phase 0"],
        max_attempts=3,
        starting_attempt=1,
        repair_report=exact,
        implementation_executor=lambda prompt, name: prompts.append(prompt) or "fixed",
        test_executor=passing_tests,
        validation_executor=lambda phase, summary: ValidationResult("PASS", "PASS\nFixed"),
        checkpoint_writer=lambda state: None,
        snapshot_reader=lambda: UNCHANGED,
    )
    flow.orchestrate()
    assert flow.state.status == "completed"
    assert flow.state.attempt == 2
    assert len(prompts) == 1
    assert exact in prompts[0]


def test_retry_limit_fails_without_checkpoint() -> None:
    checkpoints: list[object] = []
    calls = 0

    def fail(phase: str, summary: str) -> ValidationResult:
        nonlocal calls
        calls += 1
        return ValidationResult("FAIL", "FAIL\nREQUIRED_CORRECTIONS:\nStill broken")

    flow = build_flow(max_attempts=2, validator=fail, checkpoints=checkpoints)
    flow.orchestrate()

    assert flow.state.status == "failed"
    assert calls == 2
    assert flow.state.completed_phases == []
    assert checkpoints == []
    assert "Retry limit exhausted" in flow.state.failure_reason


@pytest.mark.parametrize(
    ("verdict", "report"),
    [("MAYBE", "MAYBE"), ("FAIL", "FAIL\nNo corrections")],
)
def test_malformed_validator_output_fails_closed(verdict: str, report: str) -> None:
    flow = build_flow(validator=lambda phase, summary: ValidationResult(verdict, report))
    flow.orchestrate()
    assert flow.state.status == "failed"
    assert flow.state.completed_phases == []
    assert "Validator" in flow.state.failure_reason or "verdict" in flow.state.failure_reason


@pytest.mark.parametrize(
    ("phase", "role"),
    [
        ("Phase 0", "supervisor"),
        ("Phase 1", "data_agent"),
        ("Phase 2", "feature_agent"),
        ("Phase 4", "ml_agent"),
        ("Phase 9", "timeseries_agent"),
        ("Phase 10", "transformer_agent"),
        ("Phase 11", "ui_agent"),
        ("Phase 12", "supervisor"),
    ],
)
def test_specialist_role_routing(phase: str, role: str) -> None:
    assert specialist_for_phase(phase) == role


def test_unknown_phase_fails_closed() -> None:
    flow = build_flow(phases=["invented phase"])
    flow.orchestrate()
    assert flow.state.status == "failed"
    assert flow.state.completed_phases == []


def test_validator_repository_change_fails_closed() -> None:
    snapshots = iter([UNCHANGED, GitSnapshot(status=" M file.py", diff="changed")])
    flow = build_flow(snapshot_reader=lambda: next(snapshots))
    flow.orchestrate()
    assert flow.state.status == "failed"
    assert "modified repository state" in flow.state.failure_reason


def test_test_failure_repairs_without_calling_validator() -> None:
    reports = iter(
        [
            OrchestrationTestReport(
                (CommandResult(("python", "-m", "pytest", "-q"), 1, "", "failed"),)
            ),
            passing_tests(),
        ]
    )
    validation_calls = 0

    def validate(phase: str, summary: str) -> ValidationResult:
        nonlocal validation_calls
        validation_calls += 1
        return ValidationResult("PASS", "PASS")

    flow = build_flow(test_executor=lambda: next(reports), validator=validate)
    flow.orchestrate()
    assert flow.state.status == "completed"
    assert flow.state.attempt == 2
    assert validation_calls == 1


def test_permission_required_stops_before_tests_and_validation() -> None:
    test_calls = 0
    validation_calls = 0

    def tests() -> OrchestrationTestReport:
        nonlocal test_calls
        test_calls += 1
        return passing_tests()

    def validate(phase: str, summary: str) -> ValidationResult:
        nonlocal validation_calls
        validation_calls += 1
        return ValidationResult("PASS", "PASS")

    flow = build_flow(
        implementation=lambda prompt, name: (
            "PERMISSION_REQUIRED\nApproval needed for external data scraping."
        ),
        test_executor=tests,
        validator=validate,
    )
    flow.orchestrate()
    assert flow.state.status == "failed"
    assert test_calls == 0
    assert validation_calls == 0
    assert flow.state.completed_phases == []


def test_no_implicit_phase_zero() -> None:
    flow = build_flow(phases=[])
    flow.state.phases = []
    flow.orchestrate()
    assert flow.state.status == "failed"
    assert "refusing to start Phase 0" in flow.state.failure_reason


def write_checkpoint(directory: Path, number: int, **overrides: object) -> None:
    payload: dict[str, object] = {
        "schema_version": 2,
        "phase_number": number,
        "phase_name": f"Phase {number}",
        "validation_verdict": "PASS",
        "validated_at_utc": "2026-08-15T00:00:00+00:00",
        "test_evidence": {
            "command": ["python", "-m", "pytest", "-q"],
            "returncode": 0,
        },
        "validation_report": "PASS\nevidence",
    }
    payload.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"phase_{number}_passed.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_valid_sequential_execution_uses_same_run_prerequisites() -> None:
    flow = build_flow(phases=["Phase 0", "Phase 1", "Phase 2"])
    flow.orchestrate()
    assert flow.state.status == "completed"
    assert flow.state.completed_phases == ["Phase 0", "Phase 1", "Phase 2"]


def test_resume_uses_valid_persisted_pass_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    for number in range(3):
        write_checkpoint(checkpoint_dir, number)
    monkeypatch.setattr(flow_module, "CHECKPOINT_DIR", checkpoint_dir)
    flow = build_flow(phases=["Phase 3", "Phase 4"])
    flow.orchestrate()
    assert flow.state.status == "completed"
    assert flow.state.completed_phases == ["Phase 3", "Phase 4"]


@pytest.mark.parametrize(
    ("phases", "message"),
    [
        (["Phase 0", "Phase 2"], "chronological and consecutive"),
        (["Phase 1", "Phase 0"], "chronological and consecutive"),
        (["Phase 0", "Phase 0"], "Duplicate"),
    ],
)
def test_rejects_skipped_reordered_and_duplicate_phases(
    phases: list[str], message: str
) -> None:
    flow = build_flow(phases=phases)
    flow.orchestrate()
    assert flow.state.status == "failed"
    assert message in flow.state.failure_reason
    assert flow.state.completed_phases == []


def test_rejects_missing_prerequisite_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(flow_module, "CHECKPOINT_DIR", tmp_path / "missing")
    flow = build_flow(phases=["Phase 1"])
    flow.orchestrate()
    assert flow.state.status == "failed"
    assert "missing or malformed" in flow.state.failure_reason


@pytest.mark.parametrize(
    "payload_override",
    [
        {"validation_verdict": "FAIL"},
        {"phase_number": 9},
        {"phase_name": "Phase 9"},
        {"unexpected": "field"},
    ],
)
def test_rejects_malformed_or_non_pass_prerequisite_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload_override: dict[str, object],
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    write_checkpoint(checkpoint_dir, 0, **payload_override)
    monkeypatch.setattr(flow_module, "CHECKPOINT_DIR", checkpoint_dir)
    flow = build_flow(phases=["Phase 1"])
    flow.orchestrate()
    assert flow.state.status == "failed"
    assert "checkpoint" in flow.state.failure_reason


@pytest.mark.parametrize(
    "payload_override",
    [
        {"validation_report": "FAIL\nREQUIRED_CORRECTIONS:\nBroken"},
        {"validation_report": "pass\nevidence"},
        {"validation_report": "malformed text"},
        {"validation_report": ""},
        {"test_evidence": {"command": ["python", "-m", "pytest", "-q"], "returncode": 1}},
        {"test_evidence": {"command": ["python", "-c", "print('x')"], "returncode": 0}},
        {"test_evidence": {"command": ["python", "arbitrary_script.py"], "returncode": 0}},
        {"test_evidence": {"command": ["pytest", "-q"], "returncode": 0}},
        {"test_evidence": {"command": ["python", "-m", "pytest", "-q", "extra"], "returncode": 0}},
        {"test_evidence": None},
        {"test_evidence": "tests passed"},
        {"test_evidence": {"command": ["python", "-m", "pytest", "-q"]}},
        {"test_evidence": {"command": ["python", "-m", "pytest", "-q"], "returncode": 0, "unsafe": True}},
    ],
)
def test_rejects_unsafe_persisted_checkpoint_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload_override: dict[str, object],
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    write_checkpoint(checkpoint_dir, 0, **payload_override)
    monkeypatch.setattr(flow_module, "CHECKPOINT_DIR", checkpoint_dir)
    flow = build_flow(phases=["Phase 1"])
    flow.orchestrate()
    assert flow.state.status == "failed"
    assert "checkpoint" in flow.state.failure_reason


def test_rejects_legacy_free_form_test_report_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    write_checkpoint(
        checkpoint_dir,
        0,
        schema_version=1,
        test_report="tests passed",
    )
    monkeypatch.setattr(flow_module, "CHECKPOINT_DIR", checkpoint_dir)
    flow = build_flow(phases=["Phase 1"])
    flow.orchestrate()
    assert flow.state.status == "failed"
    assert "checkpoint" in flow.state.failure_reason


def test_rejects_checkpoint_with_missing_structured_test_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    write_checkpoint(checkpoint_dir, 0)
    path = checkpoint_dir / "phase_0_passed.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["test_evidence"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(flow_module, "CHECKPOINT_DIR", checkpoint_dir)
    flow = build_flow(phases=["Phase 1"])
    flow.orchestrate()
    assert flow.state.status == "failed"
    assert "checkpoint" in flow.state.failure_reason


def test_default_checkpoint_writer_persists_only_strict_structured_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setattr(flow_module, "CHECKPOINT_DIR", checkpoint_dir)
    state = OrchestrationState(
        current_phase="Phase 0",
        validation_verdict="PASS",
        validation_report="PASS\nEvidence",
        test_report="free-form diagnostic output is not prerequisite proof",
        test_evidence=CheckpointTestEvidence(
            command=["python", "-m", "pytest", "-q"], returncode=0
        ),
    )
    flow_module._default_checkpoint_writer(state)
    payload = json.loads(
        (checkpoint_dir / "phase_0_passed.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 2
    assert payload["test_evidence"] == {
        "command": ["python", "-m", "pytest", "-q"],
        "returncode": 0,
    }
    assert "test_report" not in payload


def test_checkpoint_filename_cannot_override_mismatched_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    write_checkpoint(checkpoint_dir, 0, phase_number=1, phase_name="Phase 1")
    monkeypatch.setattr(flow_module, "CHECKPOINT_DIR", checkpoint_dir)
    flow = build_flow(phases=["Phase 1"])
    flow.orchestrate()
    assert flow.state.status == "failed"
    assert "does not match" in flow.state.failure_reason


def test_default_codex_executor_selects_workspace_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_codex(prompt: str, **kwargs: object) -> str:
        captured.update(kwargs)
        return "summary"

    monkeypatch.setattr(flow_module, "run_codex", fake_codex)
    assert flow_module._default_implementation_executor("prompt", "phase.txt") == "summary"
    assert captured["sandbox"] == "workspace-write"


@pytest.mark.parametrize(
    ("report", "accepted"),
    [
        ("PASS\nEvidence", True),
        ("\nPASS\nEvidence", False),
        (" PASS\nEvidence", False),
        ("PASS \nEvidence", False),
        ("\tPASS\nEvidence", False),
        ("pass\nEvidence", False),
        ("Pass\nEvidence", False),
    ],
)
def test_live_and_persisted_pass_report_validation_are_identical(
    report: str, accepted: bool
) -> None:
    live_accepted = True
    try:
        flow_module.validate_report("PASS", report)
    except flow_module.ValidatorError:
        live_accepted = False

    persisted_accepted = True
    try:
        PhaseCheckpoint(
            phase_number=0,
            phase_name="Phase 0",
            validation_verdict="PASS",
            validated_at_utc="2026-08-15T00:00:00+00:00",
            test_evidence=CheckpointTestEvidence(
                command=["python", "-m", "pytest", "-q"], returncode=0
            ),
            validation_report=report,
        )
    except ValueError:
        persisted_accepted = False

    assert live_accepted is accepted
    assert persisted_accepted is accepted
