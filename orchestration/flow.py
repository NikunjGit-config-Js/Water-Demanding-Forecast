from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import yaml
from crewai.flow.flow import Flow, FlowState, start
from pydantic import BaseModel, ConfigDict, Field, field_validator

from orchestration.agents.validator import (
    ValidationResult,
    ValidatorError,
    validate_phase,
    validate_report,
)
from orchestration.tools.codex_tool import run_codex
from orchestration.tools.git_tool import GitSnapshot, capture_git_snapshot
from orchestration.tools.test_tool import (
    CheckpointTestEvidence,
    TestReport,
    run_test_commands,
)
from orchestration.context import RunContext


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "orchestration" / "config"
CHECKPOINT_DIR = PROJECT_ROOT / "orchestration" / "state" / "checkpoints"

ROLE_BY_PHASE: dict[str, str] = {
    "phase_0": "supervisor",
    "phase_1": "data_agent",
    "phase_2": "feature_agent",
    "phase_3": "feature_agent",
    "phase_4": "ml_agent",
    "phase_5": "ml_agent",
    "phase_6": "ml_agent",
    "phase_7": "ml_agent",
    "phase_8": "ml_agent",
    "phase_9": "timeseries_agent",
    "phase_10": "transformer_agent",
    "phase_11": "ui_agent",
    "phase_12": "supervisor",
    "phase_13": "supervisor",
}
PHASE_NUMBERS = tuple(range(14))


class PhaseCheckpoint(BaseModel):
    """Minimal, strict persisted proof that a phase passed validation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[2] = 2
    phase_number: int
    phase_name: str
    validation_verdict: Literal["PASS"]
    validated_at_utc: str
    test_evidence: CheckpointTestEvidence
    validation_report: str = Field(min_length=1)

    @field_validator("validation_report")
    @classmethod
    def report_must_begin_with_exact_pass(cls, value: str) -> str:
        try:
            validate_report("PASS", value)
        except ValidatorError as exc:
            raise ValueError(str(exc)) from exc
        return value

    @field_validator("validated_at_utc")
    @classmethod
    def timestamp_must_be_utc(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("validated_at_utc must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
            raise ValueError("validated_at_utc must include a UTC offset")
        return value


class OrchestrationState(FlowState):
    current_phase: str = ""
    specialist_role: str = ""
    attempt: int = 0
    max_attempts: int = Field(default=3, ge=1)
    implementation_summary: str = ""
    validation_verdict: str = ""
    validation_report: str = ""
    status: Literal["ready", "running", "completed", "failed"] = "ready"
    completed_phases: list[str] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    test_report: str = ""
    test_evidence: CheckpointTestEvidence | None = None
    failure_reason: str = ""


ImplementationExecutor = Callable[[str, str], str]
TestExecutor = Callable[[], TestReport]
ValidationExecutor = Callable[[str, str], ValidationResult]
CheckpointWriter = Callable[[OrchestrationState], None]
SnapshotReader = Callable[[], GitSnapshot]
ProgressReporter = Callable[[str], None]
StateReporter = Callable[[OrchestrationState], None]


def phase_number(phase: str) -> int:
    """Return an approved phase number from a canonical phase label."""
    match = re.fullmatch(r"phase[ _-]?(\d+)", phase.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid phase label: {phase!r}")
    number = int(match.group(1))
    if number not in PHASE_NUMBERS:
        raise ValueError(f"Phase number must be between 0 and 13: {phase!r}")
    return number


def canonical_phase_name(phase: str) -> str:
    return f"Phase {phase_number(phase)}"


def _load_yaml(path: Path) -> dict[str, Any]:
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return content


def load_orchestration_config() -> tuple[dict[str, Any], dict[str, Any]]:
    agents = _load_yaml(CONFIG_DIR / "agents.yaml")
    tasks = _load_yaml(CONFIG_DIR / "tasks.yaml")
    required_agents = {
        "supervisor", "data_agent", "feature_agent", "ml_agent",
        "timeseries_agent", "transformer_agent", "ui_agent", "validator",
    }
    if not required_agents.issubset(agents):
        raise ValueError(f"Missing agent profiles: {sorted(required_agents - agents.keys())}")
    if not {"implementation", "repair", "validation"}.issubset(tasks):
        raise ValueError("tasks.yaml must define implementation, repair, and validation")
    return agents, tasks


def specialist_for_phase(phase: str) -> str:
    key = f"phase_{phase_number(phase)}"
    return ROLE_BY_PHASE[key]


def _default_implementation_executor(prompt: str, output_name: str) -> str:
    return run_codex(
        prompt,
        sandbox="workspace-write",
        output_name=output_name,
        timeout=3600,
    )


def _write_checkpoint(state: OrchestrationState, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    number = phase_number(state.current_phase)
    checkpoint = PhaseCheckpoint(
        phase_number=number,
        phase_name=f"Phase {number}",
        validation_verdict="PASS",
        validated_at_utc=datetime.now(UTC).isoformat(),
        test_evidence=state.test_evidence,
        validation_report=state.validation_report,
    )
    target = checkpoint_dir / f"phase_{number}_passed.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(checkpoint.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _default_checkpoint_writer(state: OrchestrationState) -> None:
    _write_checkpoint(state, CHECKPOINT_DIR)


def _read_persisted_checkpoint(
    number: int, checkpoint_dir: Path | None = None
) -> PhaseCheckpoint:
    path = (checkpoint_dir or CHECKPOINT_DIR) / f"phase_{number}_passed.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = PhaseCheckpoint.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Phase {number} prerequisite checkpoint is missing or malformed: {path}"
        ) from exc
    if checkpoint.phase_number != number or checkpoint.phase_name != f"Phase {number}":
        raise ValueError(
            f"Phase {number} prerequisite checkpoint content does not match its phase."
        )
    return checkpoint


def _validate_requested_phases(phases: Sequence[str]) -> list[int]:
    numbers = [phase_number(phase) for phase in phases]
    if len(numbers) != len(set(numbers)):
        raise ValueError("Duplicate phases are not allowed.")
    if any(current != previous + 1 for previous, current in zip(numbers, numbers[1:])):
        raise ValueError("Phases must be strictly chronological and consecutive.")
    return numbers


class WaterForecastFlow(Flow[OrchestrationState]):
    """Deterministic CrewAI control flow around Codex and a read-only validator."""

    _skip_auto_memory: ClassVar[bool] = True

    def __init__(
        self,
        *,
        phases: Sequence[str] | None = None,
        max_attempts: int = 3,
        implementation_executor: ImplementationExecutor = _default_implementation_executor,
        test_executor: TestExecutor = run_test_commands,
        validation_executor: ValidationExecutor = validate_phase,
        checkpoint_writer: CheckpointWriter = _default_checkpoint_writer,
        snapshot_reader: SnapshotReader = capture_git_snapshot,
        progress_reporter: ProgressReporter | None = None,
        state_reporter: StateReporter | None = None,
        starting_attempt: int = 0,
        repair_report: str = "",
        run_context: RunContext | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if starting_attempt < 0 or starting_attempt > max_attempts:
            raise ValueError("starting_attempt must be between 0 and max_attempts")
        super().__init__(
            initial_state=OrchestrationState(
                phases=list(phases or []), max_attempts=max_attempts
            ),
            tracing=False,
            suppress_flow_events=True,
            max_method_calls=max(20, max_attempts * max(1, len(phases or [])) * 4),
        )
        self._implementation_executor = implementation_executor
        self._test_executor = test_executor
        self._validation_executor = validation_executor
        self.run_context = run_context or RunContext.for_city("london")
        self._checkpoint_dir = (
            self.run_context.checkpoint_root if run_context is not None else CHECKPOINT_DIR
        )
        self._checkpoint_writer = (
            checkpoint_writer
            if checkpoint_writer is not _default_checkpoint_writer
            else lambda state: _write_checkpoint(state, self._checkpoint_dir)
        )
        self._snapshot_reader = snapshot_reader
        self._progress_reporter = progress_reporter or (lambda message: None)
        self._state_reporter = state_reporter or (lambda state: None)
        self._starting_attempt = starting_attempt
        self._repair_report = repair_report
        self._agents, self._tasks = load_orchestration_config()

    def _format_task(self, task_name: str) -> str:
        template = self._tasks[task_name]["description"]
        profile = self._agents[self.state.specialist_role]
        return template.format(
            phase_name=self.state.current_phase,
            specialist_role=self.state.specialist_role,
            attempt=self.state.attempt,
            max_attempts=self.state.max_attempts,
            validation_report=self.state.validation_report,
        ) + "\n\n" + self.run_context.prompt_block(
            phase_number(self.state.current_phase)
        ) + "\n\nSPECIALIST PROFILE:\n" + yaml.safe_dump(
            profile, sort_keys=False, allow_unicode=True
        ).strip()

    def _run_validation(self) -> ValidationResult:
        before = self._snapshot_reader()
        result = self._validation_executor(
            self.state.current_phase, self.state.implementation_summary
        )
        after = self._snapshot_reader()
        if before != after:
            raise ValidatorError("Validator modified repository state; validation failed closed.")
        return validate_report(result.verdict, result.report)

    def _fail(self, reason: str) -> OrchestrationState:
        self.state.status = "failed"
        self.state.failure_reason = reason
        return self.state

    def _execute_phase(self) -> bool:
        try:
            self.state.specialist_role = specialist_for_phase(self.state.current_phase)
        except ValueError as exc:
            self._fail(str(exc))
            return False
        self.state.attempt = self._starting_attempt
        self.state.validation_verdict = ""
        self.state.validation_report = self._repair_report
        self.state.test_evidence = None
        for attempt in range(self._starting_attempt + 1, self.state.max_attempts + 1):
            self.state.attempt = attempt
            self._state_reporter(self.state.model_copy(deep=True))
            self._progress_reporter(
                f"implementation attempt {attempt}/{self.state.max_attempts}"
            )
            task_name = "implementation" if attempt == 1 else "repair"
            prompt = self._format_task(task_name)
            output_name = f"{re.sub(r'[^a-zA-Z0-9_.-]+', '_', self.state.current_phase)}_attempt_{attempt}.txt"
            try:
                self.state.implementation_summary = self._implementation_executor(
                    prompt, output_name
                ).strip()
                if (
                    self.state.implementation_summary.splitlines()
                    and self.state.implementation_summary.splitlines()[0].strip()
                    == "PERMISSION_REQUIRED"
                ):
                    self._fail(self.state.implementation_summary)
                    self._state_reporter(self.state.model_copy(deep=True))
                    return False
                tests = self._test_executor()
                self.state.test_report = tests.summary()
                if not tests.passed:
                    self._progress_reporter("tests FAIL; scheduling repair")
                    self.state.validation_verdict = "FAIL"
                    self.state.validation_report = (
                        "FAIL\n\nREQUIRED_CORRECTIONS:\n"
                        "Fix the failing test/check output below before validation.\n\n"
                        + self.state.test_report
                    )
                    self._state_reporter(self.state.model_copy(deep=True))
                    continue
                self._progress_reporter("tests PASS")
                self.state.test_evidence = tests.checkpoint_evidence()
                validation = self._run_validation()
            except Exception as exc:
                self._fail(f"{type(exc).__name__}: {exc}")
                if not self.state.validation_report:
                    self.state.validation_report = (
                        "FAIL\n\nREQUIRED_CORRECTIONS:\n"
                        f"Resolve infrastructure failure: {self.state.failure_reason}"
                    )
                self._state_reporter(self.state.model_copy(deep=True))
                return False

            self.state.validation_verdict = validation.verdict
            self.state.validation_report = validation.report
            self._state_reporter(self.state.model_copy(deep=True))
            self._progress_reporter(f"validator {validation.verdict}")
            if validation.passed:
                self.state.completed_phases.append(self.state.current_phase)
                self._checkpoint_writer(self.state.model_copy(deep=True))
                return True
            self._progress_reporter("validator FAIL; scheduling repair")

        self._fail(
            f"Retry limit exhausted for {self.state.current_phase} "
            f"after {self.state.max_attempts} attempts."
        )
        self._state_reporter(self.state.model_copy(deep=True))
        return False

    @start()
    def orchestrate(self) -> OrchestrationState:
        if not self.state.phases:
            return self._fail("No phases supplied; refusing to start Phase 0 implicitly.")
        try:
            requested_numbers = _validate_requested_phases(self.state.phases)
        except ValueError as exc:
            return self._fail(str(exc))
        self.state.status = "running"
        passed_in_this_run: set[int] = set()
        for phase, number in zip(self.state.phases, requested_numbers):
            for prerequisite in range(number):
                if prerequisite in passed_in_this_run:
                    continue
                try:
                    _read_persisted_checkpoint(prerequisite, self._checkpoint_dir)
                except ValueError as exc:
                    return self._fail(str(exc))
            self.state.current_phase = canonical_phase_name(phase)
            if not self._execute_phase():
                return self.state
            passed_in_this_run.add(number)
        self.state.status = "completed"
        return self.state
