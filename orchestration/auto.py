from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orchestration.flow import (
    CHECKPOINT_DIR,
    PHASE_NUMBERS,
    PROJECT_ROOT,
    WaterForecastFlow,
    _read_persisted_checkpoint,
)
from orchestration.tools.git_tool import (
    GitCheckpointResult,
    GitSafetyError,
    changed_paths,
    create_local_checkpoint,
    require_clean_worktree,
)
from orchestration.tools.test_tool import TestReport, run_test_commands


AUTO_STATE_PATH = PROJECT_ROOT / "orchestration" / "state" / "auto_run.json"
FINAL_SUMMARY_PATH = PROJECT_ROOT / "orchestration" / "state" / "auto_final_summary.json"
PHASE_SHORT_NAMES = (
    "baseline reproduction",
    "data analysis and EDA",
    "feature engineering",
    "feature selection",
    "traditional ML",
    "chronological holdout",
    "time-aware cross-validation",
    "locked-test Optuna",
    "time-series CV Optuna",
    "time-series models",
    "modern forecasting",
    "Streamlit dashboard",
    "full validation",
    "documentation and final artifacts",
)

PROTECTED_PATHS = frozenset(
    {
        "AGENTS.md",
        "PROJECT_SPEC.md",
        "EXPERIMENT_PLAN.md",
        "SUCCESS_CRITERIA.md",
        "VALIDATION_RULES.md",
        "PERMISSIONS.md",
    }
)
SECRET_NAME_RE = re.compile(
    r"(^|/)(\.env($|\.)|.*(?:secret|credential|api[_-]?key|private[_-]?key).*)",
    re.IGNORECASE,
)
class AutoState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    status: Literal["ready", "running", "failed", "permission_required", "completed"]
    active_phase: int | None = None
    active_attempt: int = Field(default=0, ge=0)
    failure_report: str = ""
    attempts_per_phase: dict[str, int] = Field(default_factory=dict)
    git_checkpoint_commits: dict[str, str] = Field(default_factory=dict)
    failure_reason: str = ""
    updated_at_utc: str


FlowFactory = Callable[..., WaterForecastFlow]
Committer = Callable[..., GitCheckpointResult]
Progress = Callable[[str], None]


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _new_state() -> AutoState:
    return AutoState(status="ready", updated_at_utc=_timestamp())


def _load_state(path: Path = AUTO_STATE_PATH) -> AutoState:
    if not path.exists():
        return _new_state()
    try:
        return AutoState.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Auto state is malformed; refusing unsafe resume: {path}") from exc


def _save_state(state: AutoState, path: Path = AUTO_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at_utc = _timestamp()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validated_phase_numbers() -> list[int]:
    """Return the consecutive valid PASS prefix, rejecting gaps and bad records."""
    completed: list[int] = []
    gap_seen = False
    for number in PHASE_NUMBERS:
        path = CHECKPOINT_DIR / f"phase_{number}_passed.json"
        if not path.exists():
            gap_seen = True
            continue
        if gap_seen:
            raise ValueError(
                f"Checkpoint for Phase {number} exists after an incomplete phase; refusing resume."
            )
        _read_persisted_checkpoint(number)
        completed.append(number)
    return completed


def first_incomplete_phase() -> int | None:
    completed = validated_phase_numbers()
    return None if len(completed) == len(PHASE_NUMBERS) else len(completed)


def _permission_request(summary: str) -> str | None:
    if not summary:
        return None
    lines = summary.splitlines()
    if lines[0].strip() != "PERMISSION_REQUIRED":
        return None
    detail = "\n".join(lines[1:]).strip()
    return detail or "The specialist requested an unspecified permission."


def _phase_path_is_expected(number: int, path: str) -> bool:
    """Return whether *path* is explicitly owned by this phase."""
    if path in {"STATUS.md", "DECISIONS.md"}:
        return True
    if path == f"orchestration/state/checkpoints/phase_{number}_passed.json":
        return True
    if path.startswith((f"artifacts/phase{number}/", f"reports/phase{number}/")):
        return True
    patterns = (
        rf"experiments/phase{number}(?:_|\.|$).+",
        rf"tests/test_phase{number}(?:_|\.|$).+",
        rf"src/(?:.*/)?phase{number}_.+",
        rf"results/(?:.*/)?phase{number}(?:_|\.|$).+",
        rf"img/(?:.*/)?phase{number}(?:_|\.|$).+",
        rf"orchestration/state/[Pp]hase_{number}(?:_|\.|$).+",
        rf"orchestration/logs/[Pp]hase_{number}(?:_|\.|$).+",
    )
    if any(re.fullmatch(pattern, path) for pattern in patterns):
        return True
    if number == 11:
        if path.startswith("app/") or re.fullmatch(r"tests/test_app(?:_|\.|$).+", path):
            return True
    if number == 13:
        return path == "README.md" or path.startswith(("docs/", "reports/"))
    return False


def _file_contains_secret(path: Path) -> bool:
    patterns = (
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(rb"AKIA[0-9A-Z]{16}\b"),
        re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
        re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        re.compile(
            rb"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
            rb"[A-Za-z0-9_-]{8,}\b"
        ),
        re.compile(rb"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{16,}={0,2}\b"),
        re.compile(
            rb"(?im)^\s*(?:[A-Z0-9_]*API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|"
            rb"CLIENT[_-]?SECRET|PASSWORD|TOKEN|SECRET)\s*[:=]\s*"
            rb"(?:['\"])?(?!\$\{|<|REDACTED|CHANGEME|EXAMPLE)"
            rb"[A-Za-z0-9._~+/-]{8,}(?:['\"])?\s*$"
        ),
    )
    overlap = b""
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1_048_576):
                searchable = overlap + chunk
                if any(pattern.search(searchable) for pattern in patterns):
                    return True
                overlap = searchable[-512:]
    except OSError as exc:
        raise GitSafetyError(f"Cannot inspect candidate for secrets: {path}") from exc
    return False


def _validate_phase_paths(
    number: int, paths: tuple[str, ...], *, require_checkpoint: bool = True
) -> None:
    checkpoint = f"orchestration/state/checkpoints/phase_{number}_passed.json"
    for path in paths:
        if not _phase_path_is_expected(number, path):
            raise GitSafetyError(f"Unexpected Phase {number} file change: {path}")
        if path in PROTECTED_PATHS:
            raise GitSafetyError(f"Fundamental methodology/governance change requires approval: {path}")
        if path.startswith(("data/raw/", "data/preprocessed/")):
            raise GitSafetyError(f"Dataset change requires user intervention: {path}")
        if path.startswith((".git/", ".github/")):
            raise GitSafetyError(f"Unexpected repository-control change: {path}")
        if SECRET_NAME_RE.search(path):
            raise GitSafetyError(f"Possible secret-bearing file will not be committed: {path}")
        candidate = PROJECT_ROOT / path
        if candidate.is_file() and _file_contains_secret(candidate):
            raise GitSafetyError(f"Possible secret detected; refusing commit: {path}")
    if require_checkpoint:
        checkpoint_file = CHECKPOINT_DIR / f"phase_{number}_passed.json"
        if not checkpoint_file.is_file():
            raise GitSafetyError(f"Validated PASS checkpoint is missing: {checkpoint}")
        if _file_contains_secret(checkpoint_file):
            raise GitSafetyError(
                f"Possible secret detected in PASS checkpoint; refusing commit: {checkpoint}"
            )


def _checkpoint_commit_for_phase(number: int) -> str | None:
    import subprocess

    checkpoint = f"orchestration/state/checkpoints/phase_{number}_passed.json"
    completed = subprocess.run(
        ("git", "log", "-1", "--format=%H", "--", checkpoint),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    commit = completed.stdout.strip()
    return commit or None


def _preserved_attempt_count(number: int) -> int | None:
    state_dir = PROJECT_ROOT / "orchestration" / "state"
    patterns = (
        f"Phase_{number}_attempt_*.txt",
        f"phase_{number}_attempt_*.txt",
        f"phase_{number}_attempt_*_*.txt",
    )
    attempts: list[int] = []
    expression = re.compile(rf"[Pp]hase_{number}_attempt_(\d+)(?:_|\.)")
    for pattern in patterns:
        for path in state_dir.glob(pattern):
            match = expression.search(path.name)
            if match:
                attempts.append(int(match.group(1)))
    return max(attempts) if attempts else None


class AutoRunner:
    def __init__(
        self,
        *,
        max_attempts: int = 3,
        flow_factory: FlowFactory = WaterForecastFlow,
        committer: Committer = create_local_checkpoint,
        final_test_executor: Callable[[], TestReport] = run_test_commands,
        progress: Progress = print,
        state_path: Path = AUTO_STATE_PATH,
        summary_path: Path = FINAL_SUMMARY_PATH,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.max_attempts = max_attempts
        self.flow_factory = flow_factory
        self.committer = committer
        self.final_test_executor = final_test_executor
        self.progress = progress
        self.state_path = state_path
        self.summary_path = summary_path

    def _log(self, message: str) -> None:
        self.progress(f"[AUTO] {message}")

    def _stop(self, state: AutoState, reason: str, *, permission: bool = False) -> dict[str, Any]:
        state.status = "permission_required" if permission else "failed"
        state.failure_reason = reason
        _save_state(state, self.state_path)
        self._log(reason)
        return self._summary(state, test_result="NOT_RUN", warnings=[reason])

    def _summary(
        self,
        state: AutoState,
        *,
        test_result: str,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        completed = validated_phase_numbers()
        return {
            "status": state.status,
            "completed_phases": completed,
            "attempts_per_phase": state.attempts_per_phase,
            "test_result": test_result,
            "final_validator_status": "PASS" if 13 in completed else "NOT_RUN",
            "git_checkpoint_commits": state.git_checkpoint_commits,
            "artifacts_reports_locations": ["artifacts/", "reports/", "orchestration/state/"],
            "warnings": warnings or [],
        }

    def run(self) -> dict[str, Any]:
        state = _load_state(self.state_path)
        if state.status == "permission_required":
            self._log(state.failure_reason)
            return self._summary(
                state, test_result="NOT_RUN", warnings=[state.failure_reason]
            )
        try:
            completed = validated_phase_numbers()
        except ValueError as exc:
            return self._stop(state, str(exc))

        for number in completed:
            commit = _checkpoint_commit_for_phase(number)
            if commit:
                state.git_checkpoint_commits.setdefault(str(number), commit)
            attempts = _preserved_attempt_count(number)
            if attempts is not None:
                state.attempts_per_phase.setdefault(str(number), attempts)

        # Discover a crash after PASS checkpoint creation but before its commit
        # from the validated checkpoint chain and Git, not only ignored state.
        missing_commits = [
            number for number in completed if _checkpoint_commit_for_phase(number) is None
        ]
        if missing_commits:
            pending = missing_commits[-1]
            if len(missing_commits) != 1 or pending != completed[-1]:
                return self._stop(
                    state,
                    "Validated checkpoints without Git commits are not a single "
                    f"latest phase; refusing ambiguous recovery: {missing_commits}",
                )
            try:
                paths = changed_paths()
                _validate_phase_paths(pending, paths)
                checkpoint_path = f"orchestration/state/checkpoints/phase_{pending}_passed.json"
                result = self.committer(
                    message=(
                        f"feat: complete validated Phase {pending} "
                        f"{PHASE_SHORT_NAMES[pending]}"
                    ),
                    expected_paths=paths,
                    force_add_paths=(checkpoint_path,),
                )
                state.git_checkpoint_commits[str(pending)] = result.commit
                self._log(f"Phase {pending} pending local checkpoint committed")
            except GitSafetyError as exc:
                return self._stop(
                    state,
                    f"Phase {pending} PASS is preserved, but resume checkpoint was refused: {exc}",
                )
            if state.active_phase == pending:
                state.active_phase = None
                state.active_attempt = 0
                state.failure_report = ""
            _save_state(state, self.state_path)

        start_number = len(completed)
        continuing_interrupted_phase = state.active_phase == start_number
        if continuing_interrupted_phase:
            if state.active_attempt >= self.max_attempts:
                return self._stop(
                    state,
                    f"Phase {start_number} retry budget is exhausted at "
                    f"{state.active_attempt}/{self.max_attempts}; preserved corrections: "
                    f"{state.failure_report or state.failure_reason}",
                )
            try:
                _validate_phase_paths(
                    start_number, changed_paths(), require_checkpoint=False
                )
            except GitSafetyError as exc:
                return self._stop(
                    state,
                    f"Interrupted Phase {start_number} dirty state is unsafe: {exc}",
                )
        if start_number < len(PHASE_NUMBERS) and not continuing_interrupted_phase:
            try:
                require_clean_worktree()
            except GitSafetyError as exc:
                return self._stop(state, str(exc))

        for number in range(start_number, len(PHASE_NUMBERS)):
            continuing = state.active_phase == number
            state.status = "running"
            state.active_phase = number
            state.failure_reason = ""
            if not continuing:
                state.active_attempt = 0
                state.failure_report = ""
            _save_state(state, self.state_path)
            self._log(f"Phase {number} starting")

            def preserve_flow_state(flow_state: Any) -> None:
                state.active_attempt = flow_state.attempt
                if flow_state.validation_report:
                    state.failure_report = flow_state.validation_report
                if flow_state.failure_reason:
                    state.failure_reason = flow_state.failure_reason
                _save_state(state, self.state_path)

            flow = self.flow_factory(
                phases=[f"Phase {number}"],
                max_attempts=self.max_attempts,
                progress_reporter=lambda message, n=number: self._log(message),
                state_reporter=preserve_flow_state,
                starting_attempt=state.active_attempt,
                repair_report=state.failure_report,
            )
            flow.orchestrate()
            state.attempts_per_phase[str(number)] = flow.state.attempt
            state.active_attempt = flow.state.attempt
            report = getattr(flow.state, "validation_report", "")
            if report:
                state.failure_report = report
            elif flow.state.status != "completed" and flow.state.failure_reason:
                state.failure_report = (
                    "FAIL\n\nREQUIRED_CORRECTIONS:\n" + flow.state.failure_reason
                )
            _save_state(state, self.state_path)

            permission = _permission_request(flow.state.implementation_summary)
            if permission:
                return self._stop(
                    state,
                    f"Phase {number} stopped for permission: {permission}",
                    permission=True,
                )
            if flow.state.status != "completed":
                return self._stop(
                    state,
                    f"Phase {number} stopped: {flow.state.failure_reason}",
                )

            paths = changed_paths()
            try:
                _validate_phase_paths(number, paths)
                checkpoint_path = f"orchestration/state/checkpoints/phase_{number}_passed.json"
                result = self.committer(
                    message=f"feat: complete validated Phase {number} {PHASE_SHORT_NAMES[number]}",
                    expected_paths=paths,
                    force_add_paths=(checkpoint_path,),
                )
            except GitSafetyError as exc:
                return self._stop(
                    state,
                    f"Phase {number} PASS is preserved, but local checkpoint was refused: {exc}",
                )
            state.git_checkpoint_commits[str(number)] = result.commit
            state.active_phase = None
            state.active_attempt = 0
            state.failure_report = ""
            _save_state(state, self.state_path)
            self._log("local checkpoint committed")

        self._log("Phase 13 complete; running final full test suite")
        final_tests = self.final_test_executor()
        if not final_tests.passed:
            return self._stop(state, "Final full test suite failed.")
        self._log("final tests PASS")
        try:
            require_clean_worktree()
            import subprocess

            diff_check = subprocess.run(
                ("git", "diff", "--check"), cwd=PROJECT_ROOT, check=False, shell=False
            )
            if diff_check.returncode != 0:
                raise GitSafetyError("git diff --check failed")
            validated_phase_numbers()
        except (GitSafetyError, ValueError) as exc:
            return self._stop(state, f"Final verification failed: {exc}")

        state.status = "completed"
        state.active_phase = None
        state.failure_reason = ""
        _save_state(state, self.state_path)
        summary = self._summary(state, test_result="PASS")
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary
