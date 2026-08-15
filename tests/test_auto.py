from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestration import auto as auto_module
from orchestration import flow as flow_module
from orchestration.auto import AutoRunner, first_incomplete_phase, validated_phase_numbers
from orchestration.tools.git_tool import GitCheckpointResult, GitSafetyError
from orchestration.tools.test_tool import CommandResult, TestReport


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


@pytest.fixture
def checkpoint_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    directory = tmp_path / "checkpoints"
    monkeypatch.setattr(auto_module, "CHECKPOINT_DIR", directory)
    monkeypatch.setattr(flow_module, "CHECKPOINT_DIR", directory)
    return directory


def passing_tests() -> TestReport:
    return TestReport(
        (CommandResult(("python", "-m", "pytest", "-q"), 0, "passed", ""),)
    )


def test_first_incomplete_and_valid_pass_skipping(checkpoint_dir: Path) -> None:
    for number in range(5):
        write_checkpoint(checkpoint_dir, number)
    assert validated_phase_numbers() == [0, 1, 2, 3, 4]
    assert first_incomplete_phase() == 5


@pytest.mark.parametrize(
    "override",
    [
        {"validation_verdict": "FAIL"},
        {"schema_version": 1},
        {"validation_report": "pass"},
        {"unexpected": True},
    ],
)
def test_malformed_checkpoint_rejected(
    checkpoint_dir: Path, override: dict[str, object]
) -> None:
    write_checkpoint(checkpoint_dir, 0, **override)
    with pytest.raises(ValueError):
        validated_phase_numbers()


def test_checkpoint_gap_rejected(checkpoint_dir: Path) -> None:
    write_checkpoint(checkpoint_dir, 1)
    with pytest.raises(ValueError, match="after an incomplete phase"):
        validated_phase_numbers()


class FakeFlow:
    calls: list[int] = []
    init_calls: list[dict[str, object]] = []
    outcomes: dict[int, tuple[str, int, str, str]] = {}
    checkpoint_dir: Path

    def __init__(
        self,
        *,
        phases: list[str],
        max_attempts: int,
        progress_reporter,
        state_reporter=None,
        starting_attempt: int = 0,
        repair_report: str = "",
    ):
        self.number = int(phases[0].split()[1])
        self.init_calls.append(
            {
                "number": self.number,
                "starting_attempt": starting_attempt,
                "repair_report": repair_report,
            }
        )
        self.max_attempts = max_attempts
        self.reporter = progress_reporter
        self.state = SimpleNamespace(
            status="ready",
            attempt=starting_attempt,
            implementation_summary="",
            failure_reason="",
            validation_report=repair_report,
        )

    def orchestrate(self) -> None:
        self.calls.append(self.number)
        status, attempt, summary, reason = self.outcomes.get(
            self.number, ("completed", self.state.attempt + 1, "implemented", "")
        )
        self.reporter(f"implementation attempt {attempt}/{self.max_attempts}")
        self.state.status = status
        self.state.attempt = attempt
        self.state.implementation_summary = summary
        self.state.failure_reason = reason
        if status == "completed":
            write_checkpoint(self.checkpoint_dir, self.number)


def build_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint_dir: Path,
    *,
    outcomes: dict[int, tuple[str, int, str, str]] | None = None,
    committer=None,
) -> AutoRunner:
    FakeFlow.calls = []
    FakeFlow.init_calls = []
    FakeFlow.outcomes = outcomes or {}
    FakeFlow.checkpoint_dir = checkpoint_dir
    monkeypatch.setattr(auto_module, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(auto_module, "changed_paths", lambda: ("src/change.py",))
    monkeypatch.setattr(
        auto_module, "_validate_phase_paths", lambda number, paths, **kwargs: None
    )
    monkeypatch.setattr(auto_module, "_checkpoint_commit_for_phase", lambda number: None)
    commit = committer or (
        lambda **kwargs: GitCheckpointResult("abc123", tuple(kwargs["expected_paths"]))
    )
    return AutoRunner(
        max_attempts=3,
        flow_factory=FakeFlow,
        committer=commit,
        final_test_executor=passing_tests,
        progress=lambda message: None,
        state_path=tmp_path / "auto_state.json",
        summary_path=tmp_path / "summary.json",
    )


def test_chronological_progression_and_phase13_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint_dir: Path
) -> None:
    for number in range(12):
        write_checkpoint(checkpoint_dir, number)
    runner = build_runner(monkeypatch, tmp_path, checkpoint_dir)
    summary = runner.run()
    assert FakeFlow.calls == [12, 13]
    assert summary["status"] == "completed"
    assert summary["completed_phases"] == list(range(14))
    assert summary["test_result"] == "PASS"
    assert summary["final_validator_status"] == "PASS"


def test_no_advancement_after_fail_and_max_attempt_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint_dir: Path
) -> None:
    for number in range(5):
        write_checkpoint(checkpoint_dir, number)
    runner = build_runner(
        monkeypatch,
        tmp_path,
        checkpoint_dir,
        outcomes={5: ("failed", 3, "failed", "Retry limit exhausted")},
    )
    summary = runner.run()
    assert FakeFlow.calls == [5]
    assert summary["status"] == "failed"
    assert summary["attempts_per_phase"]["5"] == 3
    assert "Retry limit exhausted" in summary["warnings"][0]


def test_repair_loop_attempt_count_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint_dir: Path
) -> None:
    for number in range(13):
        write_checkpoint(checkpoint_dir, number)
    runner = build_runner(
        monkeypatch,
        tmp_path,
        checkpoint_dir,
        outcomes={13: ("completed", 2, "repaired", "")},
    )
    summary = runner.run()
    assert summary["attempts_per_phase"]["13"] == 2


def test_permission_required_stops_before_commit_or_advancement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint_dir: Path
) -> None:
    for number in range(5):
        write_checkpoint(checkpoint_dir, number)
    commits: list[object] = []
    runner = build_runner(
        monkeypatch,
        tmp_path,
        checkpoint_dir,
        outcomes={
            5: (
                "failed",
                1,
                "PERMISSION_REQUIRED\nApproval needed for external data scraping.",
                "permission stop",
            )
        },
        committer=lambda **kwargs: commits.append(kwargs),
    )
    summary = runner.run()
    assert summary["status"] == "permission_required"
    assert commits == []
    assert FakeFlow.calls == [5]


def test_unexpected_change_refusal_preserves_pass_for_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint_dir: Path
) -> None:
    for number in range(13):
        write_checkpoint(checkpoint_dir, number)
    runner = build_runner(
        monkeypatch,
        tmp_path,
        checkpoint_dir,
        committer=lambda **kwargs: (_ for _ in ()).throw(
            GitSafetyError("unexpected file")
        ),
    )
    summary = runner.run()
    assert summary["status"] == "failed"
    assert (checkpoint_dir / "phase_13_passed.json").exists()
    assert "local checkpoint was refused" in summary["warnings"][0]


def test_existing_phase_mode_parser_regression() -> None:
    from orchestration.main import build_parser

    args = build_parser().parse_args(["--phase", "Phase 5", "--max-attempts", "2"])
    assert args.phase == ["Phase 5"]
    assert args.auto is False
    assert args.max_attempts == 2


def test_auto_cli_parser() -> None:
    from orchestration.main import build_parser

    args = build_parser().parse_args(["--auto", "--max-attempts", "3"])
    assert args.auto is True
    assert args.phase is None
    assert args.max_attempts == 3


def test_auto_cli_dispatches_runner(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from orchestration import main as main_module

    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, *, max_attempts: int) -> None:
            captured["max_attempts"] = max_attempts

        def run(self) -> dict[str, object]:
            return {"status": "completed", "completed_phases": list(range(14))}

    monkeypatch.setattr(main_module, "AutoRunner", FakeRunner)
    monkeypatch.setattr(
        "sys.argv", ["orchestration.main", "--auto", "--max-attempts", "3"]
    )
    assert main_module.main() == 0
    assert captured["max_attempts"] == 3
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_restart_skips_all_valid_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint_dir: Path
) -> None:
    for number in range(14):
        write_checkpoint(checkpoint_dir, number)
    runner = build_runner(monkeypatch, tmp_path, checkpoint_dir)
    summary = runner.run()
    assert FakeFlow.calls == []
    assert summary["status"] == "completed"


def test_restart_commits_pass_checkpoint_without_rerunning_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint_dir: Path
) -> None:
    for number in range(14):
        write_checkpoint(checkpoint_dir, number)
    commits: list[dict[str, object]] = []
    runner = build_runner(
        monkeypatch,
        tmp_path,
        checkpoint_dir,
        committer=lambda **kwargs: commits.append(kwargs)
        or GitCheckpointResult("resumed123", tuple(kwargs["expected_paths"])),
    )
    runner.state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "running",
                "active_phase": 13,
                "attempts_per_phase": {"13": 1},
                "git_checkpoint_commits": {},
                "failure_reason": "",
                "updated_at_utc": "2026-08-15T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    summary = runner.run()
    assert FakeFlow.calls == []
    assert len(commits) == 1
    assert summary["git_checkpoint_commits"]["13"] == "resumed123"


@pytest.mark.parametrize(
    "path",
    [
        "PROJECT_SPEC.md",
        "data/raw/source.csv",
        ".github/workflow.yml",
        "surprise.txt",
        "src/unrelated.py",
        "tests/test_unrelated.py",
        "artifacts/phase6/wrong_phase.json",
    ],
)
def test_unexpected_or_protected_phase_paths_are_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint_dir: Path,
    path: str,
) -> None:
    write_checkpoint(checkpoint_dir, 5)
    monkeypatch.setattr(auto_module, "PROJECT_ROOT", tmp_path)
    with pytest.raises(GitSafetyError):
        auto_module._validate_phase_paths(5, (path,))


def test_restart_retains_corrections_and_remaining_attempt_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint_dir: Path
) -> None:
    for number in range(13):
        write_checkpoint(checkpoint_dir, number)
    exact = "FAIL\nREQUIRED_CORRECTIONS:\nFix the chronological split."
    runner = build_runner(monkeypatch, tmp_path, checkpoint_dir)
    runner.state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "active_phase": 13,
                "active_attempt": 1,
                "failure_report": exact,
                "attempts_per_phase": {"13": 1},
                "git_checkpoint_commits": {},
                "failure_reason": "interrupted",
                "updated_at_utc": "2026-08-15T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    summary = runner.run()
    assert summary["status"] == "completed"
    assert FakeFlow.init_calls[0]["starting_attempt"] == 1
    assert FakeFlow.init_calls[0]["repair_report"] == exact
    assert summary["attempts_per_phase"]["13"] == 2


def test_exhausted_restart_does_not_grant_fresh_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint_dir: Path
) -> None:
    for number in range(13):
        write_checkpoint(checkpoint_dir, number)
    runner = build_runner(monkeypatch, tmp_path, checkpoint_dir)
    runner.state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "active_phase": 13,
                "active_attempt": 3,
                "failure_report": "FAIL\nREQUIRED_CORRECTIONS:\nStill broken",
                "attempts_per_phase": {"13": 3},
                "git_checkpoint_commits": {},
                "failure_reason": "exhausted",
                "updated_at_utc": "2026-08-15T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    summary = runner.run()
    assert summary["status"] == "failed"
    assert FakeFlow.calls == []
    assert "retry budget is exhausted" in summary["warnings"][0]


@pytest.mark.parametrize(
    "payload",
    [
        b"\x00\xff-----BEGIN PRIVATE KEY-----\x00",
        b"x" * 2_100_000 + b"AKIAABCDEFGHIJKLMNOP",
    ],
    ids=("binary-private-key", "large-aws-key"),
)
def test_binary_and_large_secret_content_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint_dir: Path,
    payload: bytes,
) -> None:
    artifact = tmp_path / "artifacts" / "phase5" / "candidate.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(payload)
    write_checkpoint(checkpoint_dir, 5)
    monkeypatch.setattr(auto_module, "PROJECT_ROOT", tmp_path)
    with pytest.raises(GitSafetyError, match="secret"):
        auto_module._validate_phase_paths(5, ("artifacts/phase5/candidate.bin",))


@pytest.mark.parametrize(
    "credential",
    [
        b"OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456",
        b"TOKEN=unquotedtokenvalue12345",
        b"PASSWORD=correct-horse-battery-staple",
        b"github_token=ghp_abcdefghijklmnopqrstuvwxyz123456",
        b"Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456",
        b"eyJabcdefghijk.abcdefghijklmnop.abcdefghijklmnop",
    ],
    ids=("openai", "unquoted-token", "password", "github", "bearer", "jwt"),
)
def test_common_credentials_are_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint_dir: Path,
    credential: bytes,
) -> None:
    artifact = tmp_path / "artifacts" / "phase5" / "config.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(credential)
    write_checkpoint(checkpoint_dir, 5)
    monkeypatch.setattr(auto_module, "PROJECT_ROOT", tmp_path)
    with pytest.raises(GitSafetyError, match="secret"):
        auto_module._validate_phase_paths(5, ("artifacts/phase5/config.bin",))


def test_secret_bearing_force_added_checkpoint_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint_dir: Path
) -> None:
    write_checkpoint(
        checkpoint_dir,
        5,
        validation_report=(
            "PASS\nAccidentally included OPENAI_API_KEY="
            "sk-abcdefghijklmnopqrstuvwxyz123456"
        ),
    )
    monkeypatch.setattr(auto_module, "PROJECT_ROOT", tmp_path)
    with pytest.raises(GitSafetyError, match="PASS checkpoint"):
        auto_module._validate_phase_paths(5, ())
