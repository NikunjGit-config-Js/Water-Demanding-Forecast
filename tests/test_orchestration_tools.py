from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from orchestration.flow import load_orchestration_config
from orchestration.tools import experiment_reader
from orchestration.tools import git_tool
from orchestration.tools import test_tool


def test_configs_define_all_roles_and_tasks() -> None:
    agents, tasks = load_orchestration_config()
    assert len(agents) == 8
    assert {"implementation", "repair", "validation"}.issubset(tasks)


@pytest.mark.parametrize("path", ["../secret", "/tmp/secret", "data/raw.csv", "orchestration/../../secret"])
def test_experiment_reader_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError):
        experiment_reader.resolve_evidence_path(path)


def test_experiment_reader_reads_approved_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "project"
    evidence = root / "artifacts" / "result.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"ok": true}', encoding="utf-8")
    monkeypatch.setattr(experiment_reader, "PROJECT_ROOT", root)
    assert experiment_reader.read_json_evidence("artifacts/result.json") == {"ok": True}


@pytest.mark.parametrize("approved_root", ["artifacts", "experiments", "orchestration"])
def test_experiment_reader_rejects_symlink_escape_from_selected_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, approved_root: str
) -> None:
    root = tmp_path / "project"
    selected = root / approved_root
    outside = root / "outside"
    selected.mkdir(parents=True)
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (selected / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(experiment_reader, "PROJECT_ROOT", root)
    with pytest.raises(ValueError, match="approved root"):
        experiment_reader.resolve_evidence_path(f"{approved_root}/escape/secret.txt")


@pytest.mark.parametrize(
    "command",
    [
        ("python", "-c", "print('unsafe')"),
        ("python", "arbitrary_script.py"),
        ("python", "-m", "pytest", "-q", "-p", "arbitrary_plugin"),
        ("python", "-m", "pytest", "-q", "--basetemp", "/tmp/mutate"),
        ("sh", "-c", "pytest -q"),
        ("bash", "-c", "pytest -q"),
        ("pytest", "-q"),
        ("git", "status"),
    ],
)
def test_test_tool_rejects_unapproved_command_shapes(command: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="not approved"):
        test_tool.run_test_commands([command])


def test_test_tool_runs_only_exact_repository_pytest_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs: object) -> object:
        import subprocess

        captured.append(command)
        return subprocess.CompletedProcess(command, 0, "55 passed", "")

    monkeypatch.setattr(test_tool.subprocess, "run", fake_run)
    report = test_tool.run_test_commands()
    assert report.passed
    assert captured == [("python", "-m", "pytest", "-q")]


def test_safe_local_git_checkpoint_stages_exact_paths_and_commits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    subprocess.run(("git", "init"), cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"),
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Test User"), cwd=tmp_path, check=True
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-m", "initial"), cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("checkpoint.json\n", encoding="utf-8")
    subprocess.run(("git", "add", ".gitignore"), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-m", "ignore"), cwd=tmp_path, check=True)
    tracked.write_text("after\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(git_tool, "PROJECT_ROOT", tmp_path)

    result = git_tool.create_local_checkpoint(
        message="feat: safe checkpoint",
        expected_paths=("tracked.txt",),
        force_add_paths=("checkpoint.json",),
    )

    assert set(result.paths) == {"tracked.txt", "checkpoint.json"}
    assert git_tool.changed_paths() == ()


def test_local_git_checkpoint_refuses_unexpected_delta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    subprocess.run(("git", "init"), cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "unexpected.txt").write_text("change\n", encoding="utf-8")
    monkeypatch.setattr(git_tool, "PROJECT_ROOT", tmp_path)
    with pytest.raises(git_tool.GitSafetyError, match="delta changed"):
        git_tool.create_local_checkpoint(
            message="should not commit", expected_paths=("expected.txt",)
        )


def test_push_safety_accepts_only_approved_branch_and_remotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        ("branch", "--show-current"): git_tool.EXPECTED_PUSH_BRANCH + "\n",
        ("remote", "get-url", "origin"): git_tool.EXPECTED_ORIGIN_URL + "\n",
        ("remote", "get-url", "upstream"): git_tool.EXPECTED_UPSTREAM_URL + "\n",
    }
    monkeypatch.setattr(git_tool, "_read_git", lambda *args: values[args])
    git_tool.validate_push_target()
    with pytest.raises(git_tool.GitSafetyError, match="not approved"):
        git_tool.validate_push_target(branch="main")


def test_guarded_push_uses_normal_explicit_origin_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(git_tool, "validate_push_target", lambda **kwargs: None)
    monkeypatch.setattr(git_tool, "_run_git", lambda *args: calls.append(args) or "")
    git_tool.push_approved_branch()
    assert calls == [("push", "origin", "feature/india-multicity")]
