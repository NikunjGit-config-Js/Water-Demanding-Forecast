from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from orchestration.tools import codex_tool


@pytest.fixture
def isolated_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(codex_tool, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(codex_tool, "STATE_DIR", project_root / "orchestration" / "state")
    monkeypatch.setattr(codex_tool, "LOG_DIR", project_root / "orchestration" / "logs")
    monkeypatch.setattr(codex_tool.shutil, "which", lambda _: "/usr/bin/codex")
    return project_root


def _successful_run(
    captured_commands: list[list[str]],
):
    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured_commands.append(command)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("EXPECTED RESPONSE\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "stdout", "")

    return fake_run


def test_read_only_creates_no_repository_output_or_log_files(
    monkeypatch: pytest.MonkeyPatch,
    isolated_paths: Path,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(codex_tool.subprocess, "run", _successful_run(commands))

    response = codex_tool.run_codex(
        "validate", sandbox="read-only", output_name="validator.txt"
    )

    assert response == "EXPECTED RESPONSE"
    assert list(isolated_paths.rglob("*")) == []
    output_path = Path(commands[0][commands[0].index("--output-last-message") + 1])
    assert not output_path.exists()
    assert not output_path.is_relative_to(isolated_paths)


def test_default_output_behavior_remains_stripped(
    monkeypatch: pytest.MonkeyPatch,
    isolated_paths: Path,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(codex_tool.subprocess, "run", _successful_run(commands))

    assert codex_tool.run_codex("prompt") == "EXPECTED RESPONSE"


@pytest.mark.parametrize("sandbox", ["read-only", "workspace-write"])
def test_preserve_output_returns_exact_file_content(
    monkeypatch: pytest.MonkeyPatch,
    isolated_paths: Path,
    sandbox: str,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(codex_tool.subprocess, "run", _successful_run(commands))

    response = codex_tool.run_codex(
        "prompt",
        sandbox=sandbox,
        output_name="raw.txt",
        preserve_output=True,
    )

    assert response == "EXPECTED RESPONSE\n"


@pytest.mark.parametrize(
    "output_name",
    [
        "/tmp/result.txt",
        "../result.txt",
        "subdir/result.txt",
        r"subdir\result.txt",
        "result..txt",
        "..",
        "",
    ],
)
def test_unsafe_output_names_are_rejected(output_name: str) -> None:
    with pytest.raises(ValueError, match="output_name"):
        codex_tool.run_codex("prompt", output_name=output_name)


def test_invalid_sandbox_is_rejected() -> None:
    with pytest.raises(ValueError, match="Only 'read-only' and 'workspace-write'"):
        codex_tool.run_codex("prompt", sandbox="danger-full-access")


def test_missing_codex_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codex_tool.shutil, "which", lambda _: None)

    with pytest.raises(codex_tool.CodexExecutionError, match="not found"):
        codex_tool.run_codex("prompt")


def test_nonzero_codex_exit(
    monkeypatch: pytest.MonkeyPatch,
    isolated_paths: Path,
) -> None:
    monkeypatch.setattr(
        codex_tool.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 17, "", "failure details"
        ),
    )

    with pytest.raises(codex_tool.CodexExecutionError, match="return code 17"):
        codex_tool.run_codex("prompt", sandbox="read-only")

    assert list(isolated_paths.rglob("*")) == []


def test_timeout(
    monkeypatch: pytest.MonkeyPatch,
    isolated_paths: Path,
) -> None:
    def raise_timeout(command: list[str], **_: object) -> None:
        raise subprocess.TimeoutExpired(command, timeout=3)

    monkeypatch.setattr(codex_tool.subprocess, "run", raise_timeout)

    with pytest.raises(codex_tool.CodexExecutionError, match="timed out after 3"):
        codex_tool.run_codex("prompt", sandbox="read-only", timeout=3)

    assert list(isolated_paths.rglob("*")) == []


def test_missing_expected_output(
    monkeypatch: pytest.MonkeyPatch,
    isolated_paths: Path,
) -> None:
    monkeypatch.setattr(
        codex_tool.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )

    with pytest.raises(codex_tool.CodexExecutionError, match="expected output"):
        codex_tool.run_codex("prompt", sandbox="read-only")

    assert list(isolated_paths.rglob("*")) == []


def test_workspace_write_uses_approve_for_me_without_explicit_sandbox(
    monkeypatch: pytest.MonkeyPatch,
    isolated_paths: Path,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(codex_tool.subprocess, "run", _successful_run(commands))

    response = codex_tool.run_codex(
        "implement", sandbox="workspace-write", output_name="implementation.txt"
    )

    assert response == "EXPECTED RESPONSE"
    command = commands[0]
    assert "--approve-for-me" in command
    assert "-s" not in command
    assert "workspace-write" not in command
    assert (codex_tool.STATE_DIR / "implementation.txt").is_file()
    assert (codex_tool.LOG_DIR / "implementation.log").is_file()
