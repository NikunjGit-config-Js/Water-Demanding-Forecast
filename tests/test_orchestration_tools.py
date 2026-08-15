from __future__ import annotations

from pathlib import Path

import pytest

from orchestration.flow import load_orchestration_config
from orchestration.tools import experiment_reader
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
