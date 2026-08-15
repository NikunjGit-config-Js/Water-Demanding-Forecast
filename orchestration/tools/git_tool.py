from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GitSnapshot:
    status: str
    diff: str


@dataclass(frozen=True)
class GitCheckpointResult:
    commit: str
    paths: tuple[str, ...]


class GitSafetyError(RuntimeError):
    """Raised when a local checkpoint cannot be created safely."""


def _read_git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise GitSafetyError(
            completed.stderr.strip() or f"git {' '.join(args)} failed"
        )
    return completed.stdout


def capture_git_snapshot() -> GitSnapshot:
    """Capture repository state without staging, committing, or pushing."""
    return GitSnapshot(
        status=_read_git("status", "--short"),
        diff=_read_git("diff", "--no-ext-diff", "--"),
    )


def changed_paths() -> tuple[str, ...]:
    """Return all staged, unstaged, and untracked paths without ambiguity."""
    output = _read_git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    entries = output.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        if not entry:
            break
        status = entry[:2]
        path = entry[3:]
        if "R" in status or "C" in status:
            index += 1
            if index >= len(entries) or not entries[index]:
                raise GitSafetyError("Malformed Git rename/copy status output.")
            path = entries[index]
        paths.append(path)
        index += 1
    return tuple(paths)


def require_clean_worktree() -> None:
    paths = changed_paths()
    if paths:
        raise GitSafetyError(
            "Auto mode requires a clean worktree before starting a phase; "
            f"unexpected pre-existing changes: {', '.join(paths)}"
        )


def create_local_checkpoint(
    *,
    message: str,
    expected_paths: Iterable[str],
    force_add_paths: Iterable[str] = (),
) -> GitCheckpointResult:
    """Stage an exact reviewed path set and create a non-rewriting local commit."""
    expected = tuple(dict.fromkeys(expected_paths))
    actual = changed_paths()
    forced = set(force_add_paths)
    if not actual and not forced:
        raise GitSafetyError("No phase changes exist to checkpoint.")
    if set(actual) != set(expected):
        unexpected = sorted(set(actual) - set(expected))
        missing = sorted(set(expected) - set(actual))
        raise GitSafetyError(
            "Refusing local checkpoint because the worktree delta changed. "
            f"Unexpected: {unexpected}; missing: {missing}."
        )
    for path in actual:
        status = _read_git("status", "--porcelain=v1", "--", path)[:2]
        if "D" in status:
            raise GitSafetyError(f"Refusing to commit deletion: {path}")
        source_path = PROJECT_ROOT / path
        if source_path.is_symlink():
            raise GitSafetyError(f"Refusing to commit a symbolic link: {path}")
        resolved = source_path.resolve()
        try:
            resolved.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise GitSafetyError(f"Path escapes the project root: {path}") from exc

    for path in forced:
        candidate = PROJECT_ROOT / path
        if not candidate.is_file():
            raise GitSafetyError(f"Forced checkpoint path is missing: {path}")
    for path in actual:
        _run_git("add", "--", path)
    for path in sorted(forced):
        _run_git("add", "-f", "--", path)
    _run_git("diff", "--check")
    _run_git("diff", "--cached", "--check")
    staged = tuple(
        path for path in _read_git("diff", "--cached", "--name-only", "-z").split("\0") if path
    )
    reviewed = set(actual) | forced
    if set(staged) != reviewed:
        raise GitSafetyError(
            f"Staged paths do not match reviewed paths: staged={staged}, reviewed={sorted(reviewed)}"
        )
    _run_git("commit", "-m", message, "--", *staged)
    commit = _read_git("rev-parse", "HEAD").strip()
    return GitCheckpointResult(commit=commit, paths=staged)
