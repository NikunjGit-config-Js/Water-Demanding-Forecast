from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GitSnapshot:
    status: str
    diff: str


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


def capture_git_snapshot() -> GitSnapshot:
    """Capture repository state without staging, committing, or pushing."""
    return GitSnapshot(
        status=_read_git("status", "--short"),
        diff=_read_git("diff", "--no-ext-diff", "--"),
    )
