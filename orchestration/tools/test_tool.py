from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("python", "-m", "pytest", "-q"),
)
APPROVED_TEST_COMMANDS = frozenset(DEFAULT_TEST_COMMANDS)


class CheckpointTestEvidence(BaseModel):
    """Strict, serializable proof for the only approved prerequisite test gate."""

    model_config = ConfigDict(extra="forbid", strict=True)

    command: list[str]
    returncode: int

    @field_validator("command")
    @classmethod
    def command_must_match_approved_gate(cls, value: list[str]) -> list[str]:
        if value != list(DEFAULT_TEST_COMMANDS[0]):
            raise ValueError("checkpoint test command is not the approved test gate")
        return value

    @field_validator("returncode")
    @classmethod
    def returncode_must_be_zero(cls, value: int) -> int:
        if value != 0:
            raise ValueError("checkpoint test returncode must be exactly 0")
        return value


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class TestReport:
    __test__ = False
    results: tuple[CommandResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    def summary(self) -> str:
        sections = []
        for result in self.results:
            sections.append(
                f"$ {shlex.join(result.command)}\n"
                f"exit_code={result.returncode}\n"
                f"STDOUT:\n{result.stdout.strip()}\n"
                f"STDERR:\n{result.stderr.strip()}"
            )
        return "\n\n".join(sections)

    def checkpoint_evidence(self) -> CheckpointTestEvidence:
        if len(self.results) != 1:
            raise ValueError("Checkpoint evidence requires exactly one test result.")
        result = self.results[0]
        return CheckpointTestEvidence(
            command=list(result.command),
            returncode=result.returncode,
        )


def run_test_commands(
    commands: Sequence[Sequence[str]] = DEFAULT_TEST_COMMANDS,
    *,
    timeout: int = 1800,
) -> TestReport:
    """Run explicitly-tokenized, non-shell test commands in the project root."""
    if not commands:
        raise ValueError("At least one test command is required.")

    results = []
    for raw_command in commands:
        command = tuple(raw_command)
        if command not in APPROVED_TEST_COMMANDS:
            raise ValueError(f"Test command is not approved: {command!r}")
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            shell=False,
        )
        results.append(
            CommandResult(command, completed.returncode, completed.stdout, completed.stderr)
        )
        if completed.returncode != 0:
            break
    return TestReport(tuple(results))
