from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / "orchestration" / "state"
LOG_DIR = PROJECT_ROOT / "orchestration" / "logs"
ALLOWED_SANDBOXES = frozenset({"read-only", "workspace-write"})


class CodexExecutionError(RuntimeError):
    """Raised when a Codex execution fails."""


def _validate_output_name(output_name: str) -> str:
    """Return a safe filename that cannot escape the artifact directories."""
    if not isinstance(output_name, str) or not output_name:
        raise ValueError("output_name must be a non-empty filename.")
    if (
        Path(output_name).is_absolute()
        or "/" in output_name
        or "\\" in output_name
        or ".." in output_name
        or Path(output_name).name != output_name
    ):
        raise ValueError("output_name must be a filename without traversal components.")

    candidate = (STATE_DIR / output_name).resolve()
    try:
        candidate.relative_to(STATE_DIR.resolve())
    except ValueError as exc:
        raise ValueError("output_name resolves outside the state directory.") from exc
    return output_name


def _external_temporary_directory() -> tempfile.TemporaryDirectory[str]:
    """Create a private temporary directory and ensure it is outside the project."""
    temporary_directory = tempfile.TemporaryDirectory(prefix="water-forecast-codex-")
    temporary_path = Path(temporary_directory.name).resolve()
    try:
        temporary_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return temporary_directory

    temporary_directory.cleanup()
    raise CodexExecutionError(
        "The system temporary directory is inside PROJECT_ROOT; read-only execution "
        "cannot safely continue."
    )


def _run_command(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CodexExecutionError(
            f"Codex timed out after {timeout} seconds."
        ) from exc


def _write_log(
    output_name: str,
    command: list[str],
    result: subprocess.CompletedProcess[str],
) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{Path(output_name).stem}.log"
    log_file.write_text(
        "COMMAND:\n"
        + " ".join(command[:-1])
        + " [PROMPT]\n\n"
        + "STDOUT:\n"
        + result.stdout
        + "\n\nSTDERR:\n"
        + result.stderr,
        encoding="utf-8",
    )
    return log_file


def run_codex(
    prompt: str,
    *,
    sandbox: str = "read-only",
    output_name: str = "codex_last_message.txt",
    timeout: int = 3600,
) -> str:
    """Run Codex non-interactively and return its final response.

    Read-only execution uses a private temporary output location outside the
    repository and never creates state or log files. An explicitly requested
    ``workspace-write`` execution persists its response and command log.
    """
    if sandbox not in ALLOWED_SANDBOXES:
        raise ValueError(
            "Only 'read-only' and 'workspace-write' are allowed by this wrapper."
        )
    safe_output_name = _validate_output_name(output_name)

    codex_path = shutil.which("codex")
    if not codex_path:
        raise CodexExecutionError("Codex CLI was not found on PATH.")

    if sandbox == "read-only":
        with _external_temporary_directory() as temporary_directory:
            output_file = Path(temporary_directory) / safe_output_name
            command = [
                codex_path,
                "exec",
                "-C",
                str(PROJECT_ROOT),
                "-s",
                "read-only",
                "--output-last-message",
                str(output_file),
                prompt,
            ]
            result = _run_command(command, timeout)
            if result.returncode != 0:
                raise CodexExecutionError(
                    f"Codex failed with return code {result.returncode}.\n"
                    f"STDERR:\n{result.stderr}"
                )
            if not output_file.is_file():
                raise CodexExecutionError(
                    "Codex completed but did not create the expected output file."
                )
            return output_file.read_text(encoding="utf-8").strip()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    output_file = STATE_DIR / safe_output_name
    # Codex CLI 0.147.0 rejects --approve-for-me combined with an explicit
    # `-s workspace-write`; automatic approval already selects writable execution.
    command = [
        codex_path,
        "exec",
        "-C",
        str(PROJECT_ROOT),
        "--approve-for-me",
        "--output-last-message",
        str(output_file),
        prompt,
    ]
    result = _run_command(command, timeout)
    log_file = _write_log(safe_output_name, command, result)

    if result.returncode != 0:
        raise CodexExecutionError(
            f"Codex failed with return code {result.returncode}.\nSee log: {log_file}"
        )
    if not output_file.is_file():
        raise CodexExecutionError(
            f"Codex completed but did not create {output_file}. See log: {log_file}"
        )
    return output_file.read_text(encoding="utf-8").strip()


if __name__ == "__main__":
    response = run_codex(
        """
Read AGENTS.md and PROJECT_SPEC.md.

Do not modify project files.

Return exactly:

CODEX_WRAPPER_READY
PROJECT_RULES_LOADED: YES
WORKSPACE: water-forecast
""".strip(),
        sandbox="read-only",
    )
    print(response)
