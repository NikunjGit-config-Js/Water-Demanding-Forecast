from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_ROOT_NAMES = frozenset({"artifacts", "experiments", "orchestration"})


def resolve_evidence_path(relative_path: str) -> Path:
    """Resolve a repository evidence path while preventing traversal."""
    candidate_input = Path(relative_path)
    if candidate_input.is_absolute() or ".." in candidate_input.parts:
        raise ValueError("Evidence path must be a safe repository-relative path.")
    if not candidate_input.parts or candidate_input.parts[0] not in ALLOWED_ROOT_NAMES:
        raise ValueError("Evidence path is outside approved artifact directories.")
    resolved_project_root = PROJECT_ROOT.resolve()
    selected_root = (PROJECT_ROOT / candidate_input.parts[0]).resolve()
    try:
        selected_root.relative_to(resolved_project_root)
    except ValueError as exc:
        raise ValueError("Approved evidence root resolves outside the project.") from exc
    candidate = (PROJECT_ROOT / candidate_input).resolve()
    try:
        candidate.relative_to(selected_root)
    except ValueError as exc:
        raise ValueError("Evidence path resolves outside its approved root.") from exc
    return candidate


def read_text_evidence(relative_path: str, *, max_bytes: int = 1_000_000) -> str:
    path = resolve_evidence_path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > max_bytes:
        raise ValueError(f"Evidence file exceeds {max_bytes} bytes: {relative_path}")
    return path.read_text(encoding="utf-8")


def read_json_evidence(relative_path: str, *, max_bytes: int = 1_000_000) -> Any:
    return json.loads(read_text_evidence(relative_path, max_bytes=max_bytes))
