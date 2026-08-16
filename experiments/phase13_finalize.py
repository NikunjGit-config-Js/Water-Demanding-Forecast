"""Build the Phase 13 index of validated documentation and final artifacts.

This command only reads existing repository evidence and writes a small JSON
manifest. It does not fit models, read secrets, or invoke validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from orchestration.flow import PhaseCheckpoint


REQUIRED_CHECKPOINTS = tuple(f"phase_{number}_passed.json" for number in range(13))
CHECKPOINT_FIELDS = frozenset(PhaseCheckpoint.model_fields)
FINAL_EVIDENCE = (
    "PROJECT_SPEC.md",
    "EXPERIMENT_PLAN.md",
    "VALIDATION_RULES.md",
    "README.md",
    "docs/INTERVIEW_GUIDE.md",
    "artifacts/phase5/phase5_attempt_2_20260815T230000Z/metrics.csv",
    "artifacts/phase6/phase6_attempt_1_20260815T233000Z/metrics_summary.csv",
    "artifacts/phase7/phase7_attempt_1_20260815T223000Z/locked_test_metrics.csv",
    "artifacts/phase8/phase8_attempt_2_final_20260816T003000Z/metrics_summary.csv",
    "artifacts/phase9/phase9_attempt_1_final_20260816T012000Z/metrics.csv",
    "artifacts/phase10/phase10_attempt_1_final_20260816T021000Z/metrics.csv",
    "artifacts/phase12/phase12_attempt_2_20260816T050000Z/full_validation_report.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(repository_root: Path) -> dict[str, object]:
    """Return a deterministic manifest, failing if approved evidence is absent."""
    root = repository_root.resolve()
    checkpoint_dir = root / "orchestration/state/checkpoints"
    checkpoints: list[dict[str, object]] = []
    for expected_phase_number, filename in enumerate(REQUIRED_CHECKPOINTS):
        path = checkpoint_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing prerequisite checkpoint: {path}")
        try:
            raw_record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw_record, dict) or set(raw_record) != CHECKPOINT_FIELDS:
                raise ValueError("checkpoint fields must exactly match schema v2")
            record = PhaseCheckpoint.model_validate(raw_record)
        except (OSError, ValueError) as exc:
            raise ValueError(f"checkpoint is not a schema-v2 PASS: {path}") from exc
        expected_phase_name = f"Phase {expected_phase_number}"
        if (
            record.phase_number != expected_phase_number
            or record.phase_name != expected_phase_name
        ):
            raise ValueError(
                f"checkpoint content does not match {expected_phase_name}: {path}"
            )
        checkpoints.append(
            {
                "path": path.relative_to(root).as_posix(),
                "phase": record.phase_name,
                "sha256": _sha256(path),
            }
        )

    files: list[dict[str, object]] = []
    for relative in FINAL_EVIDENCE:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing final evidence: {path}")
        files.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )

    return {
        "phase": "Phase 13",
        "scope": "documentation and final artifact index",
        "validator_invoked": False,
        "prerequisite_checkpoints": checkpoints,
        "files": files,
        "notes": [
            "Only independently approved Phase 0-12 checkpoints are prerequisites.",
            "Phase 13 remains pending independent validation.",
            "Historical failed and superseded attempts remain preserved for audit history.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.repository_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
