"""Run Phase 12 with correct hashes for Delhi."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.phase12_full_validation import EvaluationSpec, run_audit
from scripts.delhi_pipeline import ARTIFACT_ROOT, DATASET, write_checkpoint

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()

def find_latest(subdir: Path) -> Path:
    if not subdir.exists():
        return subdir
    parts = sorted(d for d in subdir.iterdir() if d.is_dir())
    return parts[-1] if parts else subdir

dataset_sha = sha256_file(DATASET)
n_rows = len(__import__("pandas").read_csv(DATASET))
print(f"Dataset SHA256: {dataset_sha}")
print(f"Rows: {n_rows}")

p5 = find_latest(ARTIFACT_ROOT / "phase5")
p6 = find_latest(ARTIFACT_ROOT / "phase6")
p7 = find_latest(ARTIFACT_ROOT / "phase7")
p8 = find_latest(ARTIFACT_ROOT / "phase8")
p9 = find_latest(ARTIFACT_ROOT / "phase9")
p10 = find_latest(ARTIFACT_ROOT / "phase10")

print(f"Phase 5 dir: {p5}")
print(f"Phase 6 dir: {p6}")
print(f"Phase 7 dir: {p7}")
print(f"Phase 8 dir: {p8}")
print(f"Phase 9 dir: {p9}")
print(f"Phase 10 dir: {p10}")

def make_hashes(*files):
    return tuple((f.name, sha256_file(f)) for f in files if f.exists())

def manifest_sha(path: Path) -> str | None:
    if path.exists():
        return sha256_file(path)
    return None

evaluations = {
    "phase5": EvaluationSpec(
        predictions=p5 / "predictions.csv",
        metrics=p5 / "metrics.csv",
        protocol=p5 / "holdout_report.json",
        cv=False,
        approved_hashes=make_hashes(p5 / "predictions.csv", p5 / "metrics.csv", p5 / "holdout_report.json"),
    ),
    "phase6": EvaluationSpec(
        predictions=p6 / "predictions.csv",
        metrics=p6 / "metrics_summary.csv",
        protocol=p6 / "cv_report.json",
        cv=True,
        approved_hashes=make_hashes(p6 / "predictions.csv", p6 / "metrics_summary.csv", p6 / "cv_report.json"),
    ),
    "phase7": EvaluationSpec(
        predictions=p7 / "locked_test_predictions.csv",
        metrics=p7 / "locked_test_metrics.csv",
        protocol=p7 / "split_and_selection_report.json",
        cv=False,
        approved_hashes=make_hashes(p7 / "locked_test_predictions.csv", p7 / "locked_test_metrics.csv", p7 / "split_and_selection_report.json"),
    ),
    "phase8": EvaluationSpec(
        predictions=p8 / "predictions.csv",
        metrics=p8 / "metrics_summary.csv",
        protocol=p8 / "cv_selection_report.json",
        cv=True,
        manifest=p8 / "artifact_hashes.json" if (p8 / "artifact_hashes.json").exists() else None,
        approved_manifest_sha256=manifest_sha(p8 / "artifact_hashes.json"),
    ),
    "phase9": EvaluationSpec(
        predictions=p9 / "predictions.csv",
        metrics=p9 / "metrics.csv",
        protocol=p9 / "forecast_report.json",
        cv=False,
        manifest=p9 / "artifact_hashes.json" if (p9 / "artifact_hashes.json").exists() else None,
        approved_manifest_sha256=manifest_sha(p9 / "artifact_hashes.json"),
    ),
    "phase10": EvaluationSpec(
        predictions=p10 / "predictions.csv",
        metrics=p10 / "metrics.csv",
        protocol=p10 / "forecast_report.json",
        cv=False,
        manifest=p10 / "artifact_hashes.json" if (p10 / "artifact_hashes.json").exists() else None,
        approved_manifest_sha256=manifest_sha(p10 / "artifact_hashes.json"),
    ),
}

result = run_audit(
    dataset=DATASET,
    approved_dataset_sha256=dataset_sha,
    evaluations=evaluations,
    expected_rows=n_rows,
)

out_dir = ARTIFACT_ROOT / "phase12" / "delhi_validation"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "full_validation_report.json").write_text(
    json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8"
)
write_checkpoint(12, "Delhi full validation audit complete.")
print(f"\nPhase 12 PASS - artifacts: {out_dir}")
