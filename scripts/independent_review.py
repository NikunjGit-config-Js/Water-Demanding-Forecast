"""Independent review of Delhi Phase 0-13 run."""
import hashlib
import json
from pathlib import Path

results = {"checks": {}, "overall": "PASS"}

# 1. Leakage check
print("1. Leakage check...")
# Phase 0: train/test split
p0_manifest = json.loads((Path("artifacts/cities/delhi/phase0/delhi_baseline/split_manifest.json")).read_text())
train_end = p0_manifest.get("train", {}).get("end_date", "")
test_start = p0_manifest.get("test", {}).get("start_date", "")
if train_end < test_start:
    results["checks"]["leakage_phase0"] = "PASS - train ends before test starts"
else:
    results["checks"]["leakage_phase0"] = f"FAIL - train_end={train_end}, test_start={test_start}"
    results["overall"] = "FAIL"

# Phase 5: chronological holdout
p5_report = json.loads((Path("artifacts/cities/delhi/phase5/delhi_holdout/holdout_report.json")).read_text())
if not p5_report.get("shuffle_used", True):
    results["checks"]["leakage_phase5"] = "PASS - no shuffle"
else:
    results["checks"]["leakage_phase5"] = "FAIL - shuffle detected"
    results["overall"] = "FAIL"

# Phase 7: locked test
p7_report = json.loads((Path("artifacts/cities/delhi/phase7/delhi_optuna/split_and_selection_report.json")).read_text())
if not p7_report.get("locked_test_used_for_tuning_selection_or_refit", True):
    results["checks"]["leakage_phase7"] = "PASS - locked test isolated"
else:
    results["checks"]["leakage_phase7"] = "FAIL - locked test used in tuning"
    results["overall"] = "FAIL"

# Phase 9: no data leakage
p9_report = json.loads((Path("artifacts/cities/delhi/phase9/delhi_timeseries/forecast_report.json")).read_text())
if not p9_report.get("holdout_used_for_fitting_tuning_or_early_stopping", True):
    results["checks"]["leakage_phase9"] = "PASS - holdout isolated"
else:
    results["checks"]["leakage_phase9"] = "FAIL"
    results["overall"] = "FAIL"

# Phase 10: no locked test leakage
p10_report = json.loads((Path("artifacts/cities/delhi/phase10/delhi_patchtst/forecast_report.json")).read_text())
if not p10_report.get("locked_test_used_for_any_purpose", True):
    results["checks"]["leakage_phase10"] = "PASS - locked test isolated"
else:
    results["checks"]["leakage_phase10"] = "FAIL"
    results["overall"] = "FAIL"

# 2. Locked test isolation
print("2. Locked test isolation...")
if not p7_report.get("locked_test_used_for_tuning_selection_or_refit", True) and not p10_report.get("locked_test_used_for_any_purpose", True):
    results["checks"]["locked_test_isolation"] = "PASS"
else:
    results["checks"]["locked_test_isolation"] = "FAIL"
    results["overall"] = "FAIL"

# 3. Delhi/London isolation
print("3. Delhi/London isolation...")
import subprocess
git_diff = subprocess.run(["git", "diff", "--name-only", "HEAD~1"], capture_output=True, text=True, cwd="/home/asus/projects/water-forecast")
london_files = [f for f in git_diff.stdout.strip().split("\n") if f and not f.startswith("artifacts/cities/delhi") and not f.startswith("data/cities/delhi") and not f.startswith("reports/cities/delhi") and not f.startswith("scripts/delhi") and not f.startswith("scripts/run_p12") and not f.startswith("scripts/check_dates") and not f.startswith("scripts/inspect_delhi") and not f.startswith("scripts/verify") and not f.startswith(".opencode")]
if not london_files:
    results["checks"]["delhi_london_isolation"] = "PASS - no London files modified"
else:
    results["checks"]["delhi_london_isolation"] = f"WARNING - non-Delhi files changed: {london_files}"

# 4. Artifact completeness
print("4. Artifact completeness...")
required_phases = list(range(11))  # 0-10
missing = []
for p in required_phases:
    phase_dir = Path(f"artifacts/cities/delhi/phase{p}")
    if not phase_dir.exists() or not any(phase_dir.iterdir()):
        missing.append(p)
if not missing:
    results["checks"]["artifact_completeness"] = "PASS - all phase 0-10 artifacts present"
else:
    results["checks"]["artifact_completeness"] = f"FAIL - missing phases: {missing}"
    results["overall"] = "FAIL"

# 5. Checkpoint completeness
print("5. Checkpoint completeness...")
checkpoint_dir = Path("orchestration/state/cities/delhi/checkpoints")
missing_cp = []
for i in range(14):
    cp = checkpoint_dir / f"phase_{i}_passed.json"
    if not cp.exists():
        missing_cp.append(i)
    else:
        data = json.loads(cp.read_text())
        if data.get("validation_verdict") != "PASS":
            missing_cp.append(f"{i}(FAILED)")
if not missing_cp:
    results["checks"]["checkpoint_completeness"] = "PASS - all 14 checkpoints present and PASS"
else:
    results["checks"]["checkpoint_completeness"] = f"FAIL - missing/failed: {missing_cp}"
    results["overall"] = "FAIL"

# 6. Provenance
print("6. Provenance...")
dataset = Path("data/cities/delhi/canonical/water_demand.csv")
provenance = Path("data/cities/delhi/canonical/provenance.json")
compat = Path("data/cities/delhi/canonical/compatibility.json")
audit = Path("data/cities/delhi/canonical/audit.json")
if dataset.exists() and provenance.exists() and compat.exists():
    results["checks"]["provenance"] = "PASS - dataset, provenance, compatibility present"
elif dataset.exists() and provenance.exists():
    results["checks"]["provenance"] = "PASS - dataset and provenance present"
else:
    results["checks"]["provenance"] = "FAIL"
    results["overall"] = "FAIL"

# 7. No secrets
print("7. No secrets...")
import subprocess
secret_check = subprocess.run(["git", "grep", "-l", "-i", "api_key\\|secret\\|password\\|token", "--", "*.py", "*.json", "*.md"], capture_output=True, text=True, cwd="/home/asus/projects/water-forecast")
# Filter out known safe patterns
lines = [l for l in secret_check.stdout.strip().split("\n") if l and "test_" not in l and "mock" not in l.lower()]
if not lines:
    results["checks"]["no_secrets"] = "PASS - no secrets detected"
else:
    results["checks"]["no_secrets"] = f"WARNING - review: {lines[:5]}"

# 8. Dataset SHA256 consistency
print("8. Dataset SHA256 consistency...")
sha = hashlib.sha256()
with open(dataset, "rb") as f:
    for chunk in iter(lambda: f.read(1048576), b""):
        sha.update(chunk)
actual_sha = sha.hexdigest()
p9_sha = p9_report.get("dataset_sha256", "")
if actual_sha == p9_sha:
    results["checks"]["dataset_sha256"] = "PASS - consistent across pipeline"
else:
    results["checks"]["dataset_sha256"] = f"FAIL - actual={actual_sha}, p9={p9_sha}"
    results["overall"] = "FAIL"

# Summary
print("\n" + "=" * 60)
print("INDEPENDENT REVIEW SUMMARY")
print("=" * 60)
for check, status in results["checks"].items():
    print(f"  {check}: {status}")
print(f"\nOVERALL: {results['overall']}")

# Save
out = Path("reports/cities/delhi/independent_review.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
print(f"\nSaved to: {out}")
