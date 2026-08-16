"""Verify all Delhi checkpoints."""
import json
from pathlib import Path

checkpoint_dir = Path("orchestration/state/cities/delhi/checkpoints")
results = {}
for i in range(14):
    path = checkpoint_dir / f"phase_{i}_passed.json"
    if path.exists():
        data = json.loads(path.read_text())
        verdict = data.get("validation_verdict", "MISSING")
        results[i] = verdict
        print(f"Phase {i}: {verdict}")
    else:
        results[i] = "NOT_FOUND"
        print(f"Phase {i}: NOT_FOUND")

passed = sum(1 for v in results.values() if v == "PASS")
print(f"\n{passed}/14 phases PASS")
if passed == 14:
    print("ALL CHECKPOINTS VALID")
else:
    print("SOME CHECKPOINTS MISSING OR FAILED")
    for i, v in results.items():
        if v != "PASS":
            print(f"  Phase {i}: {v}")
