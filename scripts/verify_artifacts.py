"""Verify all Delhi artifacts exist."""
from pathlib import Path

artifact_root = Path("artifacts/cities/delhi")
for phase in range(11):
    phase_dir = artifact_root / f"phase{phase}"
    if phase_dir.exists():
        subdirs = [d for d in phase_dir.iterdir() if d.is_dir()]
        if subdirs:
            files = list(subdirs[0].iterdir())
            print(f"Phase {phase}: {subdirs[0].name} ({len(files)} files)")
        else:
            print(f"Phase {phase}: EMPTY DIR")
    else:
        print(f"Phase {phase}: NO DIR")

# Check reports
report_dir = Path("reports/cities/delhi")
if report_dir.exists():
    reports = list(report_dir.rglob("*.json"))
    print(f"\nReports: {len(reports)} JSON files")
    for r in reports:
        print(f"  {r.relative_to(report_dir)}")
else:
    print("\nNo reports directory")
