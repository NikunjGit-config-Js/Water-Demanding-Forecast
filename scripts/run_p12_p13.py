"""Run Phase 12 and 13 for Delhi."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.delhi_pipeline import phase12, phase13

phase9_dir = Path("artifacts/cities/delhi/phase9/delhi_timeseries")
phase10_dir = Path("artifacts/cities/delhi/phase10/delhi_patchtst")

try:
    phase12(phase9_dir, phase10_dir)
    print("Phase 12: PASS")
except Exception as e:
    print(f"Phase 12: FAIL - {e}")
    import traceback; traceback.print_exc()

try:
    phase13()
    print("Phase 13: PASS")
except Exception as e:
    print(f"Phase 13: FAIL - {e}")
    import traceback; traceback.print_exc()
