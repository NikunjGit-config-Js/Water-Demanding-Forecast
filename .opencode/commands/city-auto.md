---
description: Run full ML pipeline for a new city (Phase 0-13)
---

# City Auto Pipeline Command

Usage: `/city-auto <city>`

## What this command does

Runs the validated ML methodology on any compatible city dataset:
1. Verifies canonical dataset exists and is compatible
2. Runs Phase 0-13 experiments (baseline, EDA, features, selection, ML, holdout, CV, Optuna, timeseries, PatchTST, validation)
3. Generates city report
4. Commits and pushes all artifacts

## Steps

### 1. Verify Data Compatibility
```bash
# Check canonical dataset exists
ls data/cities/<city>/canonical/water_demand.csv

# Check compatibility report
cat data/cities/<city>/canonical/compatibility.json | python3 -m json.tool
```

If status is not READY, do NOT run pipeline. Document the incompatibility instead.

### 2. Run Pipeline
```bash
PYTHONPATH=. python3 scripts/city_pipeline.py --city <city>
```

### 3. Generate City Report
```bash
# Read Phase 5 metrics for best conventional ML
cat artifacts/cities/<city>/phase5/*/metrics.csv

# Read Phase 6 for CV results
cat artifacts/cities/<city>/phase6/*/metrics_summary.csv
```

### 4. Commit and Push
```bash
git add artifacts/cities/<city>/ reports/cities/<city>/
git commit -m "feat(<city>): complete Phase 0-13 ML pipeline"
git push origin feature/india-multicity
```

## Important Rules

- Never push to upstream (`aildnont/water-forecast`)
- Only push to origin (`NikunjGit-config-Js/Water-Demanding-Forecast`)
- Phase 2 CalendarConfig uses city-specific holidays (see `CITY_CALENDAR_CONFIG` in `scripts/city_pipeline.py`)
- Phase 10: `expected_total_rows` is set automatically from dataset
- Always run `python3 -m pytest tests/ -q` after changes
- Always run `git diff --check` before commit

## Phase Dependencies

- Phase 0-1: raw dataset
- Phase 2: raw dataset + CalendarConfig
- Phase 3-8: depend on Phase 2 output
- Phase 4-8: depend on Phase 3 output
- Phase 9-10: use raw dataset directly
- Phase 12: requires checkpoints from Phase 0-11
- Phase 13: requires Phase 12 checkpoint

## Incompatible Cities

If a city's daily data is not available:

- Bengaluru: DATA_INCOMPATIBLE (monthly BWSSB records only)
- Pune: DATA_INCOMPATIBLE (monthly averages, not daily observations)
- Gurgaon: DATA_SOURCE_REQUIRED (no verified daily dataset)
- Hyderabad: DATA_SOURCE_REQUIRED (no verified daily dataset)

Document the status in `reports/cities/<city>/source_audit.json`. Do NOT fabricate daily data.
