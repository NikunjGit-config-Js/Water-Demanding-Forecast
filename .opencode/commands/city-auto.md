---
description: Run full ML pipeline for a new city (Phase 0-13)
---

# City Auto Pipeline Command

Usage: `/city-auto <city_name>`

## What this command does

Runs the validated London ML methodology on a new city dataset:
1. Creates canonical dataset via data adapter
2. Runs Phase 0-13 experiments (baseline, EDA, features, selection, ML, holdout, CV, Optuna, timeseries, PatchTST, validation)
3. Generates comparison report
4. Commits and pushes all artifacts

## Steps

### 1. Setup
```bash
# Verify city data exists
ls data/cities/$ARGUMENTS/canonical/water_demand.csv

# Check data compatibility
python3 -c "from orchestration.data.pipeline import run_pipeline; run_pipeline('$ARGUMENTS')"
```

### 2. Run Pipeline
```bash
# Execute all phases
PYTHONPATH=. python3 scripts/delhi_pipeline.py  # or adapt for city

# Run validation
PYTHONPATH=. python3 scripts/run_p12_delhi.py
```

### 3. Generate Report
```bash
PYTHONPATH=. python3 scripts/delhi_comparison.py
```

### 4. Commit and Push
```bash
git add artifacts/cities/$ARGUMENTS/ reports/cities/$ARGUMENTS/
git commit -m "feat($ARGUMENTS): complete Phase 0-13 ML pipeline"
git push origin feature/india-multicity
```

## Important Rules

- Never push to upstream (`aildnont/water-forecast`)
- Only push to origin (`NikunjGit-config-Js/Water-Demanding-Forecast`)
- Phase 2 CalendarConfig: `country="IN"`, `subdivision=None`
- Phase 10: set `expected_total_rows` to actual dataset row count
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

## Delhi Results (Reference)

- Dataset: 944 rows, 2018-2022, MGD
- Phase 5 MAE: 16.94 (naive_lag_1)
- Phase 6 MAE: 13.20 (5-fold CV)
- Phase 7 MAE: 18.02 (Optuna)
- Phase 10 MAE: 16.28 (PatchTST)
- 351 tests pass
