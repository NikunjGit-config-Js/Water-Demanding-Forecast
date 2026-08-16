# Multi-City Water Demand Forecasting: Final Project Report

## Project Status: COMPLETE

**Methodology**: Chronological ML pipeline (Phase 0-13) with time-series cross-validation

**Dataset**: London (Thames Water treated water demand, 3800 daily rows), Delhi (Delhi Jal Board daily production, 944 rows)

**Pipeline**: `orchestration/auto.py`

---

## City Data Status

| City | Source | Status | Reason |
|------|--------|--------|--------|
| London | Thames Water daily demand | READY | 3800 daily rows |
| Delhi | Delhi Jal Board daily production | READY | 944 daily rows |
| Bengaluru | BWSSB ward-wise aggregates | DATA_INCOMPATIBLE | Not daily time-series |
| Pune | PMC partial months | DATA_INCOMPATIBLE | Only 2 months available |
| Gurgaon | MCG/GMDA | DATA_SOURCE_REQUIRED | No public daily dataset |
| Hyderabad | HMWSSB connections data | DATA_SOURCE_REQUIRED | No daily production data |

---

## Validated Results

### London

| Category | Model | Metric | Value |
|----------|-------|--------|-------|
| Best Baseline | naive_lag_1 | Phase 5 holdout MAE | 224.59 |
| Best Conventional ML | linear_regression | Phase 5 holdout MAE | 251.68 |
| Best Advanced (holdout) | xgboost | holdout MAE | 537.21 |
| Best Advanced (CV) | hist_gradient_boosting | CV MAE | 592.14 |

Advanced models do not beat naive_lag_1 or linear_regression under either protocol.

### Delhi

| Category | Model | Metric | Value |
|----------|-------|--------|-------|
| Best Baseline | naive_lag_1 | Phase 5 holdout MAE | 17.54 |
| Best Conventional ML | ridge | Phase 5 holdout MAE | 19.26 |
| Best Advanced (holdout) | hist_gradient_boosting | holdout MAE | 23.90 |
| Best Advanced (CV) | extra_trees | CV MAE | 19.22 |

Advanced models do not beat naive_lag_1 under either protocol.

---

## Phase Completion

### London: Phases 0-13 COMPLETE

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Split manifest | PASS |
| 1 | Data quality | PASS |
| 2 | Feature engineering | PASS |
| 3 | Feature selection (RFECV) | PASS |
| 4 | Traditional ML baseline | PASS |
| 5 | Chronological holdout | PASS |
| 6 | Time-series CV | PASS |
| 7 | Locked test Optuna | PASS |
| 8 | CV Optuna | PASS |
| 9 | Classical/neural baselines | PASS |
| 10 | PatchTST | PASS |
| 11 | Executive summary | PASS |
| 12 | Validator | PASS |
| 13 | Documentation | PASS |

### Delhi: Phases 0-13 COMPLETE

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Split manifest | PASS |
| 1 | Data quality | PASS |
| 2 | Feature engineering | PASS |
| 3 | Feature selection (RFECV) | PASS |
| 4 | Traditional ML baseline | PASS |
| 5 | Chronological holdout | PASS |
| 6 | Time-series CV | PASS |
| 7 | Locked test Optuna | PASS |
| 8 | CV Optuna | PASS |
| 9 | Classical/neural baselines | PASS |
| 10 | PatchTST | PASS |
| 11 | Executive summary | PASS |
| 12 | Validator | PASS |
| 13 | Documentation | PASS |

---

## Advanced Tree Benchmark (Post-Hoc)

Ran XGBoost, ExtraTrees, HistGradientBoosting on both cities using the validated feature set.

**Conclusion**: Naive lag-1 is the most robust model for both cities. Advanced tree-based models do not improve over this baseline under the current benchmark configuration.

---

## Source Provenance

| City | Status | Source |
|------|--------|--------|
| London | VALIDATED | Thames Water |
| Delhi | VALIDATED | Delhi Jal Board |
| Bengaluru | DATA_INCOMPATIBLE | BWSSB |
| Pune | DATA_INCOMPATIBLE | PMC |
| Gurgaon | DATA_SOURCE_REQUIRED | MCG/GMDA |
| Hyderabad | DATA_SOURCE_REQUIRED | HMWSSB |

---

## Reproduce

```bash
# Full pipeline for any compatible city
PYTHONPATH=. python3 scripts/city_pipeline.py --city <city>

# Advanced model benchmark
PYTHONPATH=. python3 experiments/advanced_tree_benchmark.py
```

---

## Final Commits

| Commit | Message | SHA |
|--------|---------|-----|
| Phase 0-13 London complete | feat: complete validated London water forecasting pipeline | `08b04fd` |
| Phase 0-13 Delhi complete | feat: complete Delhi water forecasting pipeline (Phases 0-13) | `e48a323` |
| Multi-city framework | feat: finalize validated multi-city water forecasting study | `3e7ee99` |
| Benchmark fix | fix: correct advanced benchmark protocol and final reports | `latest` |
