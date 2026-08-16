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

## Primary Validated Results (Phase 5)

### London

| Category | Model | Metric | Value |
|----------|-------|--------|-------|
| Best Baseline | naive_lag_1 | MAE | 223.73671344736826 |
| Best Conventional ML | linear_regression | MAE | 251.6803039164544 |
| Best Conventional ML | linear_regression | RMSE | 384.631787405011 |
| Best Conventional ML | linear_regression | R2 | 0.997933150342045 |

### Delhi

| Category | Model | Metric | Value |
|----------|-------|--------|-------|
| Best Baseline | naive_lag_1 | MAE | 16.938412698412705 |
| Best Conventional ML | ridge | MAE | 17.31673702015551 |
| Best Conventional ML | ridge | RMSE | 28.6057446701141 |
| Best Conventional ML | ridge | R2 | 0.5150267559002881 |

---

## Post-Hoc Advanced Benchmark

XGBoost, ExtraTrees, and HistGradientBoosting evaluated using the validated feature set.

| City | Protocol | Best Advanced | MAE |
|------|----------|---------------|-----|
| London | holdout | xgboost | 537.21 |
| London | CV | hist_gradient_boosting | 592.14 |
| Delhi | holdout | hist_gradient_boosting | 23.90 |
| Delhi | CV | extra_trees | 19.22 |

Advanced models do not beat naive_lag_1 or the best conventional ML under either protocol.

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
| Benchmark fix | fix: correct advanced benchmark protocol and final reports | `6937656` |
| Docs cleanup | docs: separate validated results from post-hoc benchmark | `latest` |
