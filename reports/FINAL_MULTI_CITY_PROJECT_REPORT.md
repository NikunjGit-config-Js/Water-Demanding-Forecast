# Multi-City Water Demand Forecasting: Final Project Report

## Project Status: COMPLETE

## Methodology
Chronological ML pipeline (Phase 0-13) with time-series cross-validation. No Codex dependency. No London artifact reuse.

---

## City Results

### London
- **Source**: Thames Water treated water demand
- **Frequency**: daily
- **Rows**: 3,800 | **Date range**: 2015-01-01 to 2025-03-31
- **Unit**: ML/d
- **Compatibility**: READY
- **Phase 0-13**: COMPLETE
- **Best baseline**: naive_lag_1 (MAE 223.74, Phase 5 holdout)
- **Best conventional ML**: naive_lag_1 (MAE 230.28, Phase 6 5-fold CV)
- **Advanced benchmark best**: Ridge (MAE 329.62, CV); XGBoost did not improve over naive_lag_1

### Delhi
- **Source**: Delhi Jal Board daily production data
- **Frequency**: daily
- **Rows**: 944 | **Date range**: 2018-04-01 to 2022-02-27
- **Unit**: MGD
- **Compatibility**: READY
- **Phase 0-13**: COMPLETE
- **Best baseline**: naive_lag_1 (MAE 16.94, Phase 5 holdout)
- **Best conventional ML**: Ridge (MAE 17.32, Phase 5 holdout)
- **Advanced benchmark best**: naive_lag_1 (MAE 17.54 holdout, 18.27 CV)

### Bengaluru
- **Source**: BWSSB ward-wise water consumption (data.gov.in)
- **Status**: DATA_INCOMPATIBLE
- **Reason**: Ward-wise aggregate snapshots, not daily time-series

### Pune
- **Source**: PMC open data (opencity.in)
- **Status**: DATA_INCOMPATIBLE
- **Reason**: Only 2 isolated months of daily data (Dec 2016, Dec 2017)

### Gurgaon
- **Source**: MCG/GMDA (no public daily dataset)
- **Status**: DATA_SOURCE_REQUIRED
- **Reason**: MCG establishing first SCADA system (2026-2028 timeline)

### Hyderabad
- **Source**: HMWSSB (no daily production data)
- **Status**: DATA_SOURCE_REQUIRED
- **Reason**: Publishes connection counts and tanker records, not daily production

---

## Advanced Model Benchmark (Post-Hoc)

XGBoost, ExtraTrees, and HistGradientBoosting were tested on London and Delhi.

| City | Protocol | naive_lag_1 | Ridge | XGBoost | ExtraTrees | HistGB |
|------|----------|-------------|-------|---------|------------|--------|
| Delhi | Holdout | **17.54** | 19.26 | 27.06 | 30.54 | 23.90 |
| Delhi | 5-fold CV | **18.27** | 23.33 | 21.71 | 19.22 | 21.31 |
| London | Holdout | **223.74** | 242.24 | 537.21 | 582.75 | 562.63 |
| London | 5-fold CV | 230.28 | **329.62** | 592.76 | 741.76 | 592.14 |

**Conclusion**: Advanced boosters do not beat naive lag-1 or Ridge on either city.

---

## Key Findings

1. **Naive lag-1 baseline outperforms all ML and time-series models** on both validated cities.
2. **Ridge is the best conventional ML model** when baseline is excluded.
3. **Advanced boosters provide no improvement** given current data availability.
4. **4 of 6 registered cities lack compatible daily water data.**
5. **Weather, supply infrastructure, and richer exogenous drivers** would be needed for ML to gain advantage.

---

## Reproducing Results

```bash
# Run full pipeline for a compatible city
PYTHONPATH=. python3 scripts/city_pipeline.py --city delhi

# Run advanced benchmark
PYTHONPATH=. python3 experiments/advanced_tree_benchmark.py

# Run tests
python3 -m pytest -q
```

## Repository
- **Repo**: NikunjGit-config-Js/Water-Demanding-Forecast
- **Branch**: feature/india-multicity
- **Tests**: 351 passing
