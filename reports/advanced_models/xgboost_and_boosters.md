# XGBoost, ExtraTrees, and HistGradientBoosting Post-Hoc Benchmark

This document records the post-hoc benchmark results for three advanced tree-based
models (XGBoost, ExtraTrees, HistGradientBoosting) evaluated on the validated London
and Delhi datasets using the same feature set selected by Phase 4's RFECV procedure.

**Methodological note**: The naive lag-1 baseline here uses true rolling one-step
predictions (each validation observation predicts using the immediately prior actual
observation), consistent with the validated Phase 0 methodology.

## London

### Holdout Protocol

| Model                  | MAE      | RMSE     | R2       |
|------------------------|----------|----------|----------|
| **naive_lag_1** (baseline) | **224.59** | **417.56** | **0.9979** |
| ridge (conventional ML)    | 242.24   | 383.27   | 0.9982   |
| xgboost                    | 537.21   | 1170.46  | 0.9834   |
| extra_trees                | 582.75   | 1174.29  | 0.9833   |
| hist_gradient_boosting     | 562.63   | 1172.02  | 0.9834   |

### CV Protocol (5-fold expanding TimeSeriesSplit)

| Model                  | MAE      | RMSE     | R2       |
|------------------------|----------|----------|----------|
| **naive_lag_1** (baseline) | **234.26** | **703.89** | **0.9929** |
| ridge (conventional ML)    | 329.62   | 633.46   | 0.9945   |
| xgboost                    | 592.76   | 1011.19  | 0.9872   |
| hist_gradient_boosting     | 592.14   | 1009.02  | 0.9870   |
| extra_trees                | 741.76   | 1107.23  | 0.9835   |

### Best Advanced Model per Protocol (London)

- **Holdout**: XGBoost (MAE 537.21)
- **CV**: HistGradientBoosting (MAE 592.14)

Neither advanced model beats naive_lag_1 or Ridge under either protocol.

## Delhi

### Holdout Protocol

| Model                  | MAE     | RMSE    | R2       |
|------------------------|---------|---------|----------|
| **naive_lag_1** (baseline) | **17.54** | **30.99** | **0.3282** |
| ridge (conventional ML)    | 19.26   | 29.69   | 0.3833   |
| hist_gradient_boosting     | 23.90   | 33.08   | 0.2344   |
| xgboost                    | 27.06   | 35.88   | 0.0995   |
| extra_trees                | 30.54   | 37.95   | -0.0076  |

### CV Protocol (5-fold expanding TimeSeriesSplit)

| Model                  | MAE     | RMSE    | R2       |
|------------------------|---------|---------|----------|
| **naive_lag_1** (baseline) | **13.93** | **26.43** | **0.0204** |
| extra_trees                | 19.22   | 27.35   | 0.0858   |
| hist_gradient_boosting     | 21.31   | 29.96   | -0.1719  |
| xgboost                    | 21.71   | 31.18   | -0.1869  |
| ridge (conventional ML)    | 23.33   | 33.62   | -1.2067  |

### Best Advanced Model per Protocol (Delhi)

- **Holdout**: HistGradientBoosting (MAE 23.90)
- **CV**: ExtraTrees (MAE 19.22)

Neither advanced model beats naive_lag_1 under either protocol.

## Why Advanced Models Lose to Naive Lag-1

Under this configured post-hoc benchmark, all three advanced tree-based models
consistently underperform naive lag-1 on both cities and both protocols. Key
factors contributing to this outcome:

1. **Strong temporal persistence**: Daily water demand exhibits high autocorrelation.
   The previous day's consumption is an extremely strong predictor. Complex models
   cannot easily improve on this natural persistence without exogenous signals.

2. **Limited exogenous variables**: The feature set is entirely target-derived
   (lags, rolling statistics, calendar features). Without weather, population,
   or operational data, the models lack the external information needed to
   correct lag-1 errors.

3. **Small Delhi dataset**: With only 944 daily observations, advanced models
   have insufficient training data to learn generalizable patterns. This leads
   to overfitting, especially for ExtraTrees (negative holdout R2).

4. **Bias-variance tradeoff**: Tree ensembles introduce variance that naive
   lag-1 avoids entirely. The simplicity of "predict yesterday's value" has
   low variance, which matters greatly in time-series where distributions shift.

5. **Feature engineering ceiling**: The 20 selected features are all derived
   from the target variable itself. When the best feature is literally
   `y[t-1]`, additional transformations yield diminishing returns.

These results do not imply that XGBoost or other gradient boosting methods can
never outperform naive baselines on water demand forecasting. With richer
exogenous features (weather, events, operational data), longer training windows,
and proper hyperparameter tuning, advanced models may provide meaningful gains.

## Conclusion

Naive lag-1 remains the most robust model for both London and Delhi under the
validated methodology. Ridge is the best conventional ML for London (holdout
MAE 242.24) and Delhi (holdout MAE 19.26). Advanced tree-based models do not
provide improvement under this post-hoc benchmark configuration.
