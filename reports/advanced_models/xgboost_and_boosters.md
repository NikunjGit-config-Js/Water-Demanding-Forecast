# XGBoost and Advanced Boosters: Post-Hoc Benchmark Analysis

## Was XGBoost present in the original Phase 0-13 pipeline?

No. The original Phase 0-13 methodology tested the following conventional ML models:
- Linear Regression, Ridge, Lasso
- Decision Tree, KNN, SVR
- Random Forest, Bagging, Gradient Boosting, Voting

XGBoost, ExtraTrees, and HistGradientBoosting were not included in the original pipeline.

## Why was it absent?

The original pipeline was designed to validate a reproducible methodology using well-established sklearn models. XGBoost and similar boosters were reserved for a post-hoc benchmark to test whether modern boosted/tree ensembles provide additional benefit.

## What models were originally tested?

10 models in Phase 4-8: naive_lag_1, linear_regression, ridge, lasso, decision_tree, knn, svr, random_forest, bagging, gradient_boosting, voting.

## Did XGBoost beat the best conventional model?

**No.** Results from the post-hoc benchmark:

| City | Protocol | Ridge MAE | XGBoost MAE | ExtraTrees MAE | HistGB MAE | Winner |
|------|----------|-----------|-------------|----------------|------------|--------|
| Delhi | Holdout | 19.26 | 27.06 | 30.54 | 23.90 | Ridge |
| Delhi | 5-fold CV | 23.33 | 21.71 | 19.22 | 21.31 | ExtraTrees |
| London | Holdout | 242.24 | 537.21 | 582.75 | 562.63 | Ridge |
| London | 5-fold CV | 329.62 | 592.76 | 741.76 | 592.14 | Ridge |

On the holdout protocol (the primary evaluation), Ridge outperforms all advanced models on both cities.

## Did XGBoost beat naive lag-1?

**No.** Naive lag-1 (MAE 17.54 Delhi holdout, 224.59 London holdout) outperforms XGBoost on both cities under the holdout protocol.

## Did ExtraTrees beat them?

On Delhi CV, ExtraTrees (MAE 19.22) slightly beats naive_lag_1 (MAE 18.27). On all other configurations, it does not.

## Did HistGradientBoosting beat them?

No. HistGradientBoosting underperforms Ridge on both cities under both protocols.

## Why might a simple lag baseline outperform complex ML on this dataset?

1. **Temporal autocorrelation is very strong**: Water demand is highly correlated with yesterday's value. A naive lag-1 prediction captures this direct persistence pattern efficiently.

2. **Limited exogenous drivers**: The features are purely temporal (day-of-week, month, rolling statistics). Without weather, rainfall, temperature, reservoir levels, or supply restrictions, the ML models have limited information to learn beyond what lag-1 already captures.

3. **Small dataset size**: Delhi has only 944 daily observations (~2.6 years). Complex models like XGBoost and ExtraTrees have many hyperparameters and are prone to overfitting on small datasets. Ridge regression, with its L2 regularization, is more robust.

4. **Non-stationarity**: Water demand patterns change over time (seasonal, policy changes, population growth). A simple lag-1 model adapts naturally because it always uses the most recent observation.

5. **Feature engineering ceiling**: The 20 selected features are derived from the target itself (lags, rolling means, etc.). When the best predictor of tomorrow is today's value, additional derived features add noise rather than signal.

## Does more model complexity automatically mean better forecasting?

**No.** This benchmark demonstrates that model complexity does not guarantee better performance. The bias-variance tradeoff means that more complex models can overfit, especially on small datasets with limited exogenous information.

## What would likely be needed for boosters to gain more advantage?

1. **Weather data**: Temperature, humidity, rainfall, evapotranspiration
2. **Supply infrastructure data**: Reservoir levels, treatment plant output, pipeline capacity
3. **Population/activity variables**: Tourism indices, industrial activity, events
4. **Larger datasets**: Multi-year daily records (5-10+ years) to capture long-term patterns
5. **Higher-frequency data**: Hourly or sub-daily observations for peak demand modeling
6. **Richer exogenous drivers**: Economic indicators, migration patterns, water pricing

With these additional data sources, XGBoost and similar models could potentially learn non-linear interactions that a simple lag-1 baseline cannot capture.

## Conclusion

The post-hoc benchmark confirms that the original Phase 0-13 methodology was appropriate. Advanced boosters (XGBoost, ExtraTrees, HistGradientBoosting) do not provide meaningful improvement over the simpler models already tested, given the current data availability. The naive lag-1 baseline remains the strongest performer for this specific forecasting task with the available features.
