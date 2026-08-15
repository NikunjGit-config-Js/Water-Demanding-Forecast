# Experiment Plan

## Experiment 0
Reproduce or closely reproduce the original repository baseline.

Purpose:
establish reference behavior before improvements.

## Experiment 1
Chronological 80/20 holdout.

Models:
traditional ML baseline set.

Metrics:
MAE, MSE, RMSE, R2.

## Experiment 2
5-fold time-aware cross-validation.

Use expanding/TimeSeriesSplit validation.

Report:
- each fold
- mean
- standard deviation

## Experiment 3
Chronological 70/15/15.

Train:
model fitting.

Validation:
Optuna optimization and model selection.

Test:
locked until the final chosen configuration.

## Experiment 4
TimeSeriesSplit + Optuna.

Optimization must evaluate hyperparameters using time-aware CV.

## Forecasting Comparison

Compare:
- naive/reference forecast
- Linear Regression
- best traditional ML model
- ARIMA
- SARIMAX
- Prophet
- LSTM
- GRU
- CNN
- Transformer if completed

## Evaluation Artifacts

Generate:
- actual vs predicted over time
- actual vs predicted scatter
- residual vs predicted
- residual distribution
- error over time
- highest-error dates
- metrics table

Large residuals may be called prediction-error/anomaly candidates.
Do not call them confirmed anomalies unless an anomaly-detection method is implemented.

