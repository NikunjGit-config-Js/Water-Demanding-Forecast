# Water Demand Forecasting — Improved Agentic ML Project

## Goal
Improve the original water-forecast repository into a modern, validated, reproducible water-demand forecasting system while preserving the original project as a baseline.

## Primary Dataset
Initially use the supplied London, Canada preprocessed dataset:
- Date
- Consumption

Current sample:
- 3800 rows
- 2009-07-01 to 2020-09-02
- No missing values
- No duplicates

Do not claim customer/weather/demographic features exist unless they are actually acquired.

## Optional Future Dataset Extension
Only after the London pipeline is complete:
- Delhi
- Gurgaon
- Bengaluru
- Hyderabad
- Mumbai
- Pune

Potential Selenium/web-data ingestion must NOT start without explicit user approval.

## Pipeline

### Phase 0
Repository audit and original baseline reproduction.

### Phase 1
Data validation and EDA:
- shape/dtypes
- missing values
- duplicates
- descriptive statistics
- target distribution
- time-series plot
- univariate analysis
- box plots
- outlier inspection
- correlation analysis where multiple numeric features exist
- seasonality/trend inspection

### Phase 2
Physics/domain/time-aware feature engineering:
- lag 1
- lag 7
- lag 14
- lag 30
- lag 365
- rolling mean 7/30
- rolling median 7/30
- rolling std 7/30
- rolling min/max
- exponential moving averages
- day-over-day change
- 7-day growth
- rolling slope
- day of week
- month
- weekend
- season
- holiday/festival where valid
- cyclical sin/cos features
- Fourier weekly/yearly terms
- same weekday previous week
- same period previous year

Weather/demographic interactions may only be added if such data is actually available.

All lag/rolling features must use past data only.

### Phase 3
Feature selection:
- leakage screening
- correlation/redundancy analysis
- model importance
- permutation importance
- stability across time splits
- domain sanity checks

Feature selection must be learned from training data only.

### Phase 4
Traditional ML:
- Linear Regression
- Ridge
- Lasso
- Decision Tree Regressor
- KNN Regressor
- SVR
- Random Forest Regressor
- Bagging Regressor
- AdaBoost / Gradient Boosting
- Voting Regressor

Do not add unnecessary models without justification.

### Phase 5 — Test 1
Chronological 80/20 split.

NO random train_test_split for time-series evaluation.

### Phase 6 — Test 2
Time-aware 5-fold cross-validation using TimeSeriesSplit or equivalent expanding-window validation.

### Phase 7 — Test 3
Chronological:
- 70% train
- 15% validation
- 15% locked test

Optuna tunes on train/validation only.
Locked test must not influence tuning.

### Phase 8 — Test 4
Time-series cross-validation + Optuna.

### Phase 9
Original time-series models:
- ARIMA
- SARIMAX
- Prophet
- LSTM
- GRU
- 1D CNN

Use these as comparative baselines where practical.

### Phase 10
Modern forecasting:
- PatchTST first candidate
- optionally Hugging Face TimeSeriesTransformer

Transformer must demonstrate value; do not assume it is automatically superior.

### Phase 11
Streamlit dashboard:
- simple low-eye-strain UI
- model selection
- forecast horizon
- actual vs predicted plots
- metrics
- residual plots
- downloadable predictions/results

### Phase 12
Full validation.

### Phase 13
Documentation, README, interview explanation and final artifacts.

## Metrics
Primary regression metrics:
- MAE
- MSE
- RMSE
- R2

Adjusted R2 may be reported when meaningful.

## Scaling
Use RobustScaler where scaling benefits the model:
- Linear/Ridge/Lasso
- KNN
- SVR
- neural models when appropriate

Fit scaler ONLY on training data.

Tree-based models generally do not require scaling.

## GPU Policy
Use GPU by default where supported:
- TensorFlow
- LSTM
- GRU
- CNN
- Transformer

Traditional sklearn models may remain CPU-based.

Current verified environment:
- TensorFlow 2.20
- NVIDIA GPU available through WSL

## Artifact Policy
Every experiment must preserve:
- configuration
- split information
- features
- metrics
- predictions
- plots
- model when appropriate
- random seeds
- logs

