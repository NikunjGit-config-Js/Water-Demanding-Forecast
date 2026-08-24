# Water Demand Forecasting — Siemens/KIT Hourly Dataset

An interview-oriented, leakage-safe water-demand forecasting project built on **real hourly operational data**, not synthetic data.

## Problem
Forecast one-hour-ahead aggregate water consumption for a mixed residential/commercial/industrial district so a water utility can support pump scheduling and capacity planning.

## Data
- Source: *Hourly Water Demand of a Mixed District Recorded by Supplier*
- Zenodo record: `11045013`
- DOI: `10.5281/zenodo.11045013`
- Canonical series: hourly data from 2016–2021 with Consumption, Temperature, Rain and Sun.
- Synthetic data used: **No**

## Methodology
`Baseline -> Better -> Optimal -> Locked Test`

- Chronological 70/15/15 train/validation/test split.
- Test is locked until preprocessing, model family and hyperparameters are frozen.
- Demand lag features: 1/2/3/24/48/72/168 hours.
- Past-only rolling statistics: 3h, 24h and 168h.
- Weather features are lagged to avoid using target-hour observed weather as if it were a forecast.
- Nominal categories: OneHotEncoder.
- Ordered rain-intensity auxiliary feature: OrdinalEncoder.
- Binary indicators: 0/1.
- Scale-sensitive numeric branch: median imputation -> Yeo-Johnson -> RobustScaler.
- Tree/boosting branch does not claim scaling is necessary.
- Ensembles: Bagging, boosting/XGBoost and VotingRegressor.
- ML validation: TimeSeriesSplit cross-validation.
- ML tuning: Optuna on the shortlisted XGBoost family.
- DL tuning: Keras Tuner on a GPU-backed MLP with EarlyStopping.

## Final result
The model was selected **before** viewing the locked test based on validation performance.

- Selected model: **voting_regressor**
- Locked-test MAE: **158.251**
- Locked-test RMSE: **220.992**
- Locked-test R²: **0.9588**
- Locked-test sMAPE: **8.321%**
- Best reported ensemble-family CV MAE: **154.626 ± 15.005** (xgboost)


## Reproducibility
- `fast_pipeline/deadline_water_workflow.py` — actual experiment implementation.
- `artifacts/deadline/` — metrics, predictions, split metadata, GPU report and tuning history.
- `notebooks/Water_Demand_Forecasting_Final.ipynb` — executed interview/Colab notebook.
- `artifacts/deadline/codex_audit.txt` — final read-only methodology audit when Codex is available.

## Important limitation
This project predicts hourly demand from historical demand/calendar and lagged observed weather. In production, future weather should come from a weather-forecast feed. Results are experimental and are not presented as a deployed utility system.
