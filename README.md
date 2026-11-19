# Water Demand Forecasting — Siemens/KIT Hourly Dataset

A reproducible, leakage-safe water-demand forecasting project built on **real hourly operational data**, not synthetic data.

## Problem
Forecast one-hour-ahead aggregate water consumption for a mixed residential/commercial/industrial district so a water utility can support pump scheduling and capacity planning.

## Data
- Source: *Hourly Water Demand of a Mixed District Recorded by Supplier*
- Zenodo record: `11045013`
- DOI: `10.5281/zenodo.11045013`
- Canonical series: hourly data from 2016–2021 with Consumption, Temperature, Rain and Sun.
- Synthetic data used: **No**

## Methodology
`Data Preparation -> Baseline Modeling -> Ensemble Evaluation -> Hyperparameter Tuning -> Locked-Test Evaluation`

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
- `src/forecasting_pipeline.py` — actual experiment implementation.
- `artifacts/results/` — metrics, predictions, split metadata, GPU report and tuning history.
- `notebooks/model_evaluation_report.ipynb` — executed interview/Colab notebook.
- `artifacts/results/methodology_audit.txt` — optional read-only methodology audit (`--codex-audit`).

**Provenance of the committed artifacts.** They were produced by an earlier copy of this script
(`fast_pipeline/deadline_water_workflow.py`, the file the methodology audit refers to), which wrote to
`artifacts/deadline/`; the folders were renamed to `artifacts/results/` afterwards. The script now
writes directly to `artifacts/results/`. The `orchestration/` logs, `data/preprocessed/all/`
(a daily series from 2009) and `data/cities/` come from earlier pipeline versions and are not used by
this pipeline.

## Run
1. Download the Zenodo record `11045013` archive (`11045013.zip`); the raw data is not committed.
2. `pip install -r requirements.txt` (versions are not pinned; TensorFlow and Keras Tuner are needed for the MLP).
3. From the repository root: `python src/forecasting_pipeline.py . path/to/11045013.zip` (add `--smoke` for a quick code-path check).

If `artifacts/results/` already contains results, the script stops. Pass `--overwrite` to replace them with a full run (this also regenerates this README and the report notebook); `--smoke` runs never overwrite existing results.

## Tuning caveat
Optuna tunes XGBoost with TimeSeriesSplit over train + validation, so the tuned model's validation score has already seen the validation period. The locked test is never used for tuning or selection.

## Important limitation
This project predicts hourly demand from historical demand/calendar and lagged observed weather. In production, future weather should come from a weather-forecast feed. Results are experimental and are not presented as a deployed utility system.
