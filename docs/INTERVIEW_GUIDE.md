# Interview guide: leakage-safe water-demand forecasting

## Thirty-second summary

This project modernizes a municipal daily water-demand forecasting repository
using 3,800 observations from London, Canada (2009-07-01 through 2020-09-02).
The central engineering problem was not merely fitting more models: it was
making every comparison chronological, preventing preprocessing and feature
selection leakage, preserving reproducible evidence, and keeping a simple
forecast as the benchmark. The result includes traditional ML, classical and
neural time-series models, a PatchTST candidate, a Streamlit dashboard, and an
independently validated Phase 0--12 evidence chain.

## How the methodology avoids leakage

- Every holdout and cross-validation split follows time order.
- Lag and rolling features use only observations before the predicted date.
- Imputation and scaling are fitted only on the current training prefix.
- Phase 3 feature selection uses only the first 70% training prefix.
- Phase 7 tunes and selects on train/validation, then evaluates the locked 15%
  test partition once using the frozen choice.
- Phase 8 nests feature selection and hyperparameter tuning inside each outer
  expanding-window fold. This matters because reusing Phase 3 features in early
  folds would allow later observations to influence those folds.
- Phase 9 recursive holdout forecasts never feed actual holdout targets back
  into later predictions.

## Results worth explaining

The main lesson is that model sophistication did not guarantee a better
forecast. On the chronological 80/20 holdout, naive lag-1 achieved MAE 223.74;
linear regression achieved MAE 251.68 but a lower RMSE (384.63 versus 416.74).
Across five expanding folds, naive lag-1 had the best mean MAE at 230.28. In the
locked Phase 7 test, the frozen linear model again had worse MAE than naive
(259.65 versus 225.35), but lower RMSE (381.63 versus 415.39) and higher R2
(0.99828 versus 0.99797). That is a metric tradeoff, not an unconditional win.

Nested-CV Optuna did not beat the naive forecast on mean MAE. PatchTST also did
not beat naive lag-1 on its validation window (MAE 2,123.66 versus 215.43). The
Phase 9 fixed-origin recursive multi-step experiment is a different, harder
forecasting protocol, so its metrics must not be ranked directly against the
one-step or rolling evaluations. Within that protocol, Prophet had the lowest
descriptive holdout MAE (3,670.84), with no improvement claim.

## Design decisions and tradeoffs

Why use several evaluations? A single 80/20 split is easy to explain but can be
period-specific. Expanding-window CV exposes stability over time. A locked test
supports one final unbiased assessment. Nested CV is more expensive, but is
necessary when both features and hyperparameters are selected.

Why retain naive models? Daily water demand is highly autocorrelated. A lag-1
forecast is cheap, transparent, and difficult to beat on MAE. Without it, a
high R2 from a complex model could look more impressive than it is.

Why was GPU not forced? GPU was permitted for neural models, but the preserved
Phase 9 and Phase 10 runs report CPU execution. Reproducibility and correct
causal evaluation matter more than claiming acceleration that did not occur.

## Limitations and next steps

The supplied data contains only Date and Consumption. No weather, customer,
demographic, or Indian-city data is claimed. Results come from one municipality
and do not establish external validity. R2 is very high partly because demand
has a large level and strong persistence; MAE/RMSE and baseline deltas remain
essential. Future work could add approved weather data, prediction intervals,
seasonal/regime monitoring, and multi-origin horizon-specific evaluation. Any
external scraping or new-city acquisition requires explicit permission.

## Demo path

Run `python -m streamlit run app/app.py`. The dashboard reads preserved
artifacts; it does not retrain models. Historical evaluation controls remain
separate from 1--365 day future forecasts, which use transparent causal naive
methods. This separation prevents a historical prediction from being presented
as a genuine future forecast.
