# Phase 11 dashboard

This Streamlit interface presents preserved, validated experiment artifacts and
generates transparent causal naive forecasts beyond the selected artifact's
final observed date. It does not train or execute fitted models.

From the repository root:

```bash
python -m pip install -r app/requirements.txt
python -m streamlit run app/app.py
```

The evaluation-window control selects trailing historical dates from the chosen
preserved experiment. The separate forecast-horizon control generates 1--365
future calendar days using either the last observation or a recursive seven-day
seasonal naive method. Historical downloads include actuals and residuals;
future downloads intentionally contain only dates, forecasts, and method names.
