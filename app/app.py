"""Streamlit entry point for the validated water-demand result dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase11_dashboard import (  # noqa: E402
    EXPERIMENTS,
    FORECAST_METHODS,
    METRIC_NAMES,
    causal_naive_forecast,
    downloadable_forecast,
    downloadable_predictions,
    downloadable_workbook,
    experiment_by_label,
    load_predictions,
    metric_table,
    trailing_window,
)


st.set_page_config(page_title="Water demand forecast results", page_icon="💧", layout="wide")
st.title("Water demand forecast results")
st.caption("London, Canada · preserved evaluations and causal naive future forecasts")

with st.sidebar:
    st.header("View controls")
    experiment_label = st.selectbox(
        "Evaluation experiment",
        [item.label for item in EXPERIMENTS],
        help="Only final outputs from independently approved phases are listed.",
    )
    spec = experiment_by_label(experiment_label)

try:
    predictions = load_predictions(spec)
except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as exc:
    st.error(f"The selected preserved artifact could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    model = st.selectbox("Model", spec.model_columns)
    available_days = int(predictions["Date"].nunique())
    evaluation_days = st.slider(
        "Evaluation window (days)",
        min_value=1,
        max_value=available_days,
        value=min(90, available_days),
        help="Shows the trailing dates already preserved in this experiment; it does not create a future forecast.",
    )
    chosen_metric = st.selectbox("Highlighted metric", METRIC_NAMES)
    plot_type = st.radio(
        "Plot",
        ("Actual vs predicted", "Residual over time", "Residual distribution", "Actual vs predicted scatter"),
    )

window = trailing_window(predictions, evaluation_days)
metrics = metric_table(window, spec.model_columns)
selected_metrics = metrics.set_index("model").loc[model]

st.subheader(model.replace("_", " ").title())
st.caption(
    f"{spec.evaluation} · {window['Date'].min():%Y-%m-%d} to {window['Date'].max():%Y-%m-%d} "
    f"· {len(window):,} prediction rows"
)
metric_columns = st.columns(4)
for container, metric_name in zip(metric_columns, METRIC_NAMES):
    value = selected_metrics[metric_name]
    formatted = f"{value:,.4f}" if metric_name == "R2" else f"{value:,.2f}"
    container.metric(metric_name, formatted)

plot_frame = window.loc[:, ["Date", "actual", model]].rename(columns={model: "predicted"}).copy()
plot_frame["residual"] = plot_frame["actual"] - plot_frame["predicted"]

if plot_type == "Actual vs predicted":
    st.line_chart(plot_frame.set_index("Date")[["actual", "predicted"]], color=["#2563EB", "#D97706"])
elif plot_type == "Residual over time":
    st.line_chart(plot_frame.set_index("Date")[["residual"]], color=["#7C3AED"])
elif plot_type == "Residual distribution":
    counts = pd.cut(plot_frame["residual"], bins=min(30, max(5, len(plot_frame) // 5)))
    histogram = counts.value_counts(sort=False).rename_axis("residual range").rename("count")
    st.bar_chart(histogram, color="#0F766E")
else:
    st.scatter_chart(plot_frame, x="actual", y="predicted", color="#D97706")

st.subheader("Model comparison")
st.caption(f"Sorted by {chosen_metric}; metrics are recomputed from the selected preserved rows.")
ascending = chosen_metric != "R2"
display_metrics = metrics.sort_values(chosen_metric, ascending=ascending).reset_index(drop=True)
st.dataframe(
    display_metrics.style.format({"MAE": "{:,.2f}", "MSE": "{:,.2f}", "RMSE": "{:,.2f}", "R2": "{:.4f}"}),
    hide_index=True,
    use_container_width=True,
)

st.subheader("Downloads")
download_columns = st.columns(2)
download_columns[0].download_button(
    "Download selected predictions (CSV)",
    downloadable_predictions(window, model),
    file_name=f"{spec.phase.lower().replace(' ', '_')}_{model}_{evaluation_days}d_evaluation_predictions.csv",
    mime="text/csv",
)
download_columns[1].download_button(
    "Download displayed results (Excel)",
    downloadable_workbook(window, spec.model_columns),
    file_name=f"{spec.phase.lower().replace(' ', '_')}_{evaluation_days}d_evaluation_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.divider()
st.header("Future forecast")
st.caption(
    "Forecasts begin after the final observed date and use only target observations available at that origin. "
    "They are not scored against unknown future actual values."
)
forecast_controls = st.columns(2)
with forecast_controls[0]:
    forecast_method = st.selectbox("Forecast method", tuple(FORECAST_METHODS))
with forecast_controls[1]:
    forecast_horizon = st.slider(
        "Forecast horizon (days)",
        min_value=1,
        max_value=365,
        value=30,
        help="Number of calendar days to forecast beyond the final observed date.",
    )

future = causal_naive_forecast(predictions, forecast_horizon, forecast_method)
st.line_chart(future.set_index("Date")[["forecast"]], color=["#0F766E"])
st.caption(
    f"{forecast_method} · forecast origin {predictions['Date'].max():%Y-%m-%d} · "
    f"{future['Date'].min():%Y-%m-%d} to {future['Date'].max():%Y-%m-%d}"
)
st.download_button(
    "Download future forecast (CSV)",
    downloadable_forecast(future),
    file_name=f"causal_{forecast_method.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{forecast_horizon}d_future_forecast.csv",
    mime="text/csv",
)

with st.expander("About these results"):
    st.write(
        "Historical evaluation reads preserved experiment predictions only. Its evaluation-window control is separate from "
        "the future forecast horizon. Future values are generated recursively with a last-observation or seven-day seasonal "
        "naive method using only targets at or before the displayed forecast origin; future actuals are never loaded or "
        "exported. Phase 8 invalid/interim attempts and the failed Phase 9 run are excluded. Residual is actual minus predicted."
    )
