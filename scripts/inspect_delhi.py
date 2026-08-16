"""Inspect Delhi artifacts to debug Phase 12."""
import json
from pathlib import Path

canonical = Path("data/cities/delhi/canonical/water_demand.csv")
p9_report = Path("artifacts/cities/delhi/phase9/delhi_timeseries/forecast_report.json")
p10_report = Path("artifacts/cities/delhi/phase10/delhi_patchtst/forecast_report.json")

# Canonical dataset
import pandas as pd
df = pd.read_csv(canonical)
print(f"Canonical rows: {len(df)}")
print(f"Canonical date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
print(f"Canonical columns: {list(df.columns)}")

# Phase 9 report
print("\n=== Phase 9 forecast_report.json ===")
r9 = json.loads(p9_report.read_text())
print(json.dumps(r9, indent=2)[:2000])

# Phase 10 report
print("\n=== Phase 10 forecast_report.json ===")
r10 = json.loads(p10_report.read_text())
print(json.dumps(r10, indent=2)[:2000])
