# Multi-city data extension

The extension adapts independently sourced city data to the existing validated
daily forecasting pipeline:

`official raw data -> source adapter -> Date,Consumption -> compatibility gate
-> unchanged validated ML phases -> city-isolated results`

Each registry entry records only verified facts. Bengaluru, Delhi, Gurgaon,
Hyderabad, and Pune are intentionally marked unconfigured until a verified,
compatible source is supplied. Unknown and unavailable sources fail cleanly.
Direct official downloads are preferred, followed by official APIs. Headless
Selenium is an injectable fallback only; it must not bypass authentication,
CAPTCHAs, rate limits, or other access controls.

Raw bytes and canonical CSVs are kept separately. Provenance records the source,
download time, format, transformations, units, aggregation decisions, and SHA256
hashes. Ward or zone observations may be summed to city-day totals. Genuine
additive sub-daily volumes may be summed to daily totals when that meaning is
explicit in the registry.

Monthly, quarterly, annual, and irregular summary targets are rejected as
`DATA_INCOMPATIBLE`. They are never divided across days, repeated, forward-filled,
interpolated, or otherwise made to look daily. The daily methodology is not
weakened to make an incompatible city run.

Before Phase 0, `city_data_compatibility.json` checks the canonical schema,
dates, target, deterministic duplicate handling, frequency semantics,
provenance, hashes, units, history length, split feasibility, 365-day features,
and model requirements. A city that fails the gate stops before ML and cannot
affect another city's artifacts or checkpoints.

London remains the backward-compatible default and its already validated
dataset and methodology are preserved unchanged.
