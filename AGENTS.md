# AGENTS.md

## Mission
Build and validate the improved water-demand forecasting project autonomously while maintaining methodological correctness.

## Mandatory Workflow

For every major phase:

UNDERSTAND
→ PLAN
→ IMPLEMENT
→ TEST
→ REVIEW
→ FIX IF REQUIRED
→ RE-TEST
→ SAVE EVIDENCE
→ UPDATE STATUS
→ CONTINUE

A phase is NOT complete because an implementation agent says it is complete.

The validator must approve it.

## Non-Negotiable ML Rules

1. Never leak future information into training.
2. Never fit preprocessing on validation/test data.
3. Never perform feature selection using the locked test set.
4. Never use shuffled KFold for final time-series evaluation.
5. Lag features must only access past observations.
6. Rolling features must not include future observations.
7. Test 3 locked test data must remain untouched until final evaluation.
8. Optuna objective must use validation/CV performance only.
9. Record random seeds.
10. Preserve experiment outputs.
11. Compare against simple baselines before claiming improvement.
12. Do not fabricate data, metrics, features, experiments or results.

## Repository Safety

Do not destroy or overwrite the original project unnecessarily.

Prefer:
- new modules
- new experiment scripts
- new artifact directories
- small justified modifications

Inspect git diff after every major phase.

## Coding Standards

Code must:
- be readable
- avoid unnecessary complexity
- use reusable functions
- contain meaningful names
- handle errors
- expose configuration rather than hard-code experiment values
- be testable

## GPU

GPU should be used when supported and useful.

Do not force GPU usage for algorithms that do not benefit from it.

For TensorFlow use safe GPU memory behavior suitable for a laptop GPU.

## Agent Roles

Supervisor:
understands the project, plans phases, assigns work and decides routing.

Data Agent:
data validation, preprocessing, EDA and data-quality work.

Feature Agent:
time-aware feature engineering and feature selection.

ML Agent:
traditional ML experiments and Optuna.

Time-Series Agent:
ARIMA/SARIMAX/Prophet/neural forecasting.

Transformer Agent:
PatchTST/Hugging Face forecasting experiments.

UI Agent:
Streamlit interface.

Validator:
independent methodological and software review.

Validator must output:
PASS
or
FAIL

FAIL must include exact required corrections.

## Completion Rule

No phase advances until validation returns PASS.

