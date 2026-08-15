# Validation Rules

The validator must independently inspect every major phase.

## Data Validation
Check:
- dataset exists
- row count
- columns
- types
- chronological ordering
- missing values
- duplicate rows
- invalid dates
- suspicious gaps

## Leakage Validation
Check:
- chronological splitting
- lag calculations
- rolling calculations
- scaling
- encoding
- feature selection
- Optuna
- cross-validation

No information from a later timestamp may influence an earlier training observation.

## Preprocessing Validation
Scaler/transformer must:
fit on training data
then transform validation/test data.

Forbidden:
fitting preprocessing on the full dataset before splitting.

## Test 3 Isolation
Locked test set must not participate in:
- hyperparameter tuning
- feature selection
- threshold/model selection
- preprocessing fitting

## CV Validation
Time-series experiments must use chronological/time-aware CV.

No shuffled folds.

## Metrics
Verify correct implementations of:
- MAE
- MSE
- RMSE
- R2

## Reproducibility
Check:
- random states
- environment/config
- saved outputs
- dataset paths
- experiment identifiers

## Code Validation
Run:
- imports
- tests
- smoke runs
- relevant scripts

Inspect failures.

## Artifact Validation
Verify expected:
- metrics
- predictions
- plots
- model/config
- logs

## Git Validation
Inspect:
git status
git diff

Unexpected or destructive modifications are a FAIL.

## Verdict

Return exactly one final verdict:

PASS

or

FAIL

If FAIL, list required corrections before the workflow can continue.

