# Project Status

Current Phase: Phase 11 repair attempt 2 implemented and tested; awaiting independent validation

## Completed
- Repository cloned
- Working branch created: agentic-water-forecast
- WSL configured
- NVIDIA GPU detected
- TensorFlow GPU verified
- Python 3.12 environment created
- Core dependencies installed
- Example dataset validated
- Agentic directory structure created
- Codex CLI wrapper implemented for read-only and explicitly writable execution
- Initial automation smoke test completed and preserved in `orchestration/state/`
- Validator FAIL reviewed; all required wrapper corrections implemented
- Read-only Codex responses moved to secure temporary storage outside the repository
- Persistent state/log output restricted to explicit `workspace-write` executions
- Strict sandbox and output filename validation added
- Codex CLI 0.147.0 automatic-write invocation corrected to use `--approve-for-me`
  without an explicit `-s workspace-write`
- Focused orchestration tests added (14 passing)
- Full pytest discovery, AST syntax parsing, and orchestration import checks pass
- Real post-correction Codex CLI read-only smoke completed with unchanged Git
  status and unchanged path/size/mtime/SHA-256 snapshots for 48,745 repository
  files; the external temporary response was removed after being read
- Post-correction smoke evidence preserved separately by the supervisor at
  `orchestration/state/post_correction_read_only_smoke.txt`
- Deterministic CrewAI 1.15.16 Flow orchestration implemented with typed state,
  explicit specialist routing, bounded repair attempts, test gating, strict
  validator parsing, local PASS checkpoints, and fail-closed terminal states
- Specialist and task profiles added for all approved orchestration roles
- Read-only Git/evidence tools and allowlisted non-shell test execution added
- Mocked Flow, validator-isolation, sandbox, role-routing, retry, malformed-output,
  and evidence-path tests added
- AST parsing, bytecode compilation, and orchestration import smoke checks pass
- Independent orchestration validator returned FAIL; Phase 0 remained unstarted
- Orchestration-only exact dependency pins added for CrewAI 1.15.16,
  PyYAML 6.0.3, Pydantic 2.12.5, and pytest 9.1.1, with isolated `uv`
  installation instructions; the historical root requirements remain unchanged
- Phase requests now fail closed unless they are unique, strictly chronological,
  and consecutive; every prerequisite must have either passed earlier in the
  same Flow run or have a strict, content-validated persisted PASS checkpoint
- Checkpoint records now use a dedicated strict schema and are not trusted by
  filename; missing, malformed, mismatched, or non-PASS prerequisites stop work
- Evidence reads now remain inside the resolved selected approved root after
  symlink resolution
- The normal test gate accepts only the exact repository command
  `python -m pytest -q`; Python snippets/scripts, shells, pytest mutations and
  plugins, aliases, and all other command shapes are rejected
- Remediation tests cover sequential/resumed execution, skipped/reordered/
  duplicate phases, invalid prerequisite checkpoints, symlink escapes, and
  adversarial test commands; focused discovery reports 67 passing tests
- Persisted prerequisite checkpoints now use strict schema version 2 and contain
  typed test evidence instead of a free-form test report. The evidence accepts
  only command `['python', '-m', 'pytest', '-q']` with integer return code `0`.
- Live and persisted validation now share the same exact, case-sensitive first-
  line verdict check. A persisted checkpoint is accepted only when its verdict
  and validation report both begin with exact `PASS`; contradictory, malformed,
  legacy, missing, and extra unsafe evidence fails closed.
- Checkpoint-evidence adversarial coverage and supervisor verification pass with
  89 tests. New evidence is preserved at
  `orchestration/state/checkpoint_evidence_hardening_20260815T195117Z.txt`.
- Live independent validation now requests the raw Codex response and validates
  it before stripping or normalization, using the same strict report parser as
  persisted checkpoints. Independent read-only revalidation remains pending.
- Raw-response adversarial and wrapper-regression coverage passes in full
  discovery (122 tests). Supervisor evidence is preserved at
  `orchestration/state/raw_validator_response_hardening_20260815T195910Z.txt`.
- Phase 0 supervisor attempt 1 audited the repository and supplied London
  dataset, repaired the removed pandas `DataFrame.append` call with the
  equivalent `pd.concat`, and reproduced the original linear-regression
  baseline using the existing chronological 90/10 split. Full discovery remains
  at 122 passing tests. Evidence is preserved at
  `orchestration/state/phase_0_attempt_1_20260815T200921Z.txt`.
- Phase 0 supervisor attempt 2 added a dedicated reproducible baseline runner
  and preserved configuration, split manifest, structured metrics, dated
  predictions and errors, an actual-versus-predicted plot, fitted model,
  training-only scaler, execution log, environment details, dataset SHA-256,
  and an explicit deterministic/no-seed declaration under
  `artifacts/phase0/phase0_attempt_2_20260815/`. Historical metrics were
  reproduced, and project-policy R2 was added as a separately labeled
  supplemental metric. Focused tests and full discovery pass (124 tests).
  Supervisor evidence is preserved at
  `orchestration/state/phase_0_attempt_2_20260815T201523Z.txt`.
- An independent read-only validator reproduced the Phase 0 split, scaler fit
  scope, all predictions and metrics, loaded the saved model/scaler, inspected
  the plot and artifacts, reran the full suite (124 passing), and returned exact
  PASS. The validation report is preserved at
  `orchestration/state/phase_0_attempt_2_validator_pass_20260815.txt`.
- Independent Phase 1 validation returned exact PASS. The strict schema-version-2
  checkpoint is preserved at `orchestration/state/checkpoints/phase_1_passed.json`.
- Phase 2 Feature Agent attempt 1 implemented 40 deterministic past-only candidate
  features, including shifted lag/rolling/EMA/change features, known-date calendar,
  cyclical, Fourier, Ontario holiday features, and exact-calendar previous-week/year
  values. Feature selection was explicitly deferred to Phase 3 and no validation or
  test partition was accessed. Focused tests pass (3), full discovery passes (129),
  and evidence is preserved under `artifacts/phase2/phase2_attempt_1_20260815T204500Z/`
  and `orchestration/state/phase_2_attempt_1_20260815T203808Z.txt`.
- Independent Phase 2 validation returned exact PASS. The strict schema-version-2
  checkpoint is preserved at `orchestration/state/checkpoints/phase_2_passed.json`.
- Phase 3 Feature Agent attempt 1 implemented deterministic training-only feature
  selection over the first 70% (2,660 rows through 2017-07-20). Leakage screening,
  training-only correlation analysis, random-forest model importance, permutation
  importance on later in-training folds, stability across three expanding-window
  splits, and domain sanity checks selected 20 of 40 candidates. No validation or
  locked-test rows were loaded. Focused tests pass (4), full discovery passes (133),
  and final evidence is preserved under
  `artifacts/phase3/phase3_attempt_1_final_20260815T210000Z/` and
  `orchestration/state/phase_3_attempt_1_20260815T210000Z.txt`.
- Independent Phase 3 validation returned exact PASS. The strict schema-version-2
  checkpoint is preserved at `orchestration/state/checkpoints/phase_3_passed.json`.
- Phase 4 ML Agent attempt 1 implemented the ten approved traditional model
  families with deterministic seeds and per-model sklearn pipelines. Median
  imputation and RobustScaler fitting occur only on the earlier development-fit
  portion of the already approved 70% training prefix. A chronological later
  portion of that same prefix provides smoke comparison only; all reserved rows
  remain unloaded, and formal evaluation remains deferred to Phases 5 and 6.
  Naive lag-1 and linear-regression baselines are reported without a model-selection
  or improvement claim. Focused tests pass (4), full discovery passes (137), and
  evidence is preserved under
  `artifacts/phase4/phase4_attempt_1_final_20260815T213000Z/` and
  `orchestration/state/phase_4_attempt_1_20260815T213000Z.txt`.
- Independent Phase 4 validation returned exact PASS and its local checkpoint
  commit is `ac75111`. Phases 0 through 4 now form a clean validated commit chain.
- Production-safe `--auto` orchestration is implemented without starting Phase 5.
  It validates the consecutive checkpoint prefix, resumes at the first incomplete
  phase, retains the existing exact pytest/validator/repair gates, reports live
  progress, stops on permissions and unsafe Git changes, creates local PASS-only
  phase commits, and performs final Phase 13 verification and JSON reporting.
- The first independent auto-mode review returned FAIL. Its exact corrections
  were applied: validator/test repair reports and remaining retry budgets now
  survive restart, interrupted dirty state is phase-scope checked, Git paths use
  narrow phase-specific scopes, and all candidate content is stream-scanned for
  secrets including large/binary files. A second review identified missing common
  credential formats and checkpoint scanning; both were added with regression
  coverage. Independent read-only revalidation returned exact PASS with 175 tests
  passing and `git diff --check` clean.
- Phase 5 independently passed on attempt 2, but its local Git checkpoint was
  blocked because nested attempt 1 artifacts were misclassified as unexpected.
  Both failed/repaired attempt artifact sets remain preserved.
- Phase ownership now accepts nested outputs from every attempt of only the
  current phase, and restart recovery detects a structurally valid latest PASS
  checkpoint missing its Git commit, safety-reviews and commits that phase delta,
  then continues without rerunning the validated phase.
- Phase 6 ML Agent attempt 1 implemented five-fold expanding-window cross-validation
  with `TimeSeriesSplit`. The naive lag-1 baseline and all ten approved traditional
  models are evaluated independently in every fold; preprocessing is fitted only on
  each fold's training prefix. Fold metrics/predictions/models, aggregate mean and
  sample-standard-deviation metrics, split evidence, diagnostics, configuration,
  hashes, seeds, and logs are preserved under
  `artifacts/phase6/phase6_attempt_1_20260815T233000Z/`. Focused tests pass (3),
  full discovery passes (184), and no tuning or improvement claim was made.
- Independent Phase 6 validation returned PASS. Phase 7 attempt 1 then completed
  validation-only Optuna selection and one frozen locked-test evaluation. Repair
  attempt 2 added the evaluation artifacts required by `EXPERIMENT_PLAN.md` from
  the already-preserved predictions only: actual-vs-predicted scatter,
  residual-vs-predicted, residual distribution, error-over-time, highest-error
  dates, and row-level diagnostics. The prediction and model hashes are unchanged.
  Focused tests pass (3), full discovery passes (187), and repair evidence is
  preserved in the Phase 7 artifact directory.
- Independent Phase 7 validation returned exact PASS and its validated local
  checkpoint commit is `1fbfa90`.
- Phase 8 ML Agent attempt 1 implemented validation-only Optuna optimization over
  five expanding `TimeSeriesSplit` folds. Ridge, random forest, and gradient
  boosting trials minimize mean fold MAE; every trial and final comparison fits
  preprocessing independently on the corresponding training prefix. Naive lag-1
  and linear-regression baselines are preserved alongside fold metrics,
  predictions, fitted fold models, diagnostics, configuration, hashes, seeds, and
  logs under `artifacts/phase8/phase8_attempt_1_20260815T230000Z/`. The naive
  baseline outperformed all tuned candidates, so no improvement claim was made.
  Focused tests pass (3), full discovery passes (190), and implementation evidence
  is preserved at
  `orchestration/state/phase_8_attempt_1_ml_agent_evidence_20260815.txt`.
- Independent validation rejected Phase 8 attempt 1 because its fixed Phase 3
  feature set had seen observations after folds 1--4 training boundaries. Those
  attempt-1 metrics are preserved for audit history but are invalid and must not
  be used as Phase 8 results.
- Phase 8 ML Agent repair attempt 2 initially refit feature selection per outer
  fold, but independent review rejected its globally selected hyperparameters.
  The interim `phase8_attempt_2_20260815T235500Z` outputs remain preserved but
  invalid. The final repair uses fully nested chronology: each outer fold has
  five inner folds, selection is fitted on every inner training prefix, and each
  model family's parameters and winning candidate are selected without outer
  validation access. The matching outer-training-only selection and parameters
  drive each final candidate evaluation. The strengthened two-trial regression
  test proves arbitrary post-boundary changes cannot alter fold-1 selected
  features, complete trial records/objectives, selected parameters/trial,
  preprocessing state, fitted winner, or fixed-input predictions. All 300 trials
  and every metric, prediction, model, report, plot, hash, and log were regenerated
  under `artifacts/phase8/phase8_attempt_2_final_20260816T003000Z/`. Full discovery
  passes (190 tests); 40 artifact hashes and saved prediction metrics were
  independently recomputed. Evidence is preserved at
  `orchestration/state/phase_8_attempt_2_ml_agent_evidence_20260815.txt`.
- Independent final revalidation returned exact PASS. The Validator confirmed
  nested chronology, training-only selection/tuning, fold-local parameters and
  winners, 40/40 hashes, independently recomputed saved-model metrics, consistent
  artifact counts, 190 passing tests, and a clean `git diff --check`. The report
  is preserved at
  `orchestration/state/phase_8_attempt_2_validator_pass_20260816.txt`.
- Phases 9 and 10 received independent PASS checkpoints and validated local
  commits `cb3c70c` and `fd1f1bf`, respectively.
- Phase 11 repair attempt 2 added a true 1--365 day future-horizon control with
  causal last-observation and recursive seven-day seasonal-naive forecasts.
  Historical evaluation filtering, metrics, plots, and downloads remain clearly
  separate. Explicit-origin tests prove post-origin target mutations cannot alter
  forecasts; future exports have no actual or residual fields. Focused tests pass
  (8), full discovery passes (204), syntax compilation and `git diff --check`
  pass. Evidence is preserved under
  `artifacts/phase11/phase11_attempt_2_20260816T000800Z/`.

## Current
Phases 0 through 10 are approved after exact independent validator PASS. Phase
11 repair attempt 2 is implemented and tested, awaiting independent validation.
Invalid/interim Phase 8 and failed Phase 9 attempts remain preserved but are not
exposed by the dashboard.

## Next
Independent Validator must review Phase 11 attempt 2 and return exact PASS before
Phase 12 may begin.
