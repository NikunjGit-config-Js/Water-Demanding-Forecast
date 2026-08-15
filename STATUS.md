# Project Status

Current Phase: Phase 1 implementation attempt 1 awaiting independent validation

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

## Current
Phase 0 attempt 2 is approved after exact independent validator PASS. Phase 1
Data Agent attempt 1 implemented non-mutating validation and EDA for the supplied
London dataset. Focused tests and full discovery pass (126 tests), and the
authoritative attempt artifacts are preserved under
`artifacts/phase1/phase1_attempt_1_final_20260815T203000Z/`. Two source-data
calendar gaps totaling 282 missing dates are reported; source order and bytes
remain unchanged. Independent Phase 1 validation has not been invoked.

## Next
The supervisor may invoke independent read-only Phase 1 validation. Do not
advance to Phase 2 unless the validator returns exact PASS. Preserve the locked
Phase 0 evidence, all Phase 1 outputs, and the source dataset.
