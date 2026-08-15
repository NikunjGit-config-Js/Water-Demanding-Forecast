# Decision Log

## D001
Use WSL for modern TensorFlow GPU support.

## D002
Use Python 3.12 for compatibility.

## D003
Use TensorFlow 2.20 because GPU detection is verified in the current environment.

## D004
Do not install the repository's historical requirements.txt unchanged.

## D005
Use chronological/time-aware evaluation rather than random splitting.

## D006
Use CrewAI Flow as orchestration layer.

## D007
Use Codex/coding-agent execution behind supervisor and validator gates.

## D008
Start with the supplied London example dataset.

## D009
Indian-city data collection is optional and cannot block the core London project.

## D010
Transformer forecasting is a separate later experiment rather than replacing traditional baselines.

## D011
Use a typed, deterministic CrewAI Flow as the control layer while keeping Codex
CLI as the only writable implementation executor and the existing read-only
Codex validator as the independent gate. Do not instantiate CrewAI LLM agents.

## D012
Disable CrewAI tracing, telemetry, automatic memory, and flow-event emission for
orchestration runs so the controller neither adds an API dependency nor writes
runtime memory outside the repository.

## D013
A phase checkpoint is a local JSON evidence record written only after validator
PASS. The orchestration layer does not commit or push automatically.

## D014
Require exact uppercase PASS or FAIL validator verdicts. FAIL is valid only with
a non-empty REQUIRED_CORRECTIONS section; all malformed output fails closed.

## D015
Use a separate `orchestration/requirements.txt` with exact direct dependency
pins and isolated `uv` installation. Preserve the historical root requirements.

## D016
Accept only unique, strictly increasing, consecutive Phase 0 through Phase 13
requests. Before Phase N, require valid PASS checkpoints for every Phase 0
through N-1, either produced earlier in the same Flow execution or loaded from
strict persisted checkpoint content. Filenames are locators, not proof.

## D017
Resolve evidence against the selected approved root and reject any symlink whose
resolved target escapes that root, even when the target remains in the project.

## D018
Limit the Flow test gate to the exact non-shell command `python -m pytest -q`.
Do not accept caller-defined Python, pytest, shell, or other command variants.

## D019
Persist prerequisite checkpoints with strict schema version 2. Replace free-form
test proof with typed evidence containing exactly the command token list
`['python', '-m', 'pytest', '-q']` and integer return code `0`. Reject legacy
checkpoint formats and any missing, malformed, contradictory, or extra evidence.

## D020
Use one case-sensitive validation-report rule for live and persisted results:
the report's first line must be exactly the supplied `PASS` or `FAIL` verdict.
Persisted prerequisites additionally require exact `PASS` for both fields.

## D021
Preserve the raw Codex response for live independent validation and apply the
shared strict report parser before any stripping or normalization. Keep stripped
Codex responses as the wrapper default for backward-compatible implementation
execution.

## D022
Add `--auto` as a deterministic supervisor over the existing single/multi-phase
Flow. It discovers the consecutive schema-v2 PASS checkpoint prefix, starts at
the first incomplete phase, and invokes one phase at a time so no later phase can
start before the preceding independent PASS and local checkpoint commit.

## D023
Keep auto runtime/failure state in ignored `orchestration/state/auto_run.json`.
Record the active phase before writable execution so an interruption can resume
that phase, and so a PASS checkpoint created just before interruption can be
committed safely without rerunning the validated implementation. Persist the
exact validator/test correction report and consumed attempt count; restart enters
the repair path with only the original remaining bounded attempts.

## D024
Automatic Git checkpoints require a clean starting worktree, reject deletions,
symlinks, protected methodology/data/repository-control paths, and paths outside
the current phase's narrow filename/directory scope. Stream-scan every candidate
and the force-added PASS checkpoint, including large and binary files, for private
keys, common provider tokens, JWT/bearer credentials, and credential assignments;
stage only the reviewed phase delta, run Git whitespace checks, and create a
normal local commit. Auto mode never pushes, force-pushes, deploys, or rewrites
history.

## D025
Permission-required specialist output begins with exact `PERMISSION_REQUIRED`.
Auto mode latches this state, preserves evidence, and stops until the user takes
an explicit approved recovery action. Infrastructure errors and malformed
validator output remain failures and can never create PASS checkpoints.

## D026
Treat all explicitly phase-owned outputs under a phase directory as belonging to
that phase across failed, repaired, and passing attempts. Use boundary-safe
prefix and filename checks instead of recursive `PurePosixPath.match` globs;
retain protected-path, dataset, deletion, symlink, repository-control, and
streamed secret checks, and never delete failed-attempt evidence.

## D027
Derive a missing local phase commit from the strict consecutive PASS checkpoint
chain and Git history, not only ignored runtime state. Recover only one latest
validated phase after its complete dirty delta passes the same narrow ownership
and Git safety checks, then continue at the next phase without rerunning it.
