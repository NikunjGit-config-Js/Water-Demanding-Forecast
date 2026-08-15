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
