# Orchestration environment

Install the isolated CrewAI control-layer dependencies with `uv` from the
repository root:

```bash
uv venv --python 3.12 .venv-orchestration
uv pip sync --python .venv-orchestration/bin/python orchestration/requirements.txt
```

Run the orchestration tests with the same interpreter:

```bash
.venv-orchestration/bin/python -m pytest -q
```

The root `requirements.txt` is historical and is intentionally unchanged.

Run one or more explicit phases as before:

```bash
python -m orchestration.main --phase "Phase 5" --max-attempts 3
```

Or resume automatically from the first structurally valid incomplete checkpoint:

```bash
python -m orchestration.main --auto --max-attempts 3
```

Auto mode requires a clean worktree before a new phase, never reruns a valid PASS
checkpoint, stops on permission or Git-safety concerns, creates local phase
commits only after independent PASS, and never pushes or deploys.
