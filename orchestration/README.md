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
