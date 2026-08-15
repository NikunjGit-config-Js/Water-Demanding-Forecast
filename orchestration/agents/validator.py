from __future__ import annotations

from dataclasses import dataclass

from orchestration.tools.codex_tool import run_codex


@dataclass
class ValidationResult:
    verdict: str
    report: str

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


class ValidatorError(RuntimeError):
    pass


def validate_phase(
    phase_name: str,
    implementation_summary: str = "",
) -> ValidationResult:
    """
    Independently validate a completed project phase.

    The validator:
    - cannot modify files
    - reads project methodology/rules
    - inspects repository state
    - may run read-only tests/checks
    - returns PASS or FAIL
    """

    prompt = f"""
You are the INDEPENDENT VALIDATOR for this repository.

PHASE TO VALIDATE:
{phase_name}

IMPLEMENTATION SUMMARY:
{implementation_summary or "No implementation summary supplied."}

MANDATORY INSTRUCTIONS:

1. Read:
   - AGENTS.md
   - PROJECT_SPEC.md
   - EXPERIMENT_PLAN.md
   - VALIDATION_RULES.md
   - PERMISSIONS.md
   - SUCCESS_CRITERIA.md
   - STATUS.md
   - DECISIONS.md

2. Independently inspect the repository.

3. Inspect:
   - git status
   - git diff
   - relevant source files
   - relevant tests
   - generated artifacts where applicable

4. Run safe READ-ONLY validation commands when useful.

5. For ML phases specifically check:
   - chronological splitting
   - data leakage
   - preprocessing placement
   - scaler fitting
   - feature generation
   - feature selection
   - test isolation
   - CV methodology
   - metric correctness
   - artifact existence
   - reproducibility

6. DO NOT modify any project file.

7. Do not trust the implementation summary without evidence.

8. Do not return PASS merely because scripts execute.

9. If there is insufficient evidence, return FAIL.

OUTPUT FORMAT IS MANDATORY.

The FIRST line must contain exactly one of:

PASS

or

FAIL

Then provide a concise evidence-based validation report.

If FAIL, include a section:

REQUIRED_CORRECTIONS:

with exact corrections needed before revalidation.
""".strip()

    report = run_codex(
        prompt,
        sandbox="read-only",
        timeout=3600,
    ).strip()

    if not report:
        raise ValidatorError("Validator produced an empty report.")

    first_line = report.splitlines()[0].strip().upper()

    if first_line not in {"PASS", "FAIL"}:
        raise ValidatorError(
            "Validator response did not begin with PASS or FAIL.\n\n"
            f"Response:\n{report}"
        )

    return ValidationResult(
        verdict=first_line,
        report=report,
    )


if __name__ == "__main__":
    result = validate_phase(
        "Automation infrastructure smoke test",
        """
Environment setup is complete.
Codex CLI authentication works.
The Codex Python wrapper has been created.
The wrapper successfully completed a read-only smoke test.
No ML implementation is being validated yet.
""".strip(),
    )

    print(result.report)
