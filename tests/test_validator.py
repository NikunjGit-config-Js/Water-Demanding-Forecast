from __future__ import annotations

import pytest

from orchestration.agents import validator


def test_validator_always_uses_read_only_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_codex(prompt: str, **kwargs: object) -> str:
        captured.update(kwargs)
        return "PASS\nEvidence"

    monkeypatch.setattr(validator, "run_codex", fake_codex)
    result = validator.validate_phase("Phase 1", "summary")
    assert result.passed
    assert captured["sandbox"] == "read-only"
    assert captured["preserve_output"] is True


@pytest.mark.parametrize(
    "response",
    [
        "\nPASS\nEvidence",
        " PASS\nEvidence",
        "PASS \nEvidence",
        "\tPASS\nEvidence",
        "\nFAIL\nREQUIRED_CORRECTIONS:\nFix",
        " FAIL\nREQUIRED_CORRECTIONS:\nFix",
        "FAIL \nREQUIRED_CORRECTIONS:\nFix",
    ],
)
def test_live_validator_rejects_whitespace_normalized_reports(
    monkeypatch: pytest.MonkeyPatch, response: str
) -> None:
    monkeypatch.setattr(validator, "run_codex", lambda *args, **kwargs: response)
    with pytest.raises(validator.ValidatorError):
        validator.validate_phase("Phase 1")


@pytest.mark.parametrize(
    ("response", "verdict", "passed"),
    [
        ("PASS\nEvidence", "PASS", True),
        ("FAIL\nREQUIRED_CORRECTIONS:\nFix", "FAIL", False),
    ],
)
def test_live_validator_accepts_exact_well_formed_reports(
    monkeypatch: pytest.MonkeyPatch,
    response: str,
    verdict: str,
    passed: bool,
) -> None:
    monkeypatch.setattr(validator, "run_codex", lambda *args, **kwargs: response)
    result = validator.validate_phase("Phase 1")
    assert result.verdict == verdict
    assert result.passed is passed
    assert result.report == response


@pytest.mark.parametrize(
    "response",
    ["", "pass\nEvidence", "MAYBE\nEvidence", "FAIL\nNo correction section", "FAIL\nREQUIRED_CORRECTIONS:\n"],
)
def test_validator_rejects_malformed_reports(
    monkeypatch: pytest.MonkeyPatch, response: str
) -> None:
    monkeypatch.setattr(validator, "run_codex", lambda *args, **kwargs: response)
    with pytest.raises(validator.ValidatorError):
        validator.validate_phase("Phase 1")


@pytest.mark.parametrize(
    ("verdict", "report"),
    [
        ("PASS", "FAIL\nREQUIRED_CORRECTIONS:\nBroken"),
        ("PASS", "pass\nEvidence"),
        ("PASS", "Pass\nEvidence"),
        ("PASS", "malformed text"),
        ("PASS", ""),
        ("FAIL", "PASS\nEvidence"),
    ],
)
def test_shared_validation_report_check_rejects_contradictions_and_malformed_text(
    verdict: str, report: str
) -> None:
    with pytest.raises(validator.ValidatorError):
        validator.validate_report(verdict, report)


@pytest.mark.parametrize(
    "report",
    [
        "\nPASS\nEvidence",
        " PASS\nEvidence",
        "PASS \nEvidence",
        "\tPASS\nEvidence",
        "pass\nEvidence",
        "Pass\nEvidence",
    ],
)
def test_shared_validation_report_check_rejects_non_exact_pass(report: str) -> None:
    with pytest.raises(validator.ValidatorError):
        validator.validate_report("PASS", report)


@pytest.mark.parametrize(
    "report",
    [
        "\nFAIL\nREQUIRED_CORRECTIONS:\nFix",
        " FAIL\nREQUIRED_CORRECTIONS:\nFix",
        "FAIL \nREQUIRED_CORRECTIONS:\nFix",
        "FAIL\nREQUIRED_CORRECTIONS:\n",
        "FAIL\n REQUIRED_CORRECTIONS:\nFix",
        "FAIL\nDetails REQUIRED_CORRECTIONS:\nFix",
        "FAIL\nREQUIRED_CORRECTIONS: Fix",
    ],
)
def test_shared_validation_report_check_rejects_malformed_fail(report: str) -> None:
    with pytest.raises(validator.ValidatorError):
        validator.validate_report("FAIL", report)


def test_shared_validation_report_check_accepts_well_formed_fail() -> None:
    report = "FAIL\nREQUIRED_CORRECTIONS:\nFix"
    result = validator.validate_report("FAIL", report)
    assert result.verdict == "FAIL"
    assert not result.passed
    assert result.report == report
