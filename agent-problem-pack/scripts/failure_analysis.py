"""Extract human-readable failure summaries from pytest verification output."""

from __future__ import annotations

import json
import re


PLACEHOLDER_ANSWER = "write the final answer for this run here"


def _passed(verification_text: str, explicit_passed: bool | None = None) -> bool:
    if explicit_passed is True:
        return True
    if explicit_passed is False:
        return False

    text = verification_text or ""
    match = re.search(r"exit_code=(\d+)", text)
    if match:
        return match.group(1) == "0"

    if re.search(r"^FAILED\s", text, re.MULTILINE):
        return False

    failed_match = re.search(r"(\d+) failed\b", text)
    if failed_match:
        return int(failed_match.group(1)) == 0

    if re.search(r"(\d+) passed\b", text):
        return True

    return False


def _clean_message(message: str) -> str:
    text = message.strip()
    if text.startswith("AssertionError: "):
        text = text[len("AssertionError: ") :]
    if len(text) > 180:
        text = text[:177] + "..."
    return text


def _parse_failures(verification_text: str) -> list[dict[str, str | None]]:
    failures: list[dict[str, str | None]] = []
    seen: set[str] = set()

    for line in (verification_text or "").splitlines():
        stripped = line.strip()
        summary = re.match(r"FAILED\s+\S+::(\w+)\s+-\s+(.+)$", stripped)
        if summary:
            message = _clean_message(summary.group(2))
            if message not in seen:
                failures.append({"test": summary.group(1), "message": message})
                seen.add(message)
            continue

        assertion = re.match(r"E\s+AssertionError:\s+(.+)$", stripped)
        if assertion:
            message = _clean_message(assertion.group(1))
            if message not in seen:
                failures.append({"test": None, "message": message})
                seen.add(message)

    return failures


def _collect_hints(*, answer_text: str = "", diff_text: str = "", verification_text: str = "") -> list[str]:
    hints: list[str] = []
    answer = (answer_text or "").strip().lower()
    if answer and PLACEHOLDER_ANSWER in answer:
        hints.append("The agent left the placeholder AGENT_FINAL_ANSWER.md unchanged.")

    if diff_text is not None and diff_text.strip() == "":
        hints.append("No workspace changes were captured in the git diff.")

    body = verification_text or ""
    if "ModuleNotFoundError" in body:
        hints.append("An import or module path error blocked the test suite from running.")
    if "SyntaxError" in body:
        hints.append("A syntax error prevented tests from executing.")
    if "collecting ... ERROR" in body or "errors during collection" in body.lower():
        hints.append("Pytest failed while collecting tests.")

    return hints


def _build_headline(*, passed: bool, failures: list[dict], hints: list[str]) -> str | None:
    if passed:
        return None

    placeholder_hint = any("placeholder" in hint.lower() for hint in hints)
    if placeholder_hint and failures:
        return "Final answer was never written; comprehension checks failed."

    if failures:
        first = failures[0]["message"]
        if len(failures) == 1:
            return first
        return f"{first} (+{len(failures) - 1} more failing check{'s' if len(failures) > 2 else ''})"

    if hints:
        return hints[0]

    return "Verification failed; inspect pytest output for details."


def summarize_verification(
    verification_text: str,
    *,
    answer_text: str = "",
    diff_text: str = "",
    passed: bool | None = None,
) -> dict:
    ok = _passed(verification_text, passed)
    failures = [] if ok else _parse_failures(verification_text)
    hints = [] if ok else _collect_hints(
        answer_text=answer_text,
        diff_text=diff_text,
        verification_text=verification_text,
    )
    headline = _build_headline(passed=ok, failures=failures, hints=hints)

    return {
        "passed": ok,
        "failure_count": len(failures),
        "headline": headline,
        "hints": hints,
        "failures": failures[:10],
    }


def summarize_verification_json(
    verification_text: str,
    *,
    answer_text: str = "",
    diff_text: str = "",
) -> str:
    return json.dumps(
        summarize_verification(
            verification_text,
            answer_text=answer_text,
            diff_text=diff_text,
        ),
        indent=2,
    ) + "\n"
