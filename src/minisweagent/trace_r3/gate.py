"""Execution-evidence gate used before activating recovery."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from minisweagent import Environment
from minisweagent.trace_r3.types import GateCheck, GateReport, GateState, TraceR3Config

_COMMAND_BOUNDARY = r"(?:^|(?:&&|\|\||;|\||\r?\n)\s*)"
_TEST_COMMAND = re.compile(
    _COMMAND_BOUNDARY
    + r"(?:timeout\s+\S+\s+)?(?:"
    r"pytest|py\.test|tox|nox|"
    r"python(?:\d+(?:\.\d+)?)?\s+-m\s+(?:pytest|unittest|sympy\.testing\.runtests)|"
    r"python(?:\d+(?:\.\d+)?)?\s+(?:\./)?bin/test|(?:\./)?bin/test|"
    r"\./gradlew\s+test|mvn\s+(?:test|verify)|npm\s+(?:run\s+)?test|cargo\s+test|go\s+test"
    r")(?:[\s;&]|$)",
    re.IGNORECASE,
)
_INLINE_TEST_COMMAND = re.compile(
    r"\bpython(?:\d+(?:\.\d+)?)?\s+-c\b.*"
    r"\b(?:sympy\.test|pytest\.main|unittest\.main)\s*\(",
    re.IGNORECASE | re.DOTALL,
)
_REPRO_COMMAND = re.compile(
    r"(?:python(?:\d+(?:\.\d+)?)?\s+(?:-c|-|[^;&\s]+\.py)\b.*\bassert\b|"
    r"\b(?:repro|reproduce|regression)[_\-.a-zA-Z0-9]*\.(?:py|sh)\b)",
    re.IGNORECASE | re.DOTALL,
)
_MASKED_COMMAND = re.compile(r"\|\||;\s*(?:true|:)\s*$", re.IGNORECASE)
_SKIPPED_ONLY = re.compile(
    r"(?:no tests ran|collected 0 items|\b0 passed\b|all tests (?:were )?skipped|nothing to test)",
    re.IGNORECASE,
)
_FAILED_TEST_EVIDENCE = re.compile(
    r"(?:\b[1-9]\d*\s+failed\b|\bFAILURES?\b|\bFAILED(?:\s*\(|\b)|"
    r"\bBUILD FAILURE\b|\btest result:\s*FAILED\b|\berrors?=[1-9]\d*\b)",
    re.IGNORECASE,
)
_POSITIVE_TEST_EVIDENCE = (
    re.compile(r"\b[1-9]\d*\s+passed\b", re.IGNORECASE),
    re.compile(r"\bRan\s+[1-9]\d*\s+tests?\b", re.IGNORECASE),
    re.compile(r"\btest result:\s*ok\b.*\b[1-9]\d*\s+passed\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"^\s*ok\s+\S+(?:\s+\d+(?:\.\d+)?s)?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(
        r"\bTests run:\s*[1-9]\d*\b.*\bFailures:\s*0\b.*\bErrors:\s*0\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\b[1-9]\d*\s+tests?\s+(?:completed|run)\b.*\b0\s+failed\b", re.IGNORECASE | re.DOTALL),
)
_TEST_FILE = re.compile(r"(^|/)tests?(?:/|$)|(?:^|/)test_[^/]+\.py$|_test\.py$")


@dataclass(frozen=True)
class CommandCandidate:
    command: str
    kind: str


class GateEvaluator:
    """Build a GateReport from the current workspace and an agent trajectory."""

    def __init__(self, config: TraceR3Config):
        self.config = config

    def evaluate(self, env: Environment, messages: list[dict[str, Any]]) -> GateReport:
        patch_result = self._execute(env, "git diff --binary -- . ':(exclude)patch.txt'")
        patch = patch_result["output"] if patch_result["returncode"] == 0 else ""
        changed_files = self._changed_files(env)
        added_lines, deleted_lines = _patch_size(patch)

        checks = [
            GateCheck(
                "patch_nonempty",
                bool(patch.strip()),
                "A source patch exists" if patch.strip() else "No source patch was produced",
            ),
            self._command_check(env, "diff_check", "git diff --check", "Patch has no whitespace/apply errors"),
            self._scope_check(changed_files, added_lines + deleted_lines),
            self._test_pollution_check(changed_files),
            self._syntax_check(env, changed_files),
        ]

        candidates = extract_validation_commands(messages, limit=self.config.max_replayed_commands)
        replay_checks = [self._replay(env, candidate) for candidate in candidates]
        checks.extend(replay_checks)

        has_test = any(candidate.kind == "test" for candidate in candidates)
        has_repro = any(candidate.kind == "reproduction" for candidate in candidates)
        behavioral_passed = True if has_test else None
        behavioral_detail = (
            "Replayed a real test command"
            if has_test
            else "Only reproduction evidence was replayed; a real test command is required"
            if has_repro
            else "No safe test or reproduction command found"
        )
        checks.append(GateCheck("behavioral_evidence", behavioral_passed, behavioral_detail, critical=False))

        state, reason = _classify(checks)
        return GateReport(
            state=state,
            checks=tuple(checks),
            patch=patch,
            changed_files=tuple(changed_files),
            added_lines=added_lines,
            deleted_lines=deleted_lines,
            tested_commands=tuple(candidate.command for candidate in candidates),
            reason=reason,
        )

    def _execute(self, env: Environment, command: str) -> dict[str, Any]:
        return env.execute({"command": command}, timeout=self.config.validation_timeout_seconds)

    def _changed_files(self, env: Environment) -> list[str]:
        result = self._execute(env, "git diff --name-only -- . ':(exclude)patch.txt'")
        if result["returncode"] != 0:
            return []
        return [line for line in result["output"].splitlines() if line.strip()]

    def _command_check(self, env: Environment, name: str, command: str, success: str) -> GateCheck:
        result = self._execute(env, command)
        return GateCheck(
            name,
            result["returncode"] == 0,
            success if result["returncode"] == 0 else f"Command failed with return code {result['returncode']}",
            command=command,
            returncode=result["returncode"],
            output=_bounded_output(result),
        )

    def _scope_check(self, changed_files: list[str], changed_lines: int) -> GateCheck:
        passed = len(changed_files) <= self.config.max_patch_files and changed_lines <= self.config.max_patch_lines
        detail = (
            f"{len(changed_files)} files and {changed_lines} changed lines "
            f"(limits: {self.config.max_patch_files}, {self.config.max_patch_lines})"
        )
        return GateCheck("patch_scope", passed, detail)

    def _test_pollution_check(self, changed_files: list[str]) -> GateCheck:
        polluted = [path for path in changed_files if _TEST_FILE.search(path)]
        return GateCheck(
            "no_test_modifications",
            not polluted,
            "No tests modified" if not polluted else f"Test files modified: {', '.join(polluted)}",
        )

    def _syntax_check(self, env: Environment, changed_files: list[str]) -> GateCheck:
        python_files = [path for path in changed_files if path.endswith(".py")]
        if not python_files:
            return GateCheck("python_syntax", True, "No changed Python files")
        quoted = " ".join(shlex.quote(path) for path in python_files)
        command = f"python -m py_compile {quoted}"
        return self._command_check(env, "python_syntax", command, "Changed Python files compile")

    def _replay(self, env: Environment, candidate: CommandCandidate) -> GateCheck:
        command = f"bash -o pipefail -c {shlex.quote(candidate.command)}"
        result = self._execute(env, command)
        output = str(result.get("output", ""))
        skipped_only = result["returncode"] == 0 and bool(_SKIPPED_ONLY.search(output))
        failed_evidence = candidate.kind == "test" and bool(_FAILED_TEST_EVIDENCE.search(output))
        positive_evidence = candidate.kind != "test" or any(
            pattern.search(output) for pattern in _POSITIVE_TEST_EVIDENCE
        )
        passed = (
            result["returncode"] == 0
            and not skipped_only
            and not failed_evidence
            and positive_evidence
        )
        if passed:
            detail = f"Replayed {candidate.kind} command successfully"
        elif skipped_only:
            detail = "Command produced only skipped/empty tests"
        elif failed_evidence:
            detail = "Test output reports failures despite the command return code"
        elif result["returncode"] == 0 and candidate.kind == "test":
            detail = "Runner exited successfully but gave no positive evidence that any test executed"
        else:
            detail = f"Replayed {candidate.kind} command failed with return code {result['returncode']}"
        return GateCheck(
            f"replay_{candidate.kind}",
            passed,
            detail,
            command=candidate.command,
            returncode=result["returncode"],
            output=_bounded_output(result),
        )


def extract_validation_commands(messages: Iterable[dict[str, Any]], *, limit: int) -> list[CommandCandidate]:
    """Return the last unique, safe reproduction/test commands from a trajectory."""

    found: list[CommandCandidate] = []
    seen: set[str] = set()
    for message in reversed(list(messages)):
        if message.get("role") != "assistant":
            continue
        for action in reversed(message.get("extra", {}).get("actions", [])):
            command = action.get("command", "").strip()
            if not command or command in seen or "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
                continue
            kind = (
                "test"
                if _TEST_COMMAND.search(command) or _INLINE_TEST_COMMAND.search(command)
                else "reproduction"
                if _REPRO_COMMAND.search(command)
                else ""
            )
            if not kind or _MASKED_COMMAND.search(command):
                continue
            seen.add(command)
            found.append(CommandCandidate(command=command, kind=kind))
            if len(found) >= limit:
                return list(reversed(found))
    return list(reversed(found))


def _classify(checks: list[GateCheck]) -> tuple[GateState, str]:
    critical_failures = [check.name for check in checks if check.critical and check.passed is False]
    replay_failures = [check.name for check in checks if check.name.startswith("replay_") and check.passed is False]
    if critical_failures or replay_failures:
        failed = critical_failures + replay_failures
        return GateState.RED, f"Hard validation failed: {', '.join(dict.fromkeys(failed))}"
    behavioral = next(check for check in checks if check.name == "behavioral_evidence")
    if behavioral.passed is None:
        return GateState.AMBER, "Patch is structurally valid but lacks replayable behavioral evidence"
    return GateState.GREEN, "Structural and replayed behavioral checks passed"


def _patch_size(patch: str) -> tuple[int, int]:
    added = 0
    deleted = 0
    for line in patch.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        added += line.startswith("+")
        deleted += line.startswith("-")
    return added, deleted


def _bounded_output(result: dict[str, Any], limit: int = 4000) -> str:
    output = str(result.get("output", ""))
    exception = str(result.get("exception_info", ""))
    combined = f"{exception}\n{output}".strip()
    if len(combined) <= limit:
        return combined
    half = limit // 2
    return f"{combined[:half]}\n... <truncated> ...\n{combined[-half:]}"
