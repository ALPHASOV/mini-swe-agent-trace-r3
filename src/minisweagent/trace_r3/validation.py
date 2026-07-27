"""Patch-independent frozen validation planning and execution."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from minisweagent import Environment
from minisweagent.trace_r3.types import GateCheck, TraceR3Config

VALIDATION_CATEGORIES = (
    "direct_reproduction",
    "boundary",
    "metamorphic",
    "cross_consumer",
    "focused_regression",
    "broad_regression",
)
_CATEGORY_MINIMUMS = {
    "direct_reproduction": 2,
    "boundary": 2,
    "metamorphic": 2,
    "cross_consumer": 2,
    "focused_regression": 1,
    "broad_regression": 1,
}
_TEST_COMMAND = re.compile(
    r"(?:^|&&\s*)(?:timeout\s+\S+\s+)?(?:"
    r"pytest|py\.test|tox|nox|"
    r"python(?:\d+(?:\.\d+)?)?\s+-m\s+(?:pytest|unittest|sympy\.testing\.runtests)|"
    r"python(?:\d+(?:\.\d+)?)?\s+(?:\./)?(?:[^ ]*/)*(?:run)?tests?\.py|"
    r"(?:\./)?(?:[^ ]*/)*(?:run)?tests?\.py|"
    r"\./gradlew\s+test|mvn\s+(?:test|verify)|npm\s+(?:run\s+)?test|cargo\s+test|go\s+test"
    r")(?:[\s;&]|$)",
    re.IGNORECASE,
)
_REPRODUCTION_COMMAND = re.compile(
    r"(?:^|&&\s*)python(?:\d+(?:\.\d+)?)?\s+-c\s+.+\bassert\b",
    re.IGNORECASE | re.DOTALL,
)
_FORBIDDEN_COMMAND = re.compile(
    r"(?:^|[;&]\s*|\s)(?:rm|mv|cp|install|touch|mkdir|truncate|tee|"
    r"sed\s+-i|perl\s+-i|git|pip|conda|apt|curl|wget|ssh|scp)\b|"
    r"\b(?:write|writelines|write_text|write_bytes|unlink|rmtree|remove|"
    r"rename|replace|mkdir|rmdir|touch)\s*\(|"
    r"\b(?:os\.system|os\.popen|subprocess\.(?:run|call|Popen)|"
    r"socket\.|requests\.|httpx\.|urllib\.request)\b|"
    r"(?:\bopen|\.\s*open)\s*\([^,\n]+,\s*['\"][^'\"]*[wax+]",
    re.IGNORECASE,
)
_INFRA_FAILURE = re.compile(
    r"(?:command not found|no module named (?:pytest|unittest)|"
    r"\b(?:SyntaxError|IndentationError|PermissionError)\b|"
    r"no such file or directory|not found:\s*[^ \n]+|"
    r"error:\s*file or directory not found|collected 0 items|no tests ran|"
    r"usage error|unrecognized arguments?)",
    re.IGNORECASE,
)
_SKIPPED_ONLY = re.compile(
    r"(?:all tests (?:were )?skipped|nothing to test|\b0 passed\b)",
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
)


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    category: str
    kind: str
    title: str
    rationale: str
    command: str
    oracle: str
    expected_on_base: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationPlan:
    objective: str
    cases: tuple[ValidationCase, ...]
    schema_version: str = "trace-r3-frozen-validation/1"

    @property
    def identity(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "objective": self.objective,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class ValidationCaseResult:
    case_id: str
    category: str
    kind: str
    command: str
    passed: bool
    execution_valid: bool
    detail: str
    returncode: int
    output: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationRun:
    plan_identity: str
    phase: str
    results: tuple[ValidationCaseResult, ...]

    @property
    def all_passed(self) -> bool:
        return all(result.passed and result.execution_valid for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "trace-r3-validation-run/1",
            "plan_identity": self.plan_identity,
            "phase": self.phase,
            "all_passed": self.all_passed,
            "results": [result.to_dict() for result in self.results],
        }

    def gate_checks(self) -> tuple[GateCheck, ...]:
        checks = [
            GateCheck(
                "frozen_plan_identity",
                True,
                f"Executed frozen validation plan {self.plan_identity}",
            )
        ]
        checks.extend(
            GateCheck(
                f"frozen_validation.{result.case_id}",
                result.passed and result.execution_valid,
                result.detail,
                command=result.command,
                returncode=result.returncode,
                output=result.output,
            )
            for result in self.results
        )
        return tuple(checks)


def validation_plan_prompt(problem_statement: str, graph_artifact: dict[str, Any], config: TraceR3Config) -> str:
    category_rules = "\n".join(
        f"- {category}: at least {minimum}" for category, minimum in _CATEGORY_MINIMUMS.items()
    )
    return f"""<frozen_validation_planning>
You are an independent validation planner. No patch has been generated yet.
You must never propose, describe, or edit a source patch. Produce a complete
validation plan that will be frozen before a separate repair conversation
starts. The repair model will not see this plan. Only failed executed cases may
be shown to a later repair turn.

<issue>
{problem_statement}
</issue>

<coverage_contract>
Return {config.validation_plan_min_cases} to {config.validation_plan_max_cases}
unique cases. Every case must exit zero on a correct implementation.
Required category coverage:
{category_rules}

Include at least {config.validation_plan_min_test_commands} real native test
runner commands. Use focused repository tests where possible and one broader
but bounded regression command. Direct, boundary, metamorphic, and consumer
cases should use small `python -c` assertions when practical. Include at least
one direct-reproduction case expected to fail on clean B0 and at least two
cases expected to pass on B0. Both regression categories must use native test
runner commands.

Commands run from the repository root. They must be non-interactive, offline,
read-only, and must not create or edit files. Do not use pipes, output
redirection, shell fallbacks, truncation, package installation, network access,
or git commands. A test command must emit a nonzero test-count summary.
</coverage_contract>

<json_schema>
Return only one JSON object:
{{
  "schema_version": "trace-r3-frozen-validation/1",
  "objective": "short statement of the behavioral contract",
  "cases": [
    {{
      "case_id": "vp01",
      "category": "one of {", ".join(VALIDATION_CATEGORIES)}",
      "kind": "reproduction or test",
      "title": "short unique title",
      "rationale": "what distinct risk this covers",
      "command": "one safe shell command",
      "oracle": "what exit-zero proves",
      "expected_on_base": "fail or pass"
    }}
  ]
}}
</json_schema>

{graph_artifact.get("rag_context", "")}
</frozen_validation_planning>
"""


def parse_validation_plan(value: str, config: TraceR3Config) -> ValidationPlan:
    payload = _extract_json(value)
    if payload.get("schema_version") != "trace-r3-frozen-validation/1":
        raise ValueError("validation plan has the wrong schema_version")
    objective = str(payload.get("objective", "")).strip()
    if not objective:
        raise ValueError("validation plan objective is empty")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("validation plan cases must be a list")
    if not config.validation_plan_min_cases <= len(raw_cases) <= config.validation_plan_max_cases:
        raise ValueError(
            f"validation plan needs {config.validation_plan_min_cases}-"
            f"{config.validation_plan_max_cases} cases, found {len(raw_cases)}"
        )

    cases = tuple(_parse_case(item) for item in raw_cases)
    ids = [case.case_id for case in cases]
    commands = [case.command for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("validation plan case_id values must be unique")
    if len(commands) != len(set(commands)):
        raise ValueError("validation plan commands must be unique")
    counts = Counter(case.category for case in cases)
    missing = [
        f"{category}<{minimum}"
        for category, minimum in _CATEGORY_MINIMUMS.items()
        if counts[category] < minimum
    ]
    if missing:
        raise ValueError(f"validation plan lacks required category coverage: {', '.join(missing)}")
    if any(
        case.category in {"focused_regression", "broad_regression"} and case.kind != "test"
        for case in cases
    ):
        raise ValueError("focused and broad regression cases must use native test commands")
    if sum(case.kind == "test" for case in cases) < config.validation_plan_min_test_commands:
        raise ValueError(
            f"validation plan needs at least {config.validation_plan_min_test_commands} test commands"
        )
    if not any(
        case.category == "direct_reproduction" and case.expected_on_base == "fail"
        for case in cases
    ):
        raise ValueError("validation plan needs at least one B0-failing direct reproduction")
    if sum(case.expected_on_base == "pass" for case in cases) < 2:
        raise ValueError("validation plan needs at least two B0-passing cases")
    return ValidationPlan(objective=objective, cases=cases)


def execute_validation_plan(
    env: Environment,
    plan: ValidationPlan,
    *,
    phase: str,
    timeout: int,
) -> ValidationRun:
    results = []
    for case in plan.cases:
        result = env.execute(
            {"command": f"bash -o pipefail -c {shlex.quote(case.command)}"},
            timeout=timeout,
        )
        results.append(_case_result(case, result))
    return ValidationRun(plan_identity=plan.identity, phase=phase, results=tuple(results))


def calibration_errors(plan: ValidationPlan, run: ValidationRun) -> list[str]:
    errors = []
    by_id = {result.case_id: result for result in run.results}
    for case in plan.cases:
        result = by_id[case.case_id]
        if not result.execution_valid:
            errors.append(f"{case.case_id}: invalid execution: {result.detail}")
            continue
        expected_pass = case.expected_on_base == "pass"
        if result.passed != expected_pass:
            errors.append(
                f"{case.case_id}: expected B0 {case.expected_on_base}, "
                f"observed {'pass' if result.passed else 'fail'}"
            )
    return errors


def planner_revision_prompt(errors: list[str], run: ValidationRun) -> str:
    failed = [
        {
            "case_id": result.case_id,
            "returncode": result.returncode,
            "detail": result.detail,
            "output": result.output,
        }
        for result in run.results
        if not result.execution_valid
        or any(error.startswith(f"{result.case_id}:") for error in errors)
    ]
    return (
        "The pre-patch B0 calibration rejected this draft plan. No patch exists. "
        "Replace the entire plan with a corrected JSON object that still satisfies "
        "all coverage requirements.\n\n"
        f"Errors:\n- {chr(10).join(errors)}\n\n"
        f"Relevant execution evidence:\n{json.dumps(failed, indent=2, ensure_ascii=False)}"
    )


def render_validation_plan_markdown(plan: ValidationPlan) -> str:
    rows = ["| ID | Category | Kind | B0 | Title |", "|---|---|---|---|---|"]
    for case in plan.cases:
        title = case.title.replace("|", "\\|")
        rows.append(
            f"| `{case.case_id}` | {case.category} | {case.kind} | "
            f"{case.expected_on_base} | {title} |"
        )
    return (
        f"# Frozen validation plan\n\nIdentity: `{plan.identity}`\n\n"
        f"Objective: {plan.objective}\n\n{chr(10).join(rows)}\n"
    )


def failed_validation_feedback(report_checks: tuple[GateCheck, ...]) -> str:
    failed = [
        check
        for check in report_checks
        if check.name.startswith("frozen_validation.") and check.passed is False
    ]
    if not failed:
        return ""
    blocks = []
    for check in failed:
        blocks.append(
            f"<case id={json.dumps(check.name.removeprefix('frozen_validation.'))}>\n"
            f"Command: {check.command}\n"
            f"Result: {check.detail}\n"
            f"Output:\n{_bounded(check.output)}\n"
            "</case>"
        )
    return (
        "<failed_frozen_validation_cases>\n"
        "These cases come from the immutable plan created before any patch. "
        "Use only this failure evidence; do not edit, replace, or weaken the tests.\n"
        f"{chr(10).join(blocks)}\n"
        "</failed_frozen_validation_cases>"
    )


def _parse_case(value: Any) -> ValidationCase:
    if not isinstance(value, dict):
        raise ValueError("every validation case must be an object")
    required = {
        "case_id",
        "category",
        "kind",
        "title",
        "rationale",
        "command",
        "oracle",
        "expected_on_base",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"validation case lacks fields: {', '.join(missing)}")
    case = ValidationCase(**{key: str(value[key]).strip() for key in required})
    if not re.fullmatch(r"vp[0-9]{2}", case.case_id):
        raise ValueError(f"invalid validation case id: {case.case_id}")
    if case.category not in VALIDATION_CATEGORIES:
        raise ValueError(f"invalid validation category: {case.category}")
    if case.kind not in {"reproduction", "test"}:
        raise ValueError(f"invalid validation kind: {case.kind}")
    if case.expected_on_base not in {"fail", "pass"}:
        raise ValueError(f"invalid expected_on_base: {case.expected_on_base}")
    if not all((case.title, case.rationale, case.command, case.oracle)):
        raise ValueError(f"{case.case_id}: validation fields must not be empty")
    _validate_command(case)
    return case


def _validate_command(case: ValidationCase) -> None:
    command = case.command
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError as error:
        raise ValueError(f"{case.case_id}: command has invalid quoting: {error}") from error
    if "\n" in command or "$(" in command or "`" in command or any(
        token and set(token) <= set(";&|<>") for token in tokens
    ):
        raise ValueError(f"{case.case_id}: command uses forbidden shell composition")
    if _FORBIDDEN_COMMAND.search(command):
        raise ValueError(f"{case.case_id}: command may modify state or access external resources")
    matches_kind = bool(_TEST_COMMAND.search(command)) if case.kind == "test" else bool(
        _REPRODUCTION_COMMAND.search(command)
    )
    if not matches_kind:
        raise ValueError(f"{case.case_id}: command does not match declared kind {case.kind}")


def _case_result(case: ValidationCase, result: dict[str, Any]) -> ValidationCaseResult:
    output = _bounded(f"{result.get('exception_info', '')}\n{result.get('output', '')}".strip())
    returncode = int(result.get("returncode", -1))
    infra_failure = bool(_INFRA_FAILURE.search(output))
    skipped_only = bool(_SKIPPED_ONLY.search(output))
    failed_tests = case.kind == "test" and bool(_FAILED_TEST_EVIDENCE.search(output))
    positive_tests = case.kind != "test" or any(pattern.search(output) for pattern in _POSITIVE_TEST_EVIDENCE)
    execution_valid = not infra_failure and not skipped_only and (
        case.kind != "test" or failed_tests or positive_tests
    )
    passed = returncode == 0 and execution_valid and not failed_tests and positive_tests
    if infra_failure:
        detail = "Validation command had an infrastructure or discovery failure"
    elif skipped_only:
        detail = "Validation command executed no effective tests"
    elif not execution_valid:
        detail = "Test runner produced no trustworthy execution evidence"
    elif passed:
        detail = "Frozen validation case passed"
    else:
        detail = f"Frozen validation case failed with return code {returncode}"
    return ValidationCaseResult(
        case_id=case.case_id,
        category=case.category,
        kind=case.kind,
        command=case.command,
        passed=passed,
        execution_valid=execution_valid,
        detail=detail,
        returncode=returncode,
        output=output,
    )


def _extract_json(value: str) -> dict[str, Any]:
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("validation planner did not return a JSON object")
    try:
        payload = json.loads(value[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid validation plan JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("validation plan root must be an object")
    return payload


def _bounded(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    half = limit // 2
    return f"{value[:half]}\n... <truncated> ...\n{value[-half:]}"
