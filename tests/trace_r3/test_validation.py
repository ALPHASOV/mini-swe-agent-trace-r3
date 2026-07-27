import json
from pathlib import Path

import pytest

from minisweagent.environments.local import LocalEnvironment
from minisweagent.trace_r3.types import TraceR3Config
from minisweagent.trace_r3.validation import (
    calibration_errors,
    execute_validation_plan,
    parse_validation_plan,
)


def _payload() -> dict:
    categories = [
        "direct_reproduction",
        "direct_reproduction",
        "boundary",
        "boundary",
        "metamorphic",
        "metamorphic",
        "cross_consumer",
        "cross_consumer",
        "focused_regression",
        "broad_regression",
    ]
    cases = []
    for index, category in enumerate(categories, 1):
        kind = "test" if category.endswith("regression") else "reproduction"
        command = (
            f"python -m unittest -q test_case_{index}.py"
            if kind == "test"
            else f'python -c "assert {index} == {index}"'
        )
        cases.append(
            {
                "case_id": f"vp{index:02d}",
                "category": category,
                "kind": kind,
                "title": f"case {index}",
                "rationale": f"risk {index}",
                "command": command,
                "oracle": f"oracle {index}",
                "expected_on_base": "pass",
            }
        )
    cases[0]["command"] = 'python -c "assert 1 == 2"'
    cases[0]["expected_on_base"] = "fail"
    return {
        "schema_version": "trace-r3-frozen-validation/1",
        "objective": "exercise a complete frozen plan",
        "cases": cases,
    }


def test_plan_requires_broad_unique_safe_coverage():
    plan = parse_validation_plan(json.dumps(_payload()), TraceR3Config())

    assert len(plan.cases) == 10
    assert plan.identity == parse_validation_plan(json.dumps(_payload()), TraceR3Config()).identity

    unsafe = _payload()
    unsafe["cases"][1]["command"] = "rm -f source.py"
    with pytest.raises(ValueError, match="modify state"):
        parse_validation_plan(json.dumps(unsafe), TraceR3Config())

    python_write = _payload()
    python_write["cases"][1]["command"] = (
        "python -c \"open('source.py', 'w').write('changed'); assert True\""
    )
    with pytest.raises(ValueError, match="modify state"):
        parse_validation_plan(json.dumps(python_write), TraceR3Config())

    composed = _payload()
    composed["cases"][1]["command"] = 'python -c "assert True"; true'
    with pytest.raises(ValueError, match="shell composition"):
        parse_validation_plan(json.dumps(composed), TraceR3Config())

    quoted_semicolon = _payload()
    quoted_semicolon["cases"][1]["command"] = 'python -c "assert True; assert 2 == 2"'
    parse_validation_plan(json.dumps(quoted_semicolon), TraceR3Config())

    fake_regression = _payload()
    fake_regression["cases"][8]["kind"] = "reproduction"
    fake_regression["cases"][8]["command"] = 'python -c "assert True"'
    with pytest.raises(ValueError, match="must use native test commands"):
        parse_validation_plan(json.dumps(fake_regression), TraceR3Config())


def test_every_frozen_case_executes_and_calibrates_before_patch(tmp_path: Path):
    for index in (9, 10):
        (tmp_path / f"test_case_{index}.py").write_text(
            "import unittest\n\n"
            "class TestCase(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertTrue(True)\n"
        )
    plan = parse_validation_plan(json.dumps(_payload()), TraceR3Config())
    run = execute_validation_plan(
        LocalEnvironment(cwd=str(tmp_path)),
        plan,
        phase="b0",
        timeout=30,
    )

    assert len(run.results) == 10
    assert calibration_errors(plan, run) == []
    assert run.results[0].passed is False
    assert all(result.execution_valid for result in run.results)
    assert len(run.gate_checks()) == 11
