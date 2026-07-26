import subprocess

from minisweagent.environments.local import LocalEnvironment
from minisweagent.trace_r3.gate import GateEvaluator, extract_validation_commands
from minisweagent.trace_r3.types import GateState, TraceR3Config


def _repository(tmp_path):
    source = tmp_path / "maths.py"
    source.write_text("def increment(value):\n    return value + 1\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "maths.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
        cwd=tmp_path,
        check=True,
    )
    return source


def _messages(command: str) -> list[dict]:
    return [{"role": "assistant", "extra": {"actions": [{"command": command}]}}]


def test_gate_is_amber_with_patch_and_replayed_reproduction_only(tmp_path):
    source = _repository(tmp_path)
    source.write_text("def increment(value):\n    return value + 2\n")
    env = LocalEnvironment(cwd=str(tmp_path))

    report = GateEvaluator(TraceR3Config()).evaluate(
        env,
        _messages('python -c "from maths import increment; assert increment(1) == 3"'),
    )

    assert report.state is GateState.AMBER
    assert report.changed_files == ("maths.py",)
    assert report.tested_commands
    assert report.patch


def test_gate_is_green_with_patch_and_real_test(tmp_path):
    source = _repository(tmp_path)
    source.write_text("def increment(value):\n    return value + 2\n")
    (tmp_path / "test_maths.py").write_text(
        "from maths import increment\n\n"
        "def test_increment():\n"
        "    assert increment(1) == 3\n"
    )

    report = GateEvaluator(TraceR3Config()).evaluate(
        LocalEnvironment(cwd=str(tmp_path)),
        _messages("python -m pytest -q test_maths.py"),
    )

    assert report.state is GateState.GREEN
    assert any(check.name == "replay_test" and check.passed for check in report.checks)


def test_gate_does_not_misread_ten_passed_as_zero_passed(tmp_path):
    source = _repository(tmp_path)
    source.write_text("def increment(value):\n    return value + 2\n")

    report = GateEvaluator(TraceR3Config()).evaluate(
        LocalEnvironment(cwd=str(tmp_path)),
        _messages('python -c "assert True; print(\'10 passed\')"'),
    )

    assert report.state is GateState.AMBER
    assert any(check.name == "replay_reproduction" and check.passed for check in report.checks)


def test_gate_is_amber_without_behavioral_evidence(tmp_path):
    source = _repository(tmp_path)
    source.write_text("def increment(value):\n    return value + 2\n")

    report = GateEvaluator(TraceR3Config()).evaluate(LocalEnvironment(cwd=str(tmp_path)), [])

    assert report.state is GateState.AMBER
    assert report.unknown_checks == ("behavioral_evidence",)


def test_gate_is_red_for_invalid_python(tmp_path):
    source = _repository(tmp_path)
    source.write_text("def increment(value)\n    return value + 2\n")

    report = GateEvaluator(TraceR3Config()).evaluate(
        LocalEnvironment(cwd=str(tmp_path)),
        _messages('python -c "assert True"'),
    )

    assert report.state is GateState.RED
    assert "python_syntax" in report.failed_checks


def test_masked_commands_are_not_replayed():
    messages = _messages("python -m pytest -q || true")

    assert extract_validation_commands(messages, limit=3) == []


def test_arbitrary_success_fallback_is_not_replayed_as_a_test():
    messages = _messages(
        'python -m pytest -q || python -c "import package; print(package.__file__)"'
    )

    assert extract_validation_commands(messages, limit=3) == []


def test_import_or_install_pytest_is_not_a_test_command():
    messages = _messages('python -c "import pytest" 2>&1 || pip install pytest')

    assert extract_validation_commands(messages, limit=3) == []


def test_test_command_after_cd_is_detected():
    messages = _messages("cd /testbed && python -m pytest -q sympy/core/tests/test_numbers.py")

    candidates = extract_validation_commands(messages, limit=3)

    assert len(candidates) == 1
    assert candidates[0].kind == "test"


def test_inline_native_test_api_is_detected():
    messages = _messages(
        'cd /testbed && python -c "import sympy; sympy.test(\'sympy/printing/tests/\')"'
    )

    candidates = extract_validation_commands(messages, limit=3)

    assert len(candidates) == 1
    assert candidates[0].kind == "test"


def test_gate_rejects_zero_exit_test_without_execution_summary(tmp_path):
    source = _repository(tmp_path)
    source.write_text("def increment(value):\n    return value + 2\n")

    report = GateEvaluator(TraceR3Config()).evaluate(
        LocalEnvironment(cwd=str(tmp_path)),
        _messages('python -m unittest -q 2>/dev/null; printf "runner loaded\\n"'),
    )

    assert report.state is GateState.RED
    replay = next(check for check in report.checks if check.name == "replay_test")
    assert replay.passed is False
    assert "no positive evidence" in replay.detail


def test_gate_rejects_failure_summary_even_when_shell_returns_zero(tmp_path):
    source = _repository(tmp_path)
    source.write_text("def increment(value):\n    return value + 2\n")

    report = GateEvaluator(TraceR3Config()).evaluate(
        LocalEnvironment(cwd=str(tmp_path)),
        _messages('python -m unittest -q 2>/dev/null; printf "96 passed, 1 failed\\n"'),
    )

    assert report.state is GateState.RED
    replay = next(check for check in report.checks if check.name == "replay_test")
    assert replay.passed is False
    assert "reports failures" in replay.detail
