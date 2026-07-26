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


def test_gate_is_green_with_patch_and_replayed_reproduction(tmp_path):
    source = _repository(tmp_path)
    source.write_text("def increment(value):\n    return value + 2\n")
    env = LocalEnvironment(cwd=str(tmp_path))

    report = GateEvaluator(TraceR3Config()).evaluate(
        env,
        _messages('python -c "from maths import increment; assert increment(1) == 3"'),
    )

    assert report.state is GateState.GREEN
    assert report.changed_files == ("maths.py",)
    assert report.tested_commands
    assert report.patch


def test_gate_does_not_misread_ten_passed_as_zero_passed(tmp_path):
    source = _repository(tmp_path)
    source.write_text("def increment(value):\n    return value + 2\n")

    report = GateEvaluator(TraceR3Config()).evaluate(
        LocalEnvironment(cwd=str(tmp_path)),
        _messages('python -c "assert True; print(\'10 passed\')"'),
    )

    assert report.state is GateState.GREEN


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
