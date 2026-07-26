import json
import subprocess

from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import make_output
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.trace_r3.controller import TraceR3Controller


class ResettingLocalEnvironment(LocalEnvironment):
    """Test environment that emulates a fresh task image on every construction."""

    generations = 0
    clean_snapshots: list[str] = []

    def __init__(self, **kwargs):
        repository = kwargs["cwd"]
        subprocess.run(["git", "restore", "."], cwd=repository, check=True)
        type(self).generations += 1
        type(self).clean_snapshots.append(
            subprocess.run(
                ["git", "diff", "--binary"],
                cwd=repository,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout
        )
        super().__init__(**kwargs)


def _repository(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "maths.py").write_text("def increment(value):\n    return value + 1\n")
    (repository / "test_maths.py").write_text(
        "from maths import increment\n\n"
        "def test_increment():\n"
        "    assert increment(1) == 3\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "maths.py", "test_maths.py"], cwd=repository, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
        cwd=repository,
        check=True,
    )
    return repository


def _write_action(increment: int) -> dict:
    command = (
        "python -c \"from pathlib import Path; "
        "Path('maths.py').write_text('def increment(value):\\\\n"
        f"    return value + {increment}\\\\n')\""
    )
    return make_output("editing", [{"command": command}], cost=0.01)


def _reproduction_action() -> dict:
    command = 'python -c "from maths import increment; assert increment(1) == 3"'
    return make_output("validating", [{"command": command}], cost=0.01)


def _test_action() -> dict:
    return make_output("testing", [{"command": "python -m pytest -q test_maths.py"}], cost=0.01)


def _submit_action() -> dict:
    command = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff --binary"
    return make_output("submitting", [{"command": command}], cost=0.01)


def _config(repository, baseline_outputs, recovery_outputs=None):
    config = {
        "agent": {
            "system_template": "baseline system",
            "instance_template": "{{task}}",
            "step_limit": 10,
            "cost_limit": 3,
        },
        "environment": {
            "environment_class": "local",
            "cwd": str(repository),
            "timeout": 30,
        },
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": baseline_outputs,
            "cost_per_call": 0.01,
        },
    }
    if recovery_outputs is not None:
        config["trace_r3"] = {
            "recovery_model": {
                "model_name": "deterministic",
                "outputs": recovery_outputs,
            }
        }
    return config


def _controller(tmp_path, config):
    return TraceR3Controller(
        instance={"instance_id": "demo__case-1", "problem_statement": "increment(1) must return 3"},
        output_dir=tmp_path / "outputs",
        config=config,
        progress_manager=RunBatchProgressManager(1),
    )


def test_green_baseline_never_activates_recovery(tmp_path, reset_global_stats):
    repository = _repository(tmp_path)
    outputs = [_write_action(2), _reproduction_action(), _test_action(), _submit_action()]
    controller = _controller(tmp_path, _config(repository, outputs))

    result = controller.run()

    instance_root = tmp_path / "outputs" / "demo__case-1"
    instance_dir = instance_root / "attempt-001"
    manifest = json.loads((instance_dir / "run_manifest.json").read_text())
    assert result.exit_status == "BaselineGatePassed"
    assert result.recovery_activated is False
    assert manifest["recovery_activated"] is False
    assert manifest["stages"][0]["enhancements_active"] is False
    assert not (instance_dir / "recovery").exists()
    assert (instance_dir / "baseline" / "baseline.traj.json").is_file()
    assert (instance_dir / "final" / "model.patch").read_text() == result.patch
    assert json.loads((instance_root / "latest_attempt.json").read_text())["path"] == "attempt-001"


def test_failed_baseline_activates_graph_and_recovery(tmp_path, reset_global_stats):
    repository = _repository(tmp_path)
    baseline = [_write_action(0), _reproduction_action(), _test_action(), _submit_action()]
    recovery = [_write_action(2), _reproduction_action(), _test_action(), _submit_action()]
    controller = _controller(tmp_path, _config(repository, baseline, recovery))

    result = controller.run()

    instance_dir = tmp_path / "outputs" / "demo__case-1" / "attempt-001"
    checkpoint = instance_dir / "recovery" / "epoch-01" / "checkpoint-01"
    manifest = json.loads((instance_dir / "run_manifest.json").read_text())
    graph = json.loads((checkpoint / "input_graph.json").read_text())
    assert result.exit_status == "RecoveryGatePassed"
    assert result.recovery_activated is True
    assert manifest["recovery_activated"] is True
    assert manifest["stages"][0]["enhancements_active"] is False
    assert manifest["stages"][1]["enhancements_active"] is True
    assert graph["depth"] == 1
    assert graph["llm_consumption"]["format"] == "repograph_one_hop_with_inlinecoder_projection"
    assert (checkpoint / "recovery.traj.json").is_file()
    assert (checkpoint / "rag_context.md").read_text() in (checkpoint / "prompt.md").read_text()
    assert "return value + 2" in result.patch


def test_recovery_keeps_model_and_enables_max_reasoning(tmp_path):
    repository = _repository(tmp_path)
    config = _config(repository, [_submit_action()])
    config["model"]["model_name"] = "deepseek/deepseek-v4-flash"
    controller = _controller(tmp_path, config)

    recovery_config = controller._recovery_model_config()

    assert recovery_config["model_name"] == "deepseek/deepseek-v4-flash"
    assert recovery_config["model_class"] == "minisweagent.trace_r3.deepseek.ReasoningReplayLitellmModel"
    assert recovery_config["model_kwargs"]["thinking"] == {"type": "enabled"}
    assert recovery_config["model_kwargs"]["reasoning_effort"] == "max"


def test_repeated_failures_recreate_clean_b0_and_stop_after_three_epochs(tmp_path, reset_global_stats):
    repository = _repository(tmp_path)
    failure = [_write_action(0), _reproduction_action(), _test_action(), _submit_action()]
    config = _config(repository, failure, failure)
    config["environment"]["environment_class"] = (
        "tests.trace_r3.test_controller.ResettingLocalEnvironment"
    )
    ResettingLocalEnvironment.generations = 0
    ResettingLocalEnvironment.clean_snapshots = []

    result = _controller(tmp_path, config).run()

    attempt = tmp_path / "outputs" / "demo__case-1" / "attempt-001"
    ledger = json.loads((attempt / "version_ledger.json").read_text())
    second_epoch_graph = json.loads(
        (attempt / "recovery" / "epoch-02" / "checkpoint-01" / "input_graph.json").read_text()
    )
    assert result.exit_status == "TraceR3Exhausted"
    assert ResettingLocalEnvironment.generations == 3
    assert ResettingLocalEnvironment.clean_snapshots == ["", "", ""]
    assert second_epoch_graph["changed_ranges"] == {}
    assert second_epoch_graph["anchor_provenance"] == "external_failed_candidate"
    assert "return value + 0" in second_epoch_graph["anchor_sources"]["maths.increment"]
    assert [event["decision"] for event in ledger["events"]] == [
        "rollback",
        "rollback",
        "stop",
    ]
    assert all("identical" in event["reason"] for event in ledger["events"])
