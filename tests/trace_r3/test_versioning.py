import pytest

from minisweagent.trace_r3.types import (
    CheckpointDecision,
    GateCheck,
    GateReport,
    GateState,
    TraceR3Config,
)
from minisweagent.trace_r3.versioning import HybridVersionController


def _report(state: GateState, *, replay_passed: bool, patch: str) -> GateReport:
    return GateReport(
        state=state,
        checks=(
            GateCheck("patch_nonempty", True, "patch"),
            GateCheck("diff_check", True, "diff"),
            GateCheck("python_syntax", True, "syntax"),
            GateCheck("replay_test", replay_passed, "test", returncode=0 if replay_passed else 1),
        ),
        patch=patch,
        changed_files=("source.py",),
        added_lines=1,
        deleted_lines=1,
    )


def test_version_controller_stacks_improvement_then_accepts():
    controller = HybridVersionController(TraceR3Config())
    controller.observe_baseline(_report(GateState.RED, replay_passed=False, patch="A"))
    amber = GateReport(
        state=GateState.AMBER,
        checks=(
            GateCheck("patch_nonempty", True, "patch"),
            GateCheck("diff_check", True, "diff"),
            GateCheck("python_syntax", True, "syntax"),
            GateCheck("behavioral_evidence", None, "missing", critical=False),
        ),
        patch="A+B",
        changed_files=("source.py",),
        added_lines=2,
        deleted_lines=1,
    )

    first = controller.decide(amber, patch=amber.patch, location="e1/c1")
    green = _report(GateState.GREEN, replay_passed=True, patch="A+B+C")
    second = controller.decide(green, patch=green.patch, location="e1/c2")

    assert first.decision is CheckpointDecision.STACK
    assert second.decision is CheckpointDecision.ACCEPT
    assert controller.best_patch == "A+B+C"


def test_repeated_failure_seals_epoch_and_third_epoch_stops():
    controller = HybridVersionController(TraceR3Config(max_epochs=3))
    failure = _report(GateState.RED, replay_passed=False, patch="A")
    controller.observe_baseline(failure)

    event = controller.decide(failure, patch="A+B", location="e1/c1")
    assert event.decision is CheckpointDecision.ROLLBACK
    controller.begin_next_epoch()

    event = controller.decide(failure, patch="C", location="e2/c1")
    assert event.decision is CheckpointDecision.STACK
    repeated = controller.decide(failure, patch="C+D", location="e2/c2")
    assert repeated.decision is CheckpointDecision.ROLLBACK
    controller.begin_next_epoch()

    event = controller.decide(failure, patch="E", location="e3/c1")
    assert event.decision is CheckpointDecision.STACK
    final = controller.decide(failure, patch="E+F", location="e3/c2")
    assert final.decision is CheckpointDecision.STOP


def test_failure_fingerprint_ignores_volatile_command_output():
    first = GateReport(
        state=GateState.RED,
        checks=(
            GateCheck(
                "replay_test",
                False,
                "failed",
                returncode=1,
                output="/tmp/pytest-of-user/pytest-1/test_case failed in 0.31s",
            ),
        ),
        patch="A",
        changed_files=("source.py",),
        added_lines=1,
        deleted_lines=0,
    )
    second = GateReport(
        state=GateState.RED,
        checks=(
            GateCheck(
                "replay_test",
                False,
                "failed",
                returncode=1,
                output="/tmp/pytest-of-runner/pytest-8/test_case failed in 1.72s",
            ),
        ),
        patch="B",
        changed_files=("source.py",),
        added_lines=2,
        deleted_lines=1,
    )

    assert first.fingerprint() == second.fingerprint()


def test_graph_depth_is_fixed_to_one_hop():
    with pytest.raises(ValueError, match="one-hop"):
        TraceR3Config(graph_depth=2)
