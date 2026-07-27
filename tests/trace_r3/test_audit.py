import hashlib
import json
import subprocess
import sys
from pathlib import Path

AUDIT_SCRIPT = Path(__file__).parents[2] / "scripts" / "audit_trace_r3_run.py"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _run_audit(output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )


def _finished_attempt(output: Path) -> Path:
    instance = output / "demo__case-1"
    attempt = instance / "attempt-001"
    checkpoint = attempt / "recovery" / "epoch-01" / "checkpoint-01"
    _write_json(instance / "latest_attempt.json", {"path": "attempt-001"})
    cases = [{"case_id": f"vp{index:02d}"} for index in range(1, 11)]
    frozen_plan = {
        "schema_version": "trace-r3-frozen-validation/1",
        "objective": "complete test plan",
        "cases": cases,
    }
    identity = hashlib.sha256(
        json.dumps(frozen_plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    validation_run = {
        "schema_version": "trace-r3-validation-run/1",
        "plan_identity": identity,
        "results": [{"case_id": case["case_id"]} for case in cases],
    }
    _write_json(
        attempt / "run_manifest.json",
        {
            "schema_version": "trace-r3-run/2",
            "phase": "finished",
            "instance_id": "demo__case-1",
            "trace_r3_config": {"validation_plan_min_cases": 10},
            "validation": {"plan_identity": identity, "case_count": len(cases)},
            "stages": [
                {
                    "kind": "baseline",
                    "location": "baseline",
                    "validation_plan_identity": identity,
                },
                {
                    "kind": "recovery",
                    "location": "epoch-01/checkpoint-01",
                    "validation_plan_identity": identity,
                },
            ],
        },
    )
    for name in ("rag_context.md", "frozen_plan.md"):
        path = attempt / "validation" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    (attempt / "validation" / "frozen_plan.sha256").write_text(f"{identity}\n")
    for name, value in (
        ("input_graph.json", {}),
        ("planner.traj.json", {"messages": []}),
        ("frozen_plan.json", frozen_plan),
        ("b0_results.json", validation_run),
    ):
        _write_json(attempt / "validation" / name, value)
    _write_json(attempt / "baseline" / "baseline.traj.json", {"messages": []})
    _write_json(attempt / "baseline" / "gate.json", {})
    _write_json(attempt / "baseline" / "validation_results.json", validation_run)
    (attempt / "baseline" / "candidate.patch").write_text("baseline")
    for name in (
        "prompt.md",
        "rag_context.md",
        "candidate.patch",
    ):
        path = checkpoint / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    for name, value in (
        ("input_graph.json", {}),
        ("recovery.traj.json", {"messages": []}),
        ("gate.json", {}),
        ("version_decision.json", {}),
        ("validation_results.json", validation_run),
    ):
        _write_json(checkpoint / name, value)
    _write_json(
        attempt / "final" / "selection.json",
        {"validation_plan_identity": identity},
    )
    (attempt / "final" / "model.patch").write_text("final patch")
    _write_json(
        output / "preds.json",
        {"demo__case-1": {"model_patch": "final patch"}},
    )
    return attempt


def test_audit_accepts_finished_attempt_and_counts_every_trajectory(tmp_path):
    output = tmp_path / "output"
    _finished_attempt(output)

    completed = _run_audit(output)
    report = json.loads(completed.stdout)

    assert completed.returncode == 0
    assert report == {
        "finished_attempts": 1,
        "incomplete_attempts": 0,
        "trajectories": 3,
        "errors": [],
        "warnings": [],
    }


def test_audit_rejects_a_round_that_omits_frozen_cases(tmp_path):
    output = tmp_path / "output"
    attempt = _finished_attempt(output)
    path = attempt / "baseline" / "validation_results.json"
    run = json.loads(path.read_text())
    run["results"].pop()
    _write_json(path, run)

    completed = _run_audit(output)
    report = json.loads(completed.stdout)

    assert completed.returncode == 1
    assert any("did not execute every frozen case exactly once" in error for error in report["errors"])


def test_audit_retains_and_reports_interrupted_partial_trajectory(tmp_path):
    output = tmp_path / "output"
    attempt = output / "demo__case-1" / "attempt-001"
    _write_json(
        attempt / "run_manifest.json",
        {"phase": "recovery", "instance_id": "demo__case-1", "stages": []},
    )
    _write_json(attempt / "baseline" / "baseline.traj.json", {"messages": [{"role": "assistant"}]})

    completed = _run_audit(output)
    report = json.loads(completed.stdout)

    assert completed.returncode == 0
    assert report["finished_attempts"] == 0
    assert report["incomplete_attempts"] == 1
    assert report["trajectories"] == 1
    assert report["errors"] == []
    assert "retained for audit" in report["warnings"][0]


def test_audit_rejects_corrupted_partial_trajectory(tmp_path):
    output = tmp_path / "output"
    attempt = output / "demo__case-1" / "attempt-001"
    _write_json(
        attempt / "run_manifest.json",
        {"phase": "recovery", "instance_id": "demo__case-1", "stages": []},
    )
    _write_json(attempt / "baseline" / "baseline.traj.json", {"messages": "not-a-list"})

    completed = _run_audit(output)
    report = json.loads(completed.stdout)

    assert completed.returncode == 1
    assert len(report["errors"]) == 1
    assert "invalid or missing messages" in report["errors"][0]
