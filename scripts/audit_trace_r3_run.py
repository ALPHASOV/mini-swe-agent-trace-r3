#!/usr/bin/env python3
"""Audit TRACE-R³ trajectories and checkpoint artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AuditResult:
    finished_attempts: int = 0
    incomplete_attempts: int = 0
    trajectories: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def audit(output_dir: Path) -> AuditResult:
    result = AuditResult()
    predictions = _read_json(output_dir / "preds.json", default={})
    for instance_root in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        latest = _read_json(instance_root / "latest_attempt.json", default={}).get("path", "")
        for attempt in sorted(instance_root.glob("attempt-*")):
            manifest_path = attempt / "run_manifest.json"
            manifest = _read_json(manifest_path)
            if not manifest:
                result.warnings.append(f"{attempt}: no run_manifest.json (possibly interrupted before first write)")
                result.incomplete_attempts += 1
                continue
            if manifest.get("phase") != "finished":
                result.incomplete_attempts += 1
                result.warnings.append(f"{attempt}: incomplete phase={manifest.get('phase')!r}; retained for audit")
                _count_existing_trajectories(attempt, result)
                continue
            result.finished_attempts += 1
            if manifest.get("schema_version") == "trace-r3-run/2":
                for name in (
                    "input_graph.json",
                    "rag_context.md",
                    "planner.traj.json",
                    "frozen_plan.json",
                    "frozen_plan.md",
                    "frozen_plan.sha256",
                    "b0_results.json",
                ):
                    _require(attempt / "validation" / name, result)
                _audit_frozen_validation(attempt, manifest, result)
            _require(attempt / "baseline" / "baseline.traj.json", result)
            _require(attempt / "baseline" / "gate.json", result)
            _require(attempt / "baseline" / "candidate.patch", result)
            if manifest.get("schema_version") == "trace-r3-run/2":
                _require(attempt / "baseline" / "validation_results.json", result)
            for stage in manifest.get("stages", []):
                if stage.get("kind") != "recovery":
                    continue
                location = stage.get("location", "")
                checkpoint = attempt / "recovery" / location
                for name in (
                    "prompt.md",
                    "input_graph.json",
                    "rag_context.md",
                    "recovery.traj.json",
                    "gate.json",
                    "candidate.patch",
                    "version_decision.json",
                ):
                    _require(checkpoint / name, result)
                if manifest.get("schema_version") == "trace-r3-run/2":
                    _require(checkpoint / "validation_results.json", result)
            selection_path = attempt / "final" / "selection.json"
            patch_path = attempt / "final" / "model.patch"
            _require(selection_path, result)
            _require(patch_path, result)
            _count_existing_trajectories(attempt, result)

            instance_id = manifest.get("instance_id", instance_root.name)
            if attempt.name == latest and instance_id in predictions and patch_path.is_file():
                expected = predictions[instance_id].get("model_patch", "")
                if patch_path.read_text() != expected:
                    result.errors.append(f"{attempt}: final/model.patch does not match preds.json")
    return result


def _count_existing_trajectories(attempt: Path, result: AuditResult) -> None:
    for trajectory in attempt.rglob("*.traj.json"):
        result.trajectories += 1
        data = _read_json(trajectory)
        if not data or not isinstance(data.get("messages"), list):
            result.errors.append(f"{trajectory}: invalid or missing messages")


def _require(path: Path, result: AuditResult) -> None:
    if not path.is_file():
        result.errors.append(f"{path}: required artifact missing")


def _audit_frozen_validation(attempt: Path, manifest: dict, result: AuditResult) -> None:
    plan_path = attempt / "validation" / "frozen_plan.json"
    plan = _read_json(plan_path)
    if not isinstance(plan, dict):
        if plan_path.is_file():
            result.errors.append(f"{plan_path}: invalid frozen validation plan")
        return
    raw_cases = plan.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        result.errors.append(f"{plan_path}: frozen validation plan has no cases")
        return
    case_ids = [case.get("case_id") for case in raw_cases if isinstance(case, dict)]
    if len(case_ids) != len(raw_cases) or any(not isinstance(case_id, str) for case_id in case_ids):
        result.errors.append(f"{plan_path}: invalid validation case identifiers")
        return
    if len(case_ids) != len(set(case_ids)):
        result.errors.append(f"{plan_path}: duplicate validation case identifiers")
        return
    minimum_cases = manifest.get("trace_r3_config", {}).get("validation_plan_min_cases", 10)
    if not isinstance(minimum_cases, int) or len(case_ids) < minimum_cases:
        result.errors.append(f"{plan_path}: fewer cases than the configured validation minimum")

    identity = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    identity_path = attempt / "validation" / "frozen_plan.sha256"
    if identity_path.is_file() and identity_path.read_text().strip() != identity:
        result.errors.append(f"{identity_path}: does not match frozen_plan.json")

    validation_manifest = manifest.get("validation", {})
    if validation_manifest.get("plan_identity") != identity:
        result.errors.append(f"{attempt / 'run_manifest.json'}: validation plan identity mismatch")
    if validation_manifest.get("case_count") != len(case_ids):
        result.errors.append(f"{attempt / 'run_manifest.json'}: validation case count mismatch")

    _audit_validation_run(attempt / "validation" / "b0_results.json", identity, case_ids, result)
    _audit_validation_run(attempt / "baseline" / "validation_results.json", identity, case_ids, result)
    for stage in manifest.get("stages", []):
        if stage.get("validation_plan_identity") != identity:
            result.errors.append(
                f"{attempt / 'run_manifest.json'}: stage {stage.get('location', '')!r} "
                "uses a different validation plan"
            )
        if stage.get("kind") == "recovery":
            _audit_validation_run(
                attempt / "recovery" / stage.get("location", "") / "validation_results.json",
                identity,
                case_ids,
                result,
            )

    selection_path = attempt / "final" / "selection.json"
    selection = _read_json(selection_path)
    if isinstance(selection, dict) and selection.get("validation_plan_identity") != identity:
        result.errors.append(f"{selection_path}: validation plan identity mismatch")


def _audit_validation_run(
    path: Path,
    identity: str,
    expected_case_ids: list[str],
    result: AuditResult,
) -> None:
    run = _read_json(path)
    if not isinstance(run, dict):
        if path.is_file():
            result.errors.append(f"{path}: invalid validation results")
        return
    if run.get("plan_identity") != identity:
        result.errors.append(f"{path}: frozen plan identity mismatch")
    raw_results = run.get("results")
    if not isinstance(raw_results, list):
        result.errors.append(f"{path}: validation results are missing")
        return
    actual_case_ids = [
        case.get("case_id") for case in raw_results if isinstance(case, dict)
    ]
    if actual_case_ids != expected_case_ids:
        result.errors.append(f"{path}: did not execute every frozen case exactly once")


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.output_dir)
    print(
        json.dumps(
            {
                "finished_attempts": result.finished_attempts,
                "incomplete_attempts": result.incomplete_attempts,
                "trajectories": result.trajectories,
                "errors": result.errors,
                "warnings": result.warnings,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    raise SystemExit(1 if result.errors else 0)


if __name__ == "__main__":
    main()
