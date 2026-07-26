#!/usr/bin/env python3
"""Audit TRACE-R³ trajectories and checkpoint artifacts."""

from __future__ import annotations

import argparse
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
            _require(attempt / "baseline" / "baseline.traj.json", result)
            _require(attempt / "baseline" / "gate.json", result)
            _require(attempt / "baseline" / "candidate.patch", result)
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
