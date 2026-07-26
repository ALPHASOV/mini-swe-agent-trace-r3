"""End-to-end TRACE-R³ controller for one SWE-bench instance."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minisweagent import Environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_sb_environment
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent
from minisweagent.trace_r3.context import (
    RECOVERY_INSTANCE_TEMPLATE,
    RECOVERY_SYSTEM_TEMPLATE,
    build_recovery_task,
    graph_summary_for_manifest,
    render_gate_markdown,
)
from minisweagent.trace_r3.gate import GateEvaluator
from minisweagent.trace_r3.graph import OneHopGraphRetriever, extract_anchor_sources, extract_seed_terms
from minisweagent.trace_r3.types import CheckpointDecision, GateReport, GateState, TraceR3Config
from minisweagent.trace_r3.versioning import HybridVersionController, VersionEvent
from minisweagent.utils.log import logger
from minisweagent.utils.serialize import recursive_merge


@dataclass(frozen=True)
class AgentOutcome:
    exit_status: str
    submission: str
    model_name: str
    error: str = ""


@dataclass(frozen=True)
class TraceR3Result:
    instance_id: str
    model_name: str
    patch: str
    exit_status: str
    recovery_activated: bool
    selected_location: str


class DurableProgressTrackingAgent(ProgressTrackingAgent):
    """Atomically persist the latest complete trajectory after every interaction."""

    def save(self, path: Path | None, *extra_dicts) -> dict:
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f"{path.suffix}.tmp")
            temporary.write_text(json.dumps(data, indent=2))
            temporary.replace(path)
        return data


class ArtifactStore:
    """Durable, inspectable artifact layout for one instance."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def stage(self, epoch: int, checkpoint: int) -> Path:
        path = self.root / "recovery" / f"epoch-{epoch:02d}" / f"checkpoint-{checkpoint:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative: str | Path, data: Any) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        temporary.replace(path)
        return path

    def write_text(self, relative: str | Path, data: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(data)
        temporary.replace(path)
        return path

    def save_gate(self, directory: Path, report: GateReport) -> None:
        self.write_json(directory.relative_to(self.root) / "gate.json", report.to_dict(include_patch=False))
        self.write_text(directory.relative_to(self.root) / "gate.md", render_gate_markdown(report))
        self.write_text(directory.relative_to(self.root) / "candidate.patch", report.patch)


class TraceR3Controller:
    """Baseline-first controller; recovery code is unreachable before Gate G0."""

    def __init__(
        self,
        *,
        instance: dict[str, Any],
        output_dir: Path,
        config: dict[str, Any],
        progress_manager: RunBatchProgressManager,
    ):
        self.instance = instance
        self.instance_id = instance["instance_id"]
        self.problem_statement = instance["problem_statement"]
        self.config = config
        self.progress_manager = progress_manager
        self.trace_config = TraceR3Config.from_dict(config.get("trace_r3"))
        self.instance_root = output_dir / self.instance_id
        self.attempt_number, attempt_root = _allocate_attempt(self.instance_root)
        self.store = ArtifactStore(attempt_root)
        _write_latest_attempt(self.instance_root, self.attempt_number, attempt_root)
        self.gate = GateEvaluator(self.trace_config)
        self.graph = OneHopGraphRetriever(self.trace_config)
        self.versions = HybridVersionController(self.trace_config)
        self.manifest: dict[str, Any] = {
            "schema_version": "trace-r3-run/1",
            "instance_id": self.instance_id,
            "attempt": self.attempt_number,
            "started_at": time.time(),
            "phase": "starting",
            "recovery_activated": False,
            "activation_rule": "only_after_baseline_gate_is_not_green",
            "trace_r3_config": self.trace_config.to_dict(),
            "stages": [],
        }

    def run(self) -> TraceR3Result:
        env: Environment | None = None
        self.progress_manager.on_instance_start(self.instance_id)
        self._status("Starting unchanged baseline")
        self._save_manifest()
        try:
            env = get_sb_environment(self.config, self.instance)
            baseline_dir = self.store.root / "baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            baseline = self._run_agent(
                env=env,
                task=self.problem_statement,
                model_config=self.config.get("model", {}),
                agent_config=self.config.get("agent", {}),
                trajectory_path=baseline_dir / "baseline.traj.json",
                stage={"kind": "baseline", "enhancements_active": False},
            )
            baseline_gate = self.gate.evaluate(env, self._last_messages)
            self.store.save_gate(baseline_dir, baseline_gate)
            self.versions.observe_baseline(baseline_gate)
            self._record_stage(
                "baseline",
                baseline,
                baseline_gate,
                enhancements_active=False,
                location="baseline",
            )

            if baseline_gate.state is GateState.GREEN:
                result = TraceR3Result(
                    instance_id=self.instance_id,
                    model_name=baseline.model_name,
                    patch=baseline_gate.patch,
                    exit_status="BaselineGatePassed",
                    recovery_activated=False,
                    selected_location="baseline",
                )
                return self._finish(result)

            self.manifest["recovery_activated"] = True
            self.manifest["activation_gate"] = baseline_gate.to_dict(include_patch=False)
            self._save_manifest()

            current_report = baseline_gate
            current_graph = self._retrieve_graph(env, current_report)
            clean_base = False
            model_name = baseline.model_name

            while True:
                epoch = self.versions.epoch
                checkpoint = self.versions.checkpoint + 1
                stage_dir = self.store.stage(epoch, checkpoint)
                relative_stage = stage_dir.relative_to(self.store.root)
                self.store.write_json(relative_stage / "input_graph.json", current_graph)
                self.store.write_text(relative_stage / "rag_context.md", current_graph.get("rag_context", ""))

                recovery_task = build_recovery_task(
                    problem_statement=self.problem_statement,
                    failed_patch=current_report.patch,
                    gate_report=current_report,
                    graph_artifact=current_graph,
                    epoch=epoch,
                    checkpoint=checkpoint,
                    clean_base=clean_base,
                    sealed_reasons=self.versions.sealed_reasons,
                )
                self.store.write_text(relative_stage / "prompt.md", recovery_task)
                self._status(f"Recovery epoch {epoch}/{self.trace_config.max_epochs}, checkpoint {checkpoint}")

                recovery = self._run_agent(
                    env=env,
                    task=recovery_task,
                    model_config=self._recovery_model_config(),
                    agent_config=self._recovery_agent_config(),
                    trajectory_path=stage_dir / "recovery.traj.json",
                    stage={
                        "kind": "recovery",
                        "enhancements_active": True,
                        "epoch": epoch,
                        "checkpoint": checkpoint,
                        "reasoning_effort": "max",
                        "graph_depth": 1,
                    },
                )
                model_name = recovery.model_name
                report = self.gate.evaluate(env, self._last_messages)
                self.store.save_gate(stage_dir, report)
                event = self.versions.decide(
                    report,
                    patch=report.patch,
                    location=f"epoch-{epoch:02d}/checkpoint-{checkpoint:02d}",
                )
                self.store.write_json(relative_stage / "version_decision.json", event.to_dict())
                self.store.write_json("version_ledger.json", self.versions.to_dict())
                self._record_stage(
                    "recovery",
                    recovery,
                    report,
                    enhancements_active=True,
                    location=f"epoch-{epoch:02d}/checkpoint-{checkpoint:02d}",
                    graph=current_graph,
                    event=event,
                )

                if event.decision is CheckpointDecision.ACCEPT:
                    return self._finish(
                        TraceR3Result(
                            instance_id=self.instance_id,
                            model_name=model_name,
                            patch=report.patch,
                            exit_status="RecoveryGatePassed",
                            recovery_activated=True,
                            selected_location=f"epoch-{epoch:02d}/checkpoint-{checkpoint:02d}",
                        )
                    )

                if event.decision is CheckpointDecision.STOP:
                    return self._finish_best(model_name)

                current_report = report
                if event.decision is CheckpointDecision.STACK:
                    current_graph = self._retrieve_graph(env, report)
                    clean_base = False
                    continue

                self._status(f"Sealing epoch {epoch}; recreating clean B0 environment")
                failed_graph = self._retrieve_graph(env, report)
                self.store.write_json(
                    Path("recovery") / f"epoch-{epoch:02d}" / "sealed_epoch.json",
                    {
                        "reason": event.reason,
                        "last_checkpoint": checkpoint,
                        "discarded_patch_chain": report.patch,
                        "retained_anchor_functions": sorted(extract_anchor_sources(failed_graph)),
                    },
                )
                anchor_sources = extract_anchor_sources(failed_graph)
                _cleanup(env)
                env = get_sb_environment(self.config, self.instance)
                self.versions.begin_next_epoch()
                current_graph = self._retrieve_graph(env, report, anchor_sources=anchor_sources)
                clean_base = True
        except Exception as error:
            logger.error(f"TRACE-R³ failed for {self.instance_id}: {error}", exc_info=True)
            self.manifest["phase"] = "error"
            self.manifest["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            self._save_manifest()
            return self._finish_best(
                self.config.get("model", {}).get("model_name", ""),
                exit_status=type(error).__name__,
            )
        finally:
            if env is not None:
                _cleanup(env)

    @property
    def _last_messages(self) -> list[dict[str, Any]]:
        return self._agent.messages if getattr(self, "_agent", None) is not None else []

    def _run_agent(
        self,
        *,
        env: Environment,
        task: str,
        model_config: dict[str, Any],
        agent_config: dict[str, Any],
        trajectory_path: Path,
        stage: dict[str, Any],
    ) -> AgentOutcome:
        model = get_model(config=model_config)
        durable_agent_config = {**agent_config, "output_path": trajectory_path}
        self._agent = DurableProgressTrackingAgent(
            model,
            env,
            progress_manager=self.progress_manager,
            instance_id=self.instance_id,
            **durable_agent_config,
        )
        info: dict[str, Any] = {}
        error = ""
        try:
            info = self._agent.run(task)
        except Exception as exception:
            error = traceback.format_exc()
            info = {
                "exit_status": type(exception).__name__,
                "submission": "",
                "exception_str": str(exception),
            }
        finally:
            self._agent.save(
                trajectory_path,
                {
                    "info": {
                        "exit_status": info.get("exit_status", ""),
                        "submission": info.get("submission", ""),
                        "trace_r3": stage,
                    },
                    "instance_id": self.instance_id,
                },
            )
        return AgentOutcome(
            exit_status=info.get("exit_status", ""),
            submission=info.get("submission", ""),
            model_name=model.config.model_name,
            error=error,
        )

    def _retrieve_graph(
        self,
        env: Environment,
        report: GateReport,
        *,
        anchor_sources: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self._status("Building deterministic one-hop repository graph")
        terms = extract_seed_terms(self.problem_statement, report.patch)
        return self.graph.retrieve(env, seed_terms=terms, anchor_sources=anchor_sources)

    def _recovery_model_config(self) -> dict[str, Any]:
        baseline = self.config.get("model", {})
        merged = recursive_merge(baseline, self.trace_config.recovery_model)
        if merged.get("model_name") != baseline.get("model_name"):
            raise ValueError("Recovery must retain the baseline model_name")
        return merged

    def _recovery_agent_config(self) -> dict[str, Any]:
        return recursive_merge(
            self.config.get("agent", {}),
            {
                "system_template": RECOVERY_SYSTEM_TEMPLATE,
                "instance_template": RECOVERY_INSTANCE_TEMPLATE,
            },
            self.trace_config.recovery_agent,
        )

    def _record_stage(
        self,
        kind: str,
        outcome: AgentOutcome,
        report: GateReport,
        *,
        enhancements_active: bool,
        location: str,
        graph: dict[str, Any] | None = None,
        event: VersionEvent | None = None,
    ) -> None:
        stage = {
            "kind": kind,
            "location": location,
            "enhancements_active": enhancements_active,
            "exit_status": outcome.exit_status,
            "model_name": outcome.model_name,
            "gate": report.to_dict(include_patch=False),
            "error": outcome.error,
        }
        if graph is not None:
            stage["graph"] = graph_summary_for_manifest(graph)
        if event is not None:
            stage["version_decision"] = event.to_dict()
        self.manifest["stages"].append(stage)
        self.manifest["phase"] = kind
        self._save_manifest()

    def _finish_best(self, model_name: str, *, exit_status: str = "TraceR3Exhausted") -> TraceR3Result:
        return self._finish(
            TraceR3Result(
                instance_id=self.instance_id,
                model_name=model_name,
                patch=self.versions.best_patch,
                exit_status=exit_status,
                recovery_activated=self.manifest["recovery_activated"],
                selected_location=self.versions.best_location,
            )
        )

    def _finish(self, result: TraceR3Result) -> TraceR3Result:
        self.store.write_text("final/model.patch", result.patch)
        self.store.write_json(
            "final/selection.json",
            {
                "instance_id": result.instance_id,
                "model_name": result.model_name,
                "exit_status": result.exit_status,
                "recovery_activated": result.recovery_activated,
                "selected_location": result.selected_location,
                "patch_chars": len(result.patch),
                "version_ledger": self.versions.to_dict(),
            },
        )
        self.manifest.update(
            {
                "phase": "finished",
                "finished_at": time.time(),
                "final_exit_status": result.exit_status,
                "selected_location": result.selected_location,
                "patch_chars": len(result.patch),
            }
        )
        self._save_manifest()
        self.progress_manager.on_instance_end(self.instance_id, result.exit_status)
        return result

    def _status(self, value: str) -> None:
        self.progress_manager.update_instance_status(self.instance_id, value)
        self.manifest["status"] = value
        self.manifest["updated_at"] = time.time()

    def _save_manifest(self) -> None:
        self.store.write_json("run_manifest.json", self.manifest)


def _cleanup(env: Environment) -> None:
    cleanup = getattr(env, "cleanup", None)
    if callable(cleanup):
        cleanup()


def _allocate_attempt(instance_root: Path) -> tuple[int, Path]:
    instance_root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in instance_root.glob("attempt-*"):
        try:
            numbers.append(int(path.name.removeprefix("attempt-")))
        except ValueError:
            continue
    number = max(numbers, default=0) + 1
    return number, instance_root / f"attempt-{number:03d}"


def _write_latest_attempt(instance_root: Path, number: int, attempt_root: Path) -> None:
    path = instance_root / "latest_attempt.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "attempt": number,
                "path": attempt_root.relative_to(instance_root).as_posix(),
                "allocated_at": time.time(),
            },
            indent=2,
        )
    )
    temporary.replace(path)
