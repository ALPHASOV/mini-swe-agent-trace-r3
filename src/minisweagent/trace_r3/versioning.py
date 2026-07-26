"""Hybrid incremental/rollback version controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from minisweagent.trace_r3.types import CheckpointDecision, GateReport, GateState, TraceR3Config


@dataclass(frozen=True)
class VersionEvent:
    epoch: int
    checkpoint: int
    decision: CheckpointDecision
    reason: str
    gate_state: GateState
    gate_score: tuple[int, int, int]
    failure_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision"] = self.decision.value
        data["gate_state"] = self.gate_state.value
        data["gate_score"] = list(self.gate_score)
        return data


class HybridVersionController:
    """Allow evidence-improving patch stacks, otherwise seal and return to B0."""

    def __init__(self, config: TraceR3Config):
        self.config = config
        self.epoch = 1
        self.checkpoint = 0
        self.events: list[VersionEvent] = []
        self.sealed_reasons: list[str] = []
        self._last_report: GateReport | None = None
        self._fingerprints_in_epoch: set[str] = set()
        self._non_improving = 0
        self.best_report: GateReport | None = None
        self.best_patch = ""
        self.best_location = "baseline"

    def observe_baseline(self, report: GateReport) -> None:
        """Record C0 without counting it as a recovery checkpoint."""

        self._last_report = report
        self._fingerprints_in_epoch.add(report.fingerprint())
        self._consider_best(report, location="baseline")

    def decide(self, report: GateReport, *, patch: str, location: str) -> VersionEvent:
        self.checkpoint += 1
        self._consider_best(report, patch=patch, location=location)

        if report.state is GateState.GREEN:
            return self._event(CheckpointDecision.ACCEPT, "All gate checks passed", report)

        hard_reason = self._hard_rollback_reason(report)
        repeated = report.fingerprint() in self._fingerprints_in_epoch
        improved = self._last_report is None or report.score > self._last_report.score
        self._non_improving = 0 if improved else self._non_improving + 1

        if repeated:
            hard_reason = "Repeated the same gate failure signature"
        elif self._regressed(report):
            hard_reason = "A previously passing critical check regressed"
        elif self._non_improving >= self.config.max_non_improving_checkpoints:
            hard_reason = f"No gate improvement for {self._non_improving} consecutive checkpoints"
        elif self.checkpoint >= self.config.max_checkpoints_per_epoch:
            hard_reason = f"Reached the per-epoch checkpoint limit ({self.config.max_checkpoints_per_epoch})"

        self._fingerprints_in_epoch.add(report.fingerprint())
        self._last_report = report

        if hard_reason:
            self.sealed_reasons.append(f"Epoch {self.epoch}: {hard_reason}")
            decision = CheckpointDecision.STOP if self.epoch >= self.config.max_epochs else CheckpointDecision.ROLLBACK
            return self._event(decision, hard_reason, report)
        return self._event(CheckpointDecision.STACK, "Gate evidence permits another incremental refinement", report)

    def begin_next_epoch(self) -> None:
        if self.epoch >= self.config.max_epochs:
            raise RuntimeError("Cannot advance beyond configured epoch limit")
        self.epoch += 1
        self.checkpoint = 0
        self._last_report = None
        self._fingerprints_in_epoch.clear()
        self._non_improving = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": {
                "mode": "hybrid_incremental_with_sealed_rollback",
                "max_epochs": self.config.max_epochs,
                "max_checkpoints_per_epoch": self.config.max_checkpoints_per_epoch,
                "max_non_improving_checkpoints": self.config.max_non_improving_checkpoints,
            },
            "current_epoch": self.epoch,
            "current_checkpoint": self.checkpoint,
            "sealed_reasons": self.sealed_reasons,
            "best_location": self.best_location,
            "best_gate": self.best_report.to_dict(include_patch=False) if self.best_report else None,
            "events": [event.to_dict() for event in self.events],
        }

    def _event(self, decision: CheckpointDecision, reason: str, report: GateReport) -> VersionEvent:
        event = VersionEvent(
            epoch=self.epoch,
            checkpoint=self.checkpoint,
            decision=decision,
            reason=reason,
            gate_state=report.state,
            gate_score=report.score,
            failure_fingerprint=report.fingerprint(),
        )
        self.events.append(event)
        return event

    def _hard_rollback_reason(self, report: GateReport) -> str:
        hard_checks = {"patch_scope", "no_test_modifications", "diff_check", "python_syntax"}
        failures = [name for name in report.failed_checks if name in hard_checks]
        return f"Hard safety check failed: {', '.join(failures)}" if failures else ""

    def _regressed(self, report: GateReport) -> bool:
        if self._last_report is None:
            return False
        previous = {check.name: check for check in self._last_report.checks}
        return any(
            check.critical
            and check.passed is False
            and check.name in previous
            and previous[check.name].passed is True
            for check in report.checks
        )

    def _consider_best(self, report: GateReport, *, patch: str = "", location: str) -> None:
        if self.best_report is None or report.score > self.best_report.score:
            self.best_report = report
            self.best_patch = patch or report.patch
            self.best_location = location
