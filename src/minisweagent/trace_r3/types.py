"""Shared data types for TRACE-R³."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class GateState(str, Enum):
    """Traffic-light outcome of an evidence gate."""

    GREEN = "green"
    AMBER = "amber"
    RED = "red"


class CheckpointDecision(str, Enum):
    """Version-controller action after evaluating a checkpoint."""

    ACCEPT = "accept"
    STACK = "stack"
    ROLLBACK = "rollback"
    STOP = "stop"


@dataclass(frozen=True)
class GateCheck:
    """One deterministic validation check."""

    name: str
    passed: bool | None
    detail: str
    critical: bool = True
    command: str = ""
    returncode: int | None = None
    output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateReport:
    """Complete gate result and the patch it evaluated."""

    state: GateState
    checks: tuple[GateCheck, ...]
    patch: str
    changed_files: tuple[str, ...]
    added_lines: int
    deleted_lines: int
    tested_commands: tuple[str, ...] = ()
    reason: str = ""

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if check.passed is False)

    @property
    def unknown_checks(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if check.passed is None)

    @property
    def score(self) -> tuple[int, int, int]:
        """Lexicographic quality score used only to retain the best failed candidate."""

        state_score = {GateState.RED: 0, GateState.AMBER: 1, GateState.GREEN: 2}[self.state]
        passed = sum(check.passed is True for check in self.checks)
        failed = sum(check.passed is False for check in self.checks)
        return state_score, passed, -failed

    def fingerprint(self) -> str:
        """Return a structural failure signature for deterministic loop detection.

        Command output is deliberately excluded: paths, timings, worker IDs,
        and tool versions vary across otherwise equivalent validation runs.
        """

        parts = [self.state.value]
        for check in self.checks:
            if check.passed is not True:
                parts.append(f"{check.name}:{check.passed}:{check.returncode}")
        return "|".join(parts)

    def to_dict(self, *, include_patch: bool = True) -> dict[str, Any]:
        data = {
            "state": self.state.value,
            "checks": [check.to_dict() for check in self.checks],
            "changed_files": list(self.changed_files),
            "added_lines": self.added_lines,
            "deleted_lines": self.deleted_lines,
            "tested_commands": list(self.tested_commands),
            "reason": self.reason,
            "score": list(self.score),
            "fingerprint": self.fingerprint(),
        }
        if include_patch:
            data["patch"] = self.patch
        return data


@dataclass(frozen=True)
class TraceR3Config:
    """Frozen-validation and recovery policy."""

    validation_plan_min_cases: int = 10
    validation_plan_max_cases: int = 16
    validation_plan_min_test_commands: int = 2
    validation_plan_generation_attempts: int = 3
    max_epochs: int = 3
    max_checkpoints_per_epoch: int = 3
    max_non_improving_checkpoints: int = 2
    max_replayed_commands: int = 3
    validation_timeout_seconds: int = 900
    max_patch_files: int = 12
    max_patch_lines: int = 1200
    graph_depth: int = 1
    graph_max_files: int = 2500
    graph_max_chars: int = 14_000
    graph_max_seeds: int = 12
    validation_model: dict[str, Any] = field(
        default_factory=lambda: {
            "model_class": "minisweagent.trace_r3.deepseek.ReasoningReplayLitellmModel",
            "model_kwargs": {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "max",
            },
        }
    )
    recovery_model: dict[str, Any] = field(
        default_factory=lambda: {
            "model_class": "minisweagent.trace_r3.deepseek.ReasoningReplayLitellmModel",
            "model_kwargs": {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "max",
            }
        }
    )
    recovery_agent: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.validation_plan_min_cases < 10:
            raise ValueError("frozen validation requires at least ten cases")
        if self.validation_plan_max_cases < self.validation_plan_min_cases:
            raise ValueError("validation_plan_max_cases must not be smaller than its minimum")
        if self.validation_plan_min_test_commands < 2:
            raise ValueError("frozen validation requires at least two native test commands")
        if self.validation_plan_generation_attempts < 1:
            raise ValueError("validation_plan_generation_attempts must be positive")
        if self.max_epochs < 1 or self.max_epochs > 3:
            raise ValueError("TRACE-R³ supports between one and three epochs")
        if self.max_checkpoints_per_epoch < 1:
            raise ValueError("max_checkpoints_per_epoch must be positive")
        if self.graph_depth != 1:
            raise ValueError("TRACE-R³ deliberately supports one-hop retrieval only")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TraceR3Config:
        return cls(**(data or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
