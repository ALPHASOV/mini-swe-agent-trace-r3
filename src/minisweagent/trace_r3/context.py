"""Recovery prompt construction inspired by TRACE and Context Inlining."""

from __future__ import annotations

import json
from collections.abc import Iterable

from minisweagent.trace_r3.types import GateReport
from minisweagent.trace_r3.validation import failed_validation_feedback

RECOVERY_SYSTEM_TEMPLATE = """You are the recovery stage of TRACE-R³.

The unchanged baseline mini-SWE-agent already attempted this task and failed an
execution-evidence gate. You still have a normal shell and must produce a
minimal, general source fix. Work from evidence, not from the previous patch's
assumptions.

Every response must contain concise reasoning and at least one bash tool call.
Do not modify tests or packaging/configuration files. Before submission, run a
focused reproduction and the most relevant native tests. Create patch.txt from
only intended source changes, inspect it, then submit with the exact command:

echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
"""

RECOVERY_INSTANCE_TEMPLATE = """{{task}}"""


def build_recovery_task(
    *,
    problem_statement: str,
    failed_patch: str,
    gate_report: GateReport,
    graph_artifact: dict,
    epoch: int,
    checkpoint: int,
    clean_base: bool,
    sealed_reasons: Iterable[str],
) -> str:
    """Create the user message consumed by the stronger recovery call."""

    state_text = (
        "This is a clean B0 workspace. The failed patch below is evidence only and is NOT applied."
        if clean_base
        else "The current workspace retains the preceding candidate. You may refine it incrementally, "
        "but inspect `git diff` first and replace wrong logic instead of blindly appending."
    )
    gate_lines = [
        f"- {check.name}: {_status(check.passed)} — {check.detail}"
        for check in gate_report.checks
        if (
            (check.passed is not True or check.name.startswith("replay_"))
            and not check.name.startswith("frozen_validation.")
        )
    ]
    validation_feedback = failed_validation_feedback(gate_report.checks)
    sealed = "\n".join(f"- {reason}" for reason in sealed_reasons) or "- None"
    patch = _bounded(failed_patch, 18_000)
    graph_context = graph_artifact.get("rag_context", "")
    return f"""<trace_r3_recovery epoch="{epoch}" checkpoint="{checkpoint}" reasoning_effort="max">
<workspace_state>
{state_text}
</workspace_state>

<original_issue>
{problem_statement}
</original_issue>

<failed_draft_patch>
```diff
{patch}
```
</failed_draft_patch>

<execution_gate state="{gate_report.state.value}">
Reason: {gate_report.reason}
{chr(10).join(gate_lines)}
</execution_gate>

{validation_feedback}

<sealed_failure_patterns>
{sealed}
</sealed_failure_patterns>

<context_inlining_protocol>
Use the failed patch only as a draft anchor:
1. Restate the behavioral invariant and an alternative root-cause hypothesis.
2. Inspect the changed target plus the supplied direct upstream callers and
   direct downstream callees, including inherited-method consumers when
   present. Verify every graph edge in source before relying on it; static
   resolution is navigation evidence, not semantic proof.
3. Check what callers pass, how return values/exceptions are consumed, and what
   callees promise. Preserve those contracts. A failure in a neighboring case
   is evidence that the invariant is incomplete; do not dismiss it merely
   because the issue did not name that case.
4. Implement the smallest distinct source candidate consistent with that
   evidence. A recovery checkpoint must not resubmit a byte-identical patch.
5. For shared/base/dispatch code, inspect subclass or cross-module consumers
   and test at least one relevant consumer surface in addition to the closest
   unit tests.
6. Run an explicit issue reproduction and focused native tests. Test commands
   must emit a positive nonzero test-count summary. Do not use `||` fallbacks,
   hide failures behind pipes, truncate away the summary, or accept skipped-only
   tests. If pytest is unavailable, invoke the repository's native test runner
   directly.
</context_inlining_protocol>

{graph_context}

<version_policy>
TRACE-R³ may stack this patch within the current epoch only while gate evidence
improves. Repeated failure, regression, scope pollution, or non-improvement
seals the whole chain and restarts from B0. There are at most three epochs.
</version_policy>
</trace_r3_recovery>
"""


def graph_summary_for_manifest(graph_artifact: dict) -> dict:
    """Small graph summary suitable for the top-level run manifest."""

    return {
        key: graph_artifact.get(key)
        for key in (
            "schema_version",
            "construction",
            "depth",
            "stats",
            "seed_ids",
            "error",
            "llm_consumption",
        )
        if key in graph_artifact
    }


def _status(passed: bool | None) -> str:
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return "UNKNOWN"


def _bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    half = limit // 2
    marker = "\n... <failed patch truncated for prompt; full patch is saved as an artifact> ...\n"
    return f"{value[:half]}{marker}{value[-half:]}"


def render_gate_markdown(report: GateReport) -> str:
    """Human-readable counterpart of gate.json."""

    rows = ["| Check | Result | Critical | Detail |", "|---|---:|---:|---|"]
    for check in report.checks:
        detail = check.detail.replace("|", "\\|").replace("\n", " ")
        rows.append(f"| `{check.name}` | {_status(check.passed)} | {check.critical} | {detail} |")
    metadata = json.dumps(
        {
            "state": report.state.value,
            "changed_files": report.changed_files,
            "added_lines": report.added_lines,
            "deleted_lines": report.deleted_lines,
        },
        indent=2,
    )
    return f"# TRACE-R³ Gate\n\nReason: {report.reason}\n\n{chr(10).join(rows)}\n\n```json\n{metadata}\n```\n"
