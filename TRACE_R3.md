# TRACE-R³ for mini-SWE-agent

TRACE-R³ is a baseline-first recovery layer built on mini-SWE-agent 2.4.6. It
combines execution gates, one-hop static repository graphs, context inlining,
stronger reasoning on the same model, and hybrid checkpoint/rollback control.

The central experimental invariant is enforced by control flow:

```text
B0 clean task image
  └─ unchanged mini-SWE-agent baseline
       └─ Gate G0
            ├─ GREEN ──> accept baseline; recovery code is never reached
            └─ AMBER/RED
                  └─ activate Flash-max + one-hop graph RAG + context inlining
                       ├─ evidence improves ──> stack next candidate
                       └─ chain is unsafe/stuck ──> seal chain, recreate B0
```

There are at most three recovery epochs. Each epoch permits at most three
incremental checkpoints.

## What is implemented

### Gate G0 and subsequent gates

The gate uses only evidence available inside the task container. It does not
read SWE-bench gold patches or hidden evaluation results.

It captures the real `git diff`, checks:

- a non-empty patch;
- `git diff --check`;
- changed Python syntax;
- no modifications to test files;
- configurable patch scope limits;
- replay of the final safe reproduction and real test-runner commands found in
  the trajectory.

Replayed pipelines run under `bash -o pipefail`. Commands masked with `|| true`
or equivalent constructs are excluded. A zero exit status with no collected
tests or skipped-only evidence is not accepted as a pass. Importing or
installing a test framework is not a test command. Reproduction scripts are
retained as useful evidence, but cannot make a gate green without a genuine
test-runner invocation.

### LLM-consumable one-hop graph RAG

Graph construction is static and uses Python's AST; it makes no LLM call.
The worker runs inside `/testbed` from `/tmp`, so it does not pollute the patch.
Seeds come from changed function ranges plus identifiers in the issue/diff.

The one-hop search operation is adapted from the official Apache-2.0
[RepoGraph implementation](https://github.com/ozyyshr/RepoGraph) at pinned
commit `6c3977d87845993bf2c0359b4ac752278d7f3c45`. Its dependency-heavy
Tree-Sitter construction stack is replaced with a pure-standard-library worker
so heterogeneous SWE-bench images do not need to be modified.

Only direct functional edges are retrieved:

- upstream: functions that directly call the target;
- downstream: functions directly called by the target.

The exact text sent to the recovery LLM is also saved as `rag_context.md`. It is
deterministically flattened into this contract:

```xml
<repository_graph_rag depth="1" construction="repograph-one-hop-static-python-ast-no-llm">
  <target ...>
    Signature and failed-candidate anchor source
    <upstream_callers>
      Caller signature, location, call-site source, edge evidence
      Non-executable anchor-in-caller projection
    </upstream_callers>
    <downstream_callees>
      Callee signature, bounded source, edge evidence
    </downstream_callees>
  </target>
</repository_graph_rag>
```

The recovery prompt tells the LLM that edges and inline projections are
navigation evidence and must be verified in source. Retrieval is capped by
seed, file, and character budgets. Two-hop retrieval is rejected by
configuration, preventing context explosion. Unsupported or unresolvable
graphs degrade explicitly to direct repository inspection.

### Context Inlining adaptation

The first failed patch is treated as a draft anchor. For each direct caller,
TRACE-R³ independently implements the four transformations specified by
InlineCoder: parameter substitution, return normalization, assignment
redirection, and linearized inline expansion. The projection is never executed.
Recovery receives:

1. the original issue;
2. the complete failed candidate patch (bounded only in the prompt; full copy is
   retained on disk);
3. failed Gate evidence;
4. direct upstream caller contexts with the anchor inlined;
5. direct downstream callee contexts.

The LLM must reconstruct the behavioral invariant and test an alternative root
cause before editing. This adapts the upstream/downstream design of
[InlineCoder](https://arxiv.org/abs/2601.00376) to issue repair. Runtime Gate
evidence replaces InlineCoder's completion-perplexity confidence signal.

No InlineCoder source is copied: the audited public snapshot has no LICENSE and
omits modules required by its published pipeline. See
[`RESEARCH_PROVENANCE.md`](RESEARCH_PROVENANCE.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for pinned revisions and the
exact reuse boundary.

### Same Flash model, stronger recovery reasoning

The baseline uses the model configuration unchanged. Only after G0 fails does
the recovery overlay apply:

```yaml
thinking:
  type: enabled
reasoning_effort: max
```

The model name is required to remain unchanged. A recovery-only LiteLLM adapter
preserves `reasoning_content` and normalizes non-null assistant content across
tool-call turns, as required by DeepSeek V4. This adapter is never instantiated
on a baseline that passes G0.

### Hybrid version controller

Within an epoch, candidates may stack while Gate evidence improves. The entire
chain is sealed and the Docker task environment is recreated from B0 when any
of these conditions occurs:

- the same failure signature repeats;
- a previously passing critical check regresses;
- syntax, diff, test-pollution, or scope safety fails;
- two consecutive checkpoints do not improve;
- the per-epoch checkpoint limit is reached.

Failure signatures use gate state, failed/unknown check names, and return
codes. Volatile command output such as temporary paths, worker IDs, and timing
does not affect rollback decisions.

At the third epoch, the same condition stops recovery and selects the
highest-scoring observed candidate as a clearly marked best-effort result.

On rollback, the failed candidate's filesystem is discarded before the next
graph is built. Only its changed function bodies survive as read-only anchors;
callers, callees, and definitions are resolved again from the recreated B0
workspace.

## Installation

From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Keep the task-specific global configuration isolated:

```bash
export MSWEA_GLOBAL_CONFIG_DIR="$HOME/CHKtask/.mini-swe-agent-config"
export PATH="$HOME/bin:$PATH"
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
```

## Run the seven failed seed-3318 cases

The command below reruns a clean, unchanged baseline for each selected case so
that G0 remains a valid experimental boundary:

```bash
FILTER='^(sympy__sympy-22080|pytest-dev__pytest-10356|sympy__sympy-18763|sympy__sympy-17630|pytest-dev__pytest-5840|django__django-10973|sympy__sympy-16597)$'

mini-extra trace-r3 \
  --subset verified \
  --split test \
  --filter "$FILTER" \
  --model deepseek/deepseek-v4-flash \
  --environment-class docker \
  --workers 1 \
  --output "$HOME/CHKtask/experiments/v4flash_seed3318_trace_r3"
```

The command automatically merges the unchanged `swebench.yaml` baseline with
`trace_r3.yaml`. If `-c/--config` is supplied manually, include both:

```bash
-c swebench.yaml -c trace_r3.yaml
```

## Durable artifact layout

Every invocation allocates a new attempt instead of overwriting an interrupted
one:

```text
OUTPUT/
├── preds.json
└── INSTANCE_ID/
    ├── latest_attempt.json
    ├── attempt-001/
    │   ├── run_manifest.json
    │   ├── baseline/
    │   │   ├── baseline.traj.json
    │   │   ├── candidate.patch
    │   │   ├── gate.json
    │   │   └── gate.md
    │   ├── recovery/epoch-01/checkpoint-01/
    │   │   ├── recovery.traj.json
    │   │   ├── prompt.md
    │   │   ├── input_graph.json
    │   │   ├── rag_context.md
    │   │   ├── candidate.patch
    │   │   ├── gate.json
    │   │   └── version_decision.json
    │   ├── version_ledger.json
    │   └── final/
    │       ├── model.patch
    │       └── selection.json
    └── attempt-002/                # created after an interrupted/restarted run
```

Agent trajectories are written after every interaction through the agent's
`output_path`, not only when the run finishes. An abrupt shutdown therefore
leaves the most recent valid partial trajectory. On restart, tasks already in
`preds.json` are skipped; an incomplete task gets a new attempt directory.

Audit all finished attempts and retained partial trajectories with:

```bash
python scripts/audit_trace_r3_run.py \
  "$HOME/CHKtask/experiments/v4flash_seed3318_trace_r3"
```

## Local validation

```bash
python -m ruff check \
  src/minisweagent/trace_r3 \
  src/minisweagent/run/benchmarks/trace_r3.py \
  tests/trace_r3

MSWEA_SILENT_STARTUP=1 \
MSWEA_GLOBAL_CONFIG_DIR=.test-config \
python -m pytest -q tests/trace_r3
```

The offline suite verifies real Git patches and command replay, strict one-hop
retrieval, context budgets, epoch sealing, DeepSeek reasoning-history replay,
and both end-to-end activation paths.
