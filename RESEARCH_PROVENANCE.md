# Research provenance and implementation boundary

TRACE-R³ combines three recovery-only mechanisms while preserving the original
mini-SWE-agent baseline before Gate G0.

## RepoGraph

Paper and code:

- Siru Ouyang et al., "RepoGraph: Enhancing AI Software Engineering with
  Repository-level Code Graph", ICLR 2025.
- <https://arxiv.org/abs/2410.14684>
- <https://github.com/ozyyshr/RepoGraph>
- Audited code revision:
  `6c3977d87845993bf2c0359b4ac752278d7f3c45`.

Directly adapted code is limited to the Apache-2.0 one-hop
`RepoSearcher` operation. The original construction stack depends on
Tree-Sitter, grep-ast, NetworkX, Pygments, and cached graph files. TRACE-R³
reimplements graph construction with Python's standard-library AST so the
worker runs inside heterogeneous SWE-bench containers without installing or
mutating their environments.

The resulting graph retains RepoGraph's deterministic, no-LLM construction and
ego-graph retrieval principles. It is deliberately restricted to the direct
functional projection around each seed: immediate callers and immediate
callees only.

## InlineCoder

Paper and public code snapshot:

- Chao Hu et al., "In Line with Context: Repository-Level Code Generation via
  Context Inlining", FSE 2026.
- <https://arxiv.org/abs/2601.00376>
- <https://github.com/ythere-y/InlineCoder>
- Audited code revision:
  `d9c9fedb12e5a5207fbfdad65c1ea8fddbc2c2bf`.

The audited repository has no LICENSE file. Its runnable scripts also import
`Parser`, `Preprocess`, and `ResultProcess` modules that are absent from that
revision. Consequently, TRACE-R³ copies no InlineCoder source.

Instead, `graph_worker.py` independently renders the four transformations
specified in Section 3.3.1 of the paper:

1. positional and keyword parameter substitution;
2. return normalization into `__trace_r3_result`;
3. redirection to the caller's assignment or return;
4. a linearized anchor-in-caller projection.

The projection is marked non-executable. It is supplied only as bounded RAG
context, while the actual candidate patch remains the source of truth.
Downstream callees are resolved from the failed candidate anchor, including
after a rollback, against definitions and callers from the newly recreated B0
workspace.

InlineCoder evaluates function completion and uses perplexity as a confidence
signal. TRACE-R³ targets issue repair instead, so it replaces perplexity with
the execution-evidence Gate report and uses the complete failed patch as the
anchor.

## Hybrid version controller

The version controller is original TRACE-R³ code. It permits incremental
candidate stacking only while gate evidence improves. Repeated failure,
critical regression, unsafe scope, or non-improvement seals the current chain,
recreates B0, and starts a sibling candidate epoch. A failed candidate's
function body may survive only as read-only retrieval evidence; its filesystem
state never crosses the rollback boundary.

All claims in this repository describe implementation structure, not benchmark
effectiveness. Resolve-rate improvement must be established by running the
documented baseline and TRACE-R³ experiments on the same selected instances.
