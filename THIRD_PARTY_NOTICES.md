# Third-party notices

## RepoGraph

TRACE-R³ adapts the one-hop `RepoSearcher` operation from:

- Project: [ozyyshr/RepoGraph](https://github.com/ozyyshr/RepoGraph)
- Source file: `repograph/graph_searcher.py`
- Pinned commit: `6c3977d87845993bf2c0359b4ac752278d7f3c45`
- License: Apache License 2.0

The adaptation in `src/minisweagent/trace_r3/graph_worker.py` replaces
RepoGraph's NetworkX object with a deterministic standard-library adjacency
graph, removes unused multi-hop traversal, and projects the result into direct
caller/callee contexts suitable for execution inside SWE-bench task images.

The Apache License 2.0 text is included at
`third_party/licenses/RepoGraph-APACHE-2.0.txt`.

## InlineCoder

No InlineCoder source code is redistributed. TRACE-R³ independently implements
the paper's four context-inlining transformations because the public
`ythere-y/InlineCoder` snapshot at
`d9c9fedb12e5a5207fbfdad65c1ea8fddbc2c2bf` has no repository license and does
not include several modules imported by its published pipeline. Method
provenance and the implementation boundary are documented in
`RESEARCH_PROVENANCE.md`.
