"""Transfer and run the static graph retriever in an isolated environment."""

from __future__ import annotations

import base64
import json
import shlex
from pathlib import Path
from typing import Any

from minisweagent import Environment
from minisweagent.trace_r3.types import TraceR3Config


class OneHopGraphRetriever:
    """RepoGraph-inspired one-hop retrieval with deterministic context flattening."""

    def __init__(self, config: TraceR3Config):
        self.config = config

    def retrieve(
        self,
        env: Environment,
        *,
        seed_terms: list[str],
        anchor_sources: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        worker_path = Path(__file__).with_name("graph_worker.py")
        encoded_worker = base64.b64encode(worker_path.read_bytes()).decode()
        install_command = (
            "python -c "
            + shlex.quote(
                "import base64;"
                f"open('/tmp/trace_r3_graph_worker.py','wb').write(base64.b64decode('{encoded_worker}'))"
            )
        )
        installed = env.execute({"command": install_command}, timeout=120)
        if installed["returncode"] != 0:
            return _error_artifact("worker_transfer_failed", installed)

        encoded_terms = base64.b64encode(json.dumps(seed_terms).encode()).decode()
        encoded_anchors = base64.b64encode(json.dumps(anchor_sources or {}).encode()).decode()
        command = " ".join(
            [
                "python /tmp/trace_r3_graph_worker.py",
                "--root .",
                f"--seed-terms-base64 {shlex.quote(encoded_terms)}",
                f"--anchor-sources-base64 {shlex.quote(encoded_anchors)}",
                f"--max-files {self.config.graph_max_files}",
                f"--max-seeds {self.config.graph_max_seeds}",
                f"--max-chars {self.config.graph_max_chars}",
            ]
        )
        result = env.execute({"command": command}, timeout=self.config.validation_timeout_seconds)
        if result["returncode"] != 0:
            return _error_artifact("graph_worker_failed", result)
        try:
            artifact = json.loads(result["output"])
        except json.JSONDecodeError:
            return _error_artifact("invalid_graph_json", result)
        artifact["llm_consumption"] = {
            "format": "repograph_one_hop_with_inlinecoder_projection",
            "summarizer_model": None,
            "hop_limit": 1,
            "placement": "recovery user message after the failed-patch and gate evidence",
        }
        return artifact


def extract_seed_terms(problem_statement: str, patch: str, *, limit: int = 80) -> list[str]:
    """Extract identifier candidates without using an LLM."""

    import re

    terms: list[str] = []
    seen: set[str] = set()
    def_pattern = re.compile(r"^[+-]\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
    code_pattern = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)`|\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(")
    for match in def_pattern.finditer(patch):
        terms.append(match.group(1))
    for match in code_pattern.finditer(problem_statement):
        terms.append(match.group(1) or match.group(2))
    result: list[str] = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            result.append(term)
        if len(result) >= limit:
            break
    return result


def extract_anchor_sources(graph_artifact: dict[str, Any]) -> dict[str, str]:
    """Retain only failed-candidate seed bodies for a clean B0 graph rebuild."""

    if graph_artifact.get("anchor_sources"):
        return {
            name: source
            for name, source in graph_artifact["anchor_sources"].items()
            if name and source
        }
    seed_ids = set(graph_artifact.get("seed_ids", []))
    return {
        node["qualified_name"]: node["source"]
        for node in graph_artifact.get("retrieved_nodes", [])
        if node.get("id") in seed_ids and node.get("qualified_name") and node.get("source")
    }


def _error_artifact(kind: str, result: dict[str, Any]) -> dict[str, Any]:
    output = str(result.get("output", ""))
    return {
        "schema_version": "trace-r3-repograph-rag/1",
        "construction": "repograph_one_hop_static_python_ast_no_llm",
        "depth": 1,
        "error": kind,
        "returncode": result.get("returncode"),
        "output": output[-4000:],
        "retrieved_nodes": [],
        "retrieved_edges": [],
        "rag_context": (
            '<repository_graph_rag schema="trace-r3/1" depth="1">'
            f"<unavailable reason={json.dumps(kind)}>Continue using direct repository inspection.</unavailable>"
            "</repository_graph_rag>"
        ),
    }
