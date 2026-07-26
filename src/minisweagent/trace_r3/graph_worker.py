"""Pure-stdlib, in-repository worker for one-hop Python call-graph retrieval.

The controller transfers this module into the isolated task environment and
executes it there. It intentionally has no mini-SWE-agent dependency.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import json
import subprocess
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_IGNORED_PARTS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "venv",
}


@dataclass(frozen=True)
class Definition:
    id: str
    qualified_name: str
    name: str
    kind: str
    path: str
    line: int
    end_line: int
    signature: str
    source: str
    module: str
    class_name: str
    parameters: tuple[str, ...]
    import_bindings: tuple[tuple[str, str], ...]
    class_qualified_name: str = ""
    base_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class CallSite:
    caller: str
    expression: str
    terminal_name: str
    path: str
    line: int
    evidence: str
    arguments: tuple[str, ...]
    keywords: tuple[tuple[str, str], ...]
    assignment_target: str
    qualified_hint: str


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str
    path: str
    line: int
    evidence: str
    arguments: tuple[str, ...]
    keywords: tuple[tuple[str, str], ...]
    assignment_target: str
    resolution: str


class _AdjacencyGraph:
    """Small graph surface used by RepoGraph's one-hop search contract."""

    def __init__(self, edges: list[Edge]):
        self.adjacency: dict[str, set[str]] = {}
        for edge in edges:
            self.adjacency.setdefault(edge.source, set()).add(edge.target)
            self.adjacency.setdefault(edge.target, set()).add(edge.source)

    def neighbors(self, query: str) -> list[str]:
        return sorted(self.adjacency.get(query, ()))


# Adapted from ozyyshr/RepoGraph, repograph/graph_searcher.py, commit
# 6c3977d87845993bf2c0359b4ac752278d7f3c45 (Apache-2.0). Modified to use a
# deterministic stdlib adjacency graph and expose only the one-hop operation.
class RepoSearcher:
    """One-hop search surface compatible with RepoGraph's RepoSearcher."""

    def __init__(self, graph: _AdjacencyGraph):
        self.graph = graph

    def one_hop_neighbors(self, query: str) -> list[str]:
        return list(self.graph.neighbors(query))


class DefinitionVisitor(ast.NodeVisitor):
    def __init__(
        self,
        path: str,
        module: str,
        source: str,
        *,
        import_bindings: dict[str, str] | None = None,
        parents: dict[int, ast.AST] | None = None,
    ):
        self.path = path
        self.module = module
        self.source = source
        self.lines = source.splitlines()
        self.import_bindings = import_bindings or {}
        self.parents = parents or {}
        self.definitions: list[Definition] = []
        self.calls: list[CallSite] = []
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.definition_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        parents = [*self.class_stack, *self.function_stack]
        qualified_name = ".".join([self.module, *parents, node.name])
        class_id = f"{self.path}:{node.lineno}:{qualified_name}"
        base_hints = tuple(
            hint
            for base in node.bases
            if (
                hint := _qualified_hint(
                    _dotted_name(base),
                    module=self.module,
                    class_name=self.class_stack[-1] if self.class_stack else "",
                    import_bindings=self.import_bindings,
                )
            )
        )
        self.definitions.append(
            Definition(
                id=class_id,
                qualified_name=qualified_name,
                name=node.name,
                kind="class",
                path=self.path,
                line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                signature=f"class {node.name}({', '.join(_unparse(base) for base in node.bases)})",
                source=_source_slice(
                    self.lines,
                    node.lineno,
                    getattr(node, "end_lineno", node.lineno),
                    max_lines=60,
                ),
                module=self.module,
                class_name=node.name,
                parameters=(),
                import_bindings=tuple(sorted(self.import_bindings.items())),
                class_qualified_name=qualified_name,
                base_hints=base_hints,
            )
        )
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function(node, "async_function")

    def visit_Call(self, node: ast.Call) -> Any:
        if self.definition_stack:
            expression = _dotted_name(node.func)
            terminal = expression.rsplit(".", 1)[-1] if expression else ""
            if terminal:
                arguments = tuple(_unparse(argument) for argument in node.args)
                keywords = tuple((keyword.arg or "**", _unparse(keyword.value)) for keyword in node.keywords)
                self.calls.append(
                    CallSite(
                        caller=self.definition_stack[-1],
                        expression=expression,
                        terminal_name=terminal,
                        path=self.path,
                        line=node.lineno,
                        evidence=_line(self.lines, node.lineno),
                        arguments=arguments,
                        keywords=keywords,
                        assignment_target=_assignment_target(node, self.parents),
                        qualified_hint=_qualified_hint(
                            expression,
                            module=self.module,
                            class_name=self.class_stack[-1] if self.class_stack else "",
                            import_bindings=self.import_bindings,
                        ),
                    )
                )
        self.generic_visit(node)

    def _visit_function(self, node: ast.AST, kind: str) -> None:
        parents = [*self.class_stack, *self.function_stack]
        qualified_name = ".".join([self.module, *parents, node.name])
        definition_id = f"{self.path}:{node.lineno}:{qualified_name}"
        end_line = getattr(node, "end_lineno", node.lineno)
        class_name = self.class_stack[-1] if self.class_stack else ""
        class_qualified_name = ".".join([self.module, *self.class_stack]) if self.class_stack else ""
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        self.definitions.append(
            Definition(
                id=definition_id,
                qualified_name=qualified_name,
                name=node.name,
                kind="method" if class_name else kind,
                path=self.path,
                line=node.lineno,
                end_line=end_line,
                signature=f"{prefix} {node.name}({_format_arguments(node.args)})",
                source=_source_slice(
                    self.lines,
                    node.lineno,
                    end_line,
                    max_lines=max(1, end_line - node.lineno + 1),
                ),
                module=self.module,
                class_name=class_name,
                parameters=tuple(
                    argument.arg
                    for argument in [
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    ]
                ),
                import_bindings=tuple(sorted(self.import_bindings.items())),
                class_qualified_name=class_qualified_name,
            )
        )
        self.function_stack.append(node.name)
        self.definition_stack.append(definition_id)
        self.generic_visit(node)
        self.definition_stack.pop()
        self.function_stack.pop()


def build_graph(root: Path, *, max_files: int) -> tuple[list[Definition], list[Edge], dict[str, int]]:
    definitions: list[Definition] = []
    calls: list[CallSite] = []
    parsed_files = 0
    syntax_errors = 0
    for path in _python_files(root, max_files=max_files):
        relative = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=relative)
        except (OSError, SyntaxError):
            syntax_errors += 1
            continue
        visitor = DefinitionVisitor(
            relative,
            _module_name(relative),
            source,
            import_bindings=_import_bindings(tree, _module_name(relative)),
            parents=_parent_map(tree),
        )
        visitor.visit(tree)
        definitions.extend(visitor.definitions)
        calls.extend(visitor.calls)
        parsed_files += 1
    call_edges = _resolve_calls(definitions, calls)
    inheritance_edges = _resolve_inheritance(definitions)
    edges = [*call_edges, *inheritance_edges]
    return definitions, edges, {
        "parsed_files": parsed_files,
        "syntax_errors": syntax_errors,
        "definitions": len(definitions),
        "calls": len(calls),
        "resolved_edges": len(edges),
        "resolved_call_edges": len(call_edges),
        "inheritance_edges": len(inheritance_edges),
        "unresolved_calls": max(0, len(calls) - len(call_edges)),
    }


def retrieve(
    root: Path,
    definitions: list[Definition],
    edges: list[Edge],
    *,
    seed_terms: list[str],
    max_seeds: int,
    max_chars: int,
    anchor_sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    anchor_sources = anchor_sources or {}
    changed_ranges = _changed_ranges(root)
    by_id = {definition.id: definition for definition in definitions}
    seeds = _select_seeds(
        definitions,
        changed_ranges,
        [*anchor_sources, *seed_terms],
        max_seeds=max_seeds,
    )
    seed_ids = {seed.id for seed in seeds}
    effective_edges = [
        edge
        for edge in edges
        if not (edge.source in seed_ids and edge.relation == "calls")
    ]
    effective_edges.extend(
        edge
        for seed in seeds
        for edge in _anchor_edges(
            seed,
            anchor_sources.get(seed.qualified_name, seed.source),
            definitions,
        )
    )
    effective_edges.extend(_inherited_method_edges(seeds, definitions, effective_edges))
    searcher = RepoSearcher(_AdjacencyGraph(effective_edges))
    selected_ids = seed_ids | {
        neighbor
        for seed_id in seed_ids
        for neighbor in searcher.one_hop_neighbors(seed_id)
    }
    selected_edges = [
        edge
        for edge in effective_edges
        if edge.source in selected_ids
        and edge.target in selected_ids
        and (edge.source in seed_ids or edge.target in seed_ids)
    ]
    selected_nodes = [by_id[node_id] for node_id in sorted(selected_ids) if node_id in by_id]
    retained_anchors = {
        seed.qualified_name: anchor_sources.get(seed.qualified_name, seed.source)
        for seed in seeds
        if seed.qualified_name in anchor_sources or _definition_is_changed(seed, changed_ranges)
    }
    context = _flatten_context(
        root,
        seeds,
        selected_edges,
        by_id,
        anchor_sources=anchor_sources,
        max_chars=max_chars,
    )
    return {
        "schema_version": "trace-r3-repograph-rag/1",
        "construction": "repograph_one_hop_static_python_ast_no_llm",
        "depth": 1,
        "seed_terms": seed_terms,
        "changed_ranges": {path: ranges for path, ranges in changed_ranges.items()},
        "seed_ids": [seed.id for seed in seeds],
        "anchor_provenance": "external_failed_candidate" if anchor_sources else "workspace_candidate",
        "anchor_sources": retained_anchors,
        "retrieved_nodes": [_node_artifact(node) for node in selected_nodes],
        "retrieved_edges": [asdict(edge) for edge in selected_edges],
        "rag_context": context,
    }


def _python_files(root: Path, *, max_files: int) -> list[Path]:
    paths = [
        path
        for path in root.rglob("*.py")
        if not any(part in _IGNORED_PARTS for part in path.relative_to(root).parts)
    ]
    return sorted(paths)[:max_files]


def _node_artifact(definition: Definition) -> dict[str, Any]:
    data = asdict(definition)
    lines = definition.source.splitlines()
    data["source"] = _source_slice(lines, 1, len(lines), max_lines=60)
    return data


def _definition_is_changed(
    definition: Definition,
    changed_ranges: dict[str, list[tuple[int, int]]],
) -> bool:
    return any(
        start <= definition.end_line and definition.line <= end
        for start, end in changed_ranges.get(definition.path, ())
    )


def _resolve_calls(definitions: list[Definition], calls: list[CallSite]) -> list[Edge]:
    by_name: dict[str, list[Definition]] = {}
    by_qualified = {definition.qualified_name: definition for definition in definitions}
    by_id = {definition.id: definition for definition in definitions}
    for definition in definitions:
        by_name.setdefault(definition.name, []).append(definition)
    edges: list[Edge] = []
    seen: set[tuple[str, str, int]] = set()
    for call in calls:
        caller = by_id[call.caller]
        candidates = by_name.get(call.terminal_name, [])
        hinted = by_qualified.get(call.qualified_hint)
        local_class = [
            item
            for item in candidates
            if caller.class_name and item.module == caller.module and item.class_name == caller.class_name
        ]
        local_module = [item for item in candidates if item.module == caller.module]
        if hinted is not None:
            resolved = hinted
            resolution = "qualified_import_or_scope"
        elif call.expression.startswith(("self.", "cls.")) and len(local_class) == 1:
            resolved = local_class[0]
            resolution = "same_class"
        elif len(local_module) == 1:
            resolved = local_module[0]
            resolution = "same_module"
        elif len(candidates) == 1:
            resolved = candidates[0]
            resolution = "unique_repository_name"
        else:
            continue
        key = (call.caller, resolved.id, call.line)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            Edge(
                source=call.caller,
                target=resolved.id,
                relation="calls",
                path=call.path,
                line=call.line,
                evidence=f"{call.expression} @ {call.evidence}",
                arguments=call.arguments,
                keywords=call.keywords,
                assignment_target=call.assignment_target,
                resolution=resolution,
            )
        )
    return edges


def _resolve_inheritance(definitions: list[Definition]) -> list[Edge]:
    classes = [definition for definition in definitions if definition.kind == "class"]
    by_qualified = {definition.qualified_name: definition for definition in classes}
    by_name: dict[str, list[Definition]] = {}
    for definition in classes:
        by_name.setdefault(definition.name, []).append(definition)

    edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()
    for child in classes:
        for hint in child.base_hints:
            parent = by_qualified.get(hint)
            if parent is None:
                candidates = by_name.get(hint.rsplit(".", 1)[-1], [])
                if len(candidates) == 1:
                    parent = candidates[0]
            if parent is None or parent.id == child.id or (child.id, parent.id) in seen:
                continue
            seen.add((child.id, parent.id))
            edges.append(
                Edge(
                    source=child.id,
                    target=parent.id,
                    relation="inherits",
                    path=child.path,
                    line=child.line,
                    evidence=f"{child.signature} inherits {parent.qualified_name}",
                    arguments=(),
                    keywords=(),
                    assignment_target="",
                    resolution="static_class_hierarchy",
                )
            )
    return edges


def _inherited_method_edges(
    seeds: list[Definition],
    definitions: list[Definition],
    edges: list[Edge],
    *,
    max_consumers_per_method: int = 24,
) -> list[Edge]:
    """Connect a changed method directly to classes that inherit its implementation."""

    classes = {
        definition.qualified_name: definition
        for definition in definitions
        if definition.kind == "class"
    }
    class_by_id = {definition.id: definition for definition in classes.values()}
    methods = {
        (definition.class_qualified_name, definition.name)
        for definition in definitions
        if definition.kind == "method" and definition.class_qualified_name
    }
    children: dict[str, list[str]] = {}
    for edge in edges:
        if edge.relation == "inherits" and edge.source in class_by_id and edge.target in class_by_id:
            children.setdefault(edge.target, []).append(edge.source)

    result: list[Edge] = []
    for seed in seeds:
        if seed.kind != "method" or not seed.class_qualified_name:
            continue
        owner = classes.get(seed.class_qualified_name)
        if owner is None:
            continue
        queue = sorted(children.get(owner.id, ()))
        visited: set[str] = set()
        while queue and len(visited) < max_consumers_per_method:
            child_id = queue.pop(0)
            if child_id in visited:
                continue
            visited.add(child_id)
            child = class_by_id[child_id]
            if (child.qualified_name, seed.name) in methods:
                continue
            result.append(
                Edge(
                    source=child.id,
                    target=seed.id,
                    relation="inherits_method",
                    path=child.path,
                    line=child.line,
                    evidence=f"{child.qualified_name} inherits {seed.qualified_name}",
                    arguments=(),
                    keywords=(),
                    assignment_target="",
                    resolution="transitive_static_class_hierarchy",
                )
            )
            queue.extend(sorted(children.get(child_id, ())))
    return result


def _anchor_edges(seed: Definition, source: str, definitions: list[Definition]) -> list[Edge]:
    """Resolve calls from the candidate anchor against definitions in the current clean repository."""

    try:
        normalized = textwrap.dedent(source)
        tree = ast.parse(normalized)
    except SyntaxError:
        return []
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == seed.name
        ),
        None,
    )
    if function is None:
        return []
    parents = _parent_map(function)
    lines = normalized.splitlines()
    calls = []
    for node in _calls_without_nested_functions(function):
        expression = _dotted_name(node.func)
        terminal = expression.rsplit(".", 1)[-1] if expression else ""
        if not terminal:
            continue
        calls.append(
            CallSite(
                caller=seed.id,
                expression=expression,
                terminal_name=terminal,
                path=seed.path,
                line=seed.line + node.lineno - 1,
                evidence=_line(lines, node.lineno),
                arguments=tuple(_unparse(argument) for argument in node.args),
                keywords=tuple((keyword.arg or "**", _unparse(keyword.value)) for keyword in node.keywords),
                assignment_target=_assignment_target(node, parents),
                qualified_hint=_qualified_hint(
                    expression,
                    module=seed.module,
                    class_name=seed.class_name,
                    import_bindings=dict(seed.import_bindings),
                ),
            )
        )
    return _resolve_calls(definitions, calls)


def _calls_without_nested_functions(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Call]:
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.calls: list[ast.Call] = []

        def visit_Call(self, node: ast.Call) -> Any:
            self.calls.append(node)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return None

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return None

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return None

    visitor = Visitor()
    for statement in function.body:
        visitor.visit(statement)
    return visitor.calls


def _import_bindings(tree: ast.AST, module: str) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".", 1)[0]
                bindings[local] = alias.name if alias.asname else local
        elif isinstance(node, ast.ImportFrom):
            imported_module = node.module or ""
            if node.level:
                package = module.rpartition(".")[0].split(".") if "." in module else []
                prefix = package[: max(0, len(package) - node.level + 1)]
                imported_module = ".".join([*prefix, *([imported_module] if imported_module else [])])
            for alias in node.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = ".".join(
                        part for part in (imported_module, alias.name) if part
                    )
    return bindings


def _parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    return {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _assignment_target(node: ast.Call, parents: dict[int, ast.AST]) -> str:
    current: ast.AST = node
    while id(current) in parents:
        current = parents[id(current)]
        if isinstance(current, ast.Assign):
            return _unparse(current.targets[0])
        if isinstance(current, ast.AnnAssign):
            return _unparse(current.target)
        if isinstance(current, ast.Return):
            return "return"
        if isinstance(current, (ast.Expr, ast.stmt)):
            break
    return ""


def _qualified_hint(
    expression: str,
    *,
    module: str,
    class_name: str,
    import_bindings: dict[str, str],
) -> str:
    if not expression:
        return ""
    parts = expression.split(".")
    if parts[0] in {"self", "cls"} and class_name:
        return ".".join([module, class_name, *parts[1:]])
    if parts[0] in import_bindings:
        return ".".join([import_bindings[parts[0]], *parts[1:]])
    if len(parts) == 1:
        return f"{module}.{expression}"
    return expression


def _inline_projection(target: Definition, anchor_source: str, edge: Edge) -> str:
    """Render InlineCoder's four transformations as a non-executable caller projection."""

    try:
        tree = ast.parse(textwrap.dedent(anchor_source))
    except SyntaxError:
        return "# Candidate anchor could not be parsed; inspect the saved failed patch directly."
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target.name
        ),
        None,
    )
    if function is None:
        return "# Candidate anchor function was not found; inspect the saved failed patch directly."

    parameters = [
        argument.arg
        for argument in [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ]
    ]
    if target.class_name and parameters and parameters[0] in {"self", "cls"}:
        parameters.pop(0)
    bindings: dict[str, ast.AST] = {}
    binding_text = []
    positional_arguments = list(edge.arguments)
    if target.class_name and len(positional_arguments) == len(parameters) + 1:
        positional_arguments.pop(0)
    for parameter, argument in zip(parameters, positional_arguments):
        parsed = _parse_expression(argument)
        if parsed is not None:
            bindings[parameter] = parsed
            binding_text.append(f"{parameter} <- {argument}")
    for parameter, argument in edge.keywords:
        if parameter in parameters:
            parsed = _parse_expression(argument)
            if parsed is not None:
                bindings[parameter] = parsed
                binding_text.append(f"{parameter} <- {argument}")

    class Transformer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load) and node.id in bindings:
                return ast.copy_location(copy.deepcopy(bindings[node.id]), node)
            return node

        def visit_Return(self, node: ast.Return) -> ast.AST:
            return ast.copy_location(
                ast.Assign(
                    targets=[ast.Name(id="__trace_r3_result", ctx=ast.Store())],
                    value=self.visit(node.value) if node.value is not None else ast.Constant(value=None),
                ),
                node,
            )

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            return node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            return node

    transformer = Transformer()
    transformed = [ast.fix_missing_locations(transformer.visit(copy.deepcopy(statement))) for statement in function.body]
    body = "\n".join(_unparse(statement) for statement in transformed)
    redirected = (
        "return __trace_r3_result"
        if edge.assignment_target == "return"
        else f"{edge.assignment_target} = __trace_r3_result"
        if edge.assignment_target
        else "# call result is not assigned"
    )
    bindings_line = ", ".join(binding_text) if binding_text else "(no positional/keyword bindings resolved)"
    return "\n".join(
        [
            f"# Original call: {edge.evidence}",
            f"# Parameter substitution: {bindings_line}",
            "# Return normalization uses __trace_r3_result.",
            body,
            f"# Assignment redirection: {redirected}",
        ]
    )


def _parse_expression(value: str) -> ast.AST | None:
    try:
        return ast.parse(value, mode="eval").body
    except SyntaxError:
        return None


def _changed_ranges(root: Path) -> dict[str, list[tuple[int, int]]]:
    try:
        result = subprocess.run(
            ["git", "diff", "--unified=0", "--", ".", ":(exclude)patch.txt"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    current = ""
    ranges: dict[str, list[tuple[int, int]]] = {}
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif current and line.startswith("@@"):
            match = __import__("re").search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                start = int(match.group(1))
                count = max(1, int(match.group(2) or "1"))
                ranges.setdefault(current, []).append((start, start + count - 1))
    return ranges


def _select_seeds(
    definitions: list[Definition],
    changed_ranges: dict[str, list[tuple[int, int]]],
    seed_terms: list[str],
    *,
    max_seeds: int,
) -> list[Definition]:
    normalized_terms = [term.strip() for term in seed_terms if term.strip()]
    changed_functions: list[Definition] = []
    changed_classes: list[Definition] = []
    for definition in definitions:
        ranges = changed_ranges.get(definition.path, [])
        if any(definition.line <= end and definition.end_line >= start for start, end in ranges):
            if definition.kind == "class":
                changed_classes.append(definition)
            else:
                changed_functions.append(definition)

    changed = changed_functions or changed_classes
    ordered = list(sorted(changed, key=_definition_key))
    changed_paths = {definition.path for definition in changed}
    for term in normalized_terms:
        exact = [definition for definition in definitions if definition.qualified_name == term]
        matches = exact or [definition for definition in definitions if definition.name == term]
        if not exact and len(matches) > 3:
            continue
        if matches:
            ordered.append(min(matches, key=lambda item: _seed_rank(item, changed_paths)))

    if not ordered and changed_ranges:
        changed_paths = set(changed_ranges)
        ordered = sorted(
            [
                definition
                for definition in definitions
                if definition.path in changed_paths and definition.kind != "class"
            ],
            key=_definition_key,
        )
    unique: list[Definition] = []
    seen: set[str] = set()
    for definition in ordered:
        if definition.id not in seen:
            unique.append(definition)
            seen.add(definition.id)
        if len(unique) >= max_seeds:
            break
    return unique


def _seed_rank(definition: Definition, changed_paths: set[str]) -> tuple[int, int, int, str, int]:
    changed_directories = {
        path.rsplit("/", 1)[0] if "/" in path else ""
        for path in changed_paths
    }
    directory = definition.path.rsplit("/", 1)[0] if "/" in definition.path else ""
    related = any(
        directory == changed or directory.startswith(f"{changed}/") or changed.startswith(f"{directory}/")
        for changed in changed_directories
        if changed or directory
    )
    is_test = "/test" in definition.path or definition.path.startswith("test")
    return (
        0 if definition.path in changed_paths else 1,
        0 if related else 1,
        1 if is_test else 0,
        definition.path,
        definition.line,
    )


def _flatten_context(
    root: Path,
    seeds: list[Definition],
    edges: list[Edge],
    by_id: dict[str, Definition],
    *,
    anchor_sources: dict[str, str],
    max_chars: int,
) -> str:
    chunks = [
        (
            '<repository_graph_rag schema="trace-r3/1" depth="1" '
            'construction="repograph-one-hop-static-python-ast-no-llm">'
        ),
        "Retrieval contract: only direct callers and direct callees are included, "
        "plus statically derived inherited-method consumers in one retrieval hop. "
        "Treat edge evidence as navigation evidence, not proof of the bug.",
        "Inline projections are analysis-only views of the failed candidate anchor and are never executed.",
    ]
    for seed in seeds:
        anchor_source = anchor_sources.get(seed.qualified_name, seed.source)
        provenance = "external-failed-candidate" if seed.qualified_name in anchor_sources else "workspace-candidate"
        chunks.extend(
            [
                f'\n<target id="{seed.qualified_name}" location="{seed.path}:{seed.line}-{seed.end_line}">',
                f"Signature: {seed.signature}",
                f'<anchor_candidate provenance="{provenance}">',
                "```python",
                anchor_source,
                "```",
                "</anchor_candidate>",
            ]
        )
        incoming = sorted(
            (edge for edge in edges if edge.target == seed.id and edge.relation == "calls"),
            key=_edge_key,
        )
        outgoing = sorted(
            (edge for edge in edges if edge.source == seed.id and edge.relation == "calls"),
            key=_edge_key,
        )
        inherited_consumers = sorted(
            (edge for edge in edges if edge.target == seed.id and edge.relation == "inherits_method"),
            key=_edge_key,
        )
        inherited_subclasses = sorted(
            (edge for edge in edges if edge.target == seed.id and edge.relation == "inherits"),
            key=_edge_key,
        )
        base_classes = sorted(
            (edge for edge in edges if edge.source == seed.id and edge.relation == "inherits"),
            key=_edge_key,
        )
        chunks.append("<upstream_callers>")
        if not incoming:
            chunks.append("(none resolved statically)")
        for edge in incoming:
            caller = by_id[edge.source]
            chunks.extend(
                [
                    f"- {caller.qualified_name} ({caller.path}:{caller.line}) calls target at line {edge.line}",
                    f"  Edge evidence: {edge.evidence}",
                    f"  Static resolution: {edge.resolution}",
                    f"  Caller signature: {caller.signature}",
                    "  ```python",
                    _source_window(root / caller.path, edge.line),
                    "  ```",
                    '  <anchor_inline_projection executable="false">',
                    "  ```python",
                    textwrap.indent(_inline_projection(seed, anchor_source, edge), "  "),
                    "  ```",
                    "  </anchor_inline_projection>",
                ]
            )
        chunks.append("</upstream_callers>")
        chunks.append("<inherited_method_consumers>")
        if not inherited_consumers:
            chunks.append("(none resolved statically)")
        for edge in inherited_consumers:
            consumer = by_id[edge.source]
            chunks.extend(
                [
                    f"- {consumer.qualified_name} ({consumer.path}:{consumer.line}) "
                    f"inherits this implementation",
                    f"  Edge evidence: {edge.evidence}",
                    f"  Static resolution: {edge.resolution}",
                    f"  Consumer signature: {consumer.signature}",
                    "  ```python",
                    _source_slice(
                        consumer.source.splitlines(),
                        1,
                        len(consumer.source.splitlines()),
                        max_lines=12,
                    ),
                    "  ```",
                ]
            )
        chunks.append("</inherited_method_consumers>")
        chunks.append("<inherited_subclasses>")
        if not inherited_subclasses:
            chunks.append("(none resolved statically)")
        for edge in inherited_subclasses:
            subclass = by_id[edge.source]
            chunks.append(
                f"- {subclass.qualified_name} ({subclass.path}:{subclass.line}); "
                f"evidence: {edge.evidence}"
            )
        chunks.append("</inherited_subclasses>")
        chunks.append("<base_classes>")
        if not base_classes:
            chunks.append("(none resolved statically)")
        for edge in base_classes:
            base = by_id[edge.target]
            chunks.append(
                f"- {base.qualified_name} ({base.path}:{base.line}); evidence: {edge.evidence}"
            )
        chunks.append("</base_classes>")
        chunks.append("<downstream_callees>")
        if not outgoing:
            chunks.append("(none resolved statically)")
        for edge in outgoing:
            callee = by_id[edge.target]
            chunks.extend(
                [
                    f"- {callee.qualified_name} ({callee.path}:{callee.line})",
                    f"  Edge evidence: {edge.evidence}",
                    f"  Static resolution: {edge.resolution}",
                    f"  Callee signature: {callee.signature}",
                    "  ```python",
                    _source_slice(callee.source.splitlines(), 1, len(callee.source.splitlines()), max_lines=24),
                    "  ```",
                ]
            )
        chunks.extend(["</downstream_callees>", "</target>"])
    if not seeds:
        chunks.append(
            "<no_seed>Static parsing succeeded, but no changed or explicitly named function could be selected.</no_seed>"
        )
    chunks.append("</repository_graph_rag>")
    text = "\n".join(chunks)
    if len(text) <= max_chars:
        return text
    marker = "\n<!-- context truncated at configured character budget -->\n"
    suffix = f"{marker}</repository_graph_rag>"
    return f"{text[: max(0, max_chars - len(suffix))]}{suffix}"[-max_chars:]


def _source_window(path: Path, line: int, radius: int = 5) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(source unavailable)"
    start = max(1, line - radius)
    end = min(len(lines), line + radius)
    return "\n".join(f"{number:>5}: {lines[number - 1]}" for number in range(start, end + 1))


def _format_arguments(arguments: ast.arguments) -> str:
    names = [argument.arg for argument in [*arguments.posonlyargs, *arguments.args]]
    if arguments.vararg:
        names.append(f"*{arguments.vararg.arg}")
    elif arguments.kwonlyargs:
        names.append("*")
    names.extend(argument.arg for argument in arguments.kwonlyargs)
    if arguments.kwarg:
        names.append(f"**{arguments.kwarg.arg}")
    return ", ".join(names)


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return ""


def _module_name(path: str) -> str:
    value = path[:-3].replace("/", ".")
    return value[: -len(".__init__")] if value.endswith(".__init__") else value


def _line(lines: list[str], line: int) -> str:
    return lines[line - 1].strip() if 0 < line <= len(lines) else ""


def _source_slice(lines: list[str], start: int, end: int, *, max_lines: int) -> str:
    selected = lines[max(0, start - 1) : min(end, start - 1 + max_lines)]
    if end - start + 1 > max_lines:
        selected.append("# ... source truncated ...")
    return "\n".join(selected)


def _definition_key(definition: Definition) -> tuple[str, int, str]:
    return definition.path, definition.line, definition.qualified_name


def _edge_key(edge: Edge) -> tuple[str, int, str, str]:
    return edge.path, edge.line, edge.source, edge.target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed-terms-base64", default="")
    parser.add_argument("--anchor-sources-base64", default="")
    parser.add_argument("--max-files", type=int, default=2500)
    parser.add_argument("--max-seeds", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=14000)
    args = parser.parse_args()
    seed_terms = json.loads(base64.b64decode(args.seed_terms_base64).decode()) if args.seed_terms_base64 else []
    anchor_sources = (
        json.loads(base64.b64decode(args.anchor_sources_base64).decode()) if args.anchor_sources_base64 else {}
    )
    definitions, edges, stats = build_graph(args.root.resolve(), max_files=args.max_files)
    result = retrieve(
        args.root.resolve(),
        definitions,
        edges,
        seed_terms=seed_terms,
        max_seeds=args.max_seeds,
        max_chars=args.max_chars,
        anchor_sources=anchor_sources,
    )
    result["stats"] = stats
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
