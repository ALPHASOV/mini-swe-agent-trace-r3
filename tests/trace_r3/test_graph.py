import subprocess

from minisweagent.trace_r3.graph_worker import build_graph, retrieve


def test_one_hop_graph_is_directly_flattened_for_the_llm(tmp_path):
    path = tmp_path / "flow.py"
    path.write_text(
        "\n".join(
            [
                "def leaf(value):",
                "    return value + 1",
                "",
                "def target(value):",
                "    return leaf(value)",
                "",
                "def caller():",
                "    return target(1)",
                "",
                "def second_hop():",
                "    return caller()",
                "",
            ]
        )
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "flow.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
        cwd=tmp_path,
        check=True,
    )
    path.write_text(path.read_text().replace("return leaf(value)", "return leaf(value) + 1"))

    definitions, edges, stats = build_graph(tmp_path, max_files=50)
    result = retrieve(
        tmp_path,
        definitions,
        edges,
        seed_terms=[],
        max_seeds=3,
        max_chars=20_000,
    )

    names = {node["qualified_name"] for node in result["retrieved_nodes"]}
    edge_names = {
        (
            next(node["qualified_name"] for node in result["retrieved_nodes"] if node["id"] == edge["source"]),
            next(node["qualified_name"] for node in result["retrieved_nodes"] if node["id"] == edge["target"]),
        )
        for edge in result["retrieved_edges"]
    }
    assert result["depth"] == 1
    assert result["construction"] == "repograph_one_hop_static_python_ast_no_llm"
    assert {"flow.target", "flow.caller", "flow.leaf"} <= names
    assert "flow.second_hop" not in names
    assert ("flow.caller", "flow.target") in edge_names
    assert ("flow.target", "flow.leaf") in edge_names
    assert ("flow.second_hop", "flow.caller") not in edge_names
    assert "<upstream_callers>" in result["rag_context"]
    assert "<downstream_callees>" in result["rag_context"]
    assert '<anchor_inline_projection executable="false">' in result["rag_context"]
    assert "Parameter substitution: value <- 1" in result["rag_context"]
    assert "__trace_r3_result = leaf(1) + 1" in result["rag_context"]
    assert "Assignment redirection: return __trace_r3_result" in result["rag_context"]
    assert "Retrieval contract: only direct callers and direct callees" in result["rag_context"]
    assert result["anchor_sources"] == {
        "flow.target": "def target(value):\n    return leaf(value) + 1"
    }
    assert stats["resolved_edges"] == 3


def test_graph_context_respects_character_budget(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("def target(value):\n    return value\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "module.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
        cwd=tmp_path,
        check=True,
    )
    path.write_text("def target(value):\n    return value + 1\n")
    definitions, edges, _ = build_graph(tmp_path, max_files=10)

    result = retrieve(
        tmp_path,
        definitions,
        edges,
        seed_terms=[],
        max_seeds=2,
        max_chars=450,
    )

    assert len(result["rag_context"]) <= 450
    assert "context truncated" in result["rag_context"]


def test_import_alias_resolution_finds_the_exact_cross_module_callee(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "helpers.py").write_text("def leaf(value):\n    return value + 1\n")
    app = tmp_path / "app.py"
    app.write_text(
        "from pkg.helpers import leaf as imported_leaf\n\n"
        "def target(value):\n"
        "    return imported_leaf(value)\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
        cwd=tmp_path,
        check=True,
    )
    app.write_text(app.read_text().replace("imported_leaf(value)", "imported_leaf(value) + 1"))

    definitions, edges, _ = build_graph(tmp_path, max_files=20)
    result = retrieve(
        tmp_path,
        definitions,
        edges,
        seed_terms=[],
        max_seeds=2,
        max_chars=10_000,
    )

    assert {node["qualified_name"] for node in result["retrieved_nodes"]} == {
        "app.target",
        "pkg.helpers.leaf",
    }
    assert result["retrieved_edges"][0]["resolution"] == "qualified_import_or_scope"


def test_external_failed_anchor_is_projected_over_a_clean_base_graph(tmp_path):
    path = tmp_path / "flow.py"
    path.write_text(
        "def leaf(value):\n"
        "    return value + 1\n\n"
        "def target(value):\n"
        "    return value\n\n"
        "def caller():\n"
        "    return target(5)\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "flow.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
        cwd=tmp_path,
        check=True,
    )
    definitions, edges, _ = build_graph(tmp_path, max_files=10)

    result = retrieve(
        tmp_path,
        definitions,
        edges,
        seed_terms=["flow.target"],
        max_seeds=2,
        max_chars=10_000,
        anchor_sources={"flow.target": "def target(value):\n    return leaf(value)\n"},
    )

    assert result["changed_ranges"] == {}
    assert result["anchor_provenance"] == "external_failed_candidate"
    assert {node["qualified_name"] for node in result["retrieved_nodes"]} == {
        "flow.caller",
        "flow.leaf",
        "flow.target",
    }
    assert "external-failed-candidate" in result["rag_context"]
    assert "__trace_r3_result = leaf(5)" in result["rag_context"]


def test_ambiguous_issue_term_does_not_crowd_out_changed_function(tmp_path):
    changed = tmp_path / "changed.py"
    changed.write_text(
        "def target(value):\n"
        "    return value\n\n"
        "def relevant(value):\n"
        "    return target(value)\n"
    )
    for index in range(5):
        (tmp_path / f"noise_{index}.py").write_text(
            f"def symbols(value):\n    return value + {index}\n"
        )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
        cwd=tmp_path,
        check=True,
    )
    changed.write_text(changed.read_text().replace("return value", "return value + 1", 1))

    definitions, edges, _ = build_graph(tmp_path, max_files=20)
    result = retrieve(
        tmp_path,
        definitions,
        edges,
        seed_terms=["symbols", "relevant"],
        max_seeds=3,
        max_chars=10_000,
    )

    seed_names = {
        next(node["qualified_name"] for node in result["retrieved_nodes"] if node["id"] == seed_id)
        for seed_id in result["seed_ids"]
    }
    assert "changed.target" in seed_names
    assert "changed.relevant" in seed_names
    assert not any(name.endswith(".symbols") for name in seed_names)


def test_changed_method_retrieves_transitive_inheritance_consumers(tmp_path):
    base = tmp_path / "base.py"
    base.write_text(
        "class Base:\n"
        "    def target(self, value):\n"
        "        return value\n"
    )
    (tmp_path / "children.py").write_text(
        "from base import Base\n\n"
        "class Child(Base):\n"
        "    pass\n\n"
        "class GrandChild(Child):\n"
        "    pass\n\n"
        "class Override(Base):\n"
        "    def target(self, value):\n"
        "        return value * 2\n\n"
        "class BelowOverride(Override):\n"
        "    pass\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
        cwd=tmp_path,
        check=True,
    )
    base.write_text(base.read_text().replace("return value", "return value + 1"))

    definitions, edges, _ = build_graph(tmp_path, max_files=20)
    result = retrieve(
        tmp_path,
        definitions,
        edges,
        seed_terms=[],
        max_seeds=3,
        max_chars=20_000,
    )

    inherited = {
        next(node["qualified_name"] for node in result["retrieved_nodes"] if node["id"] == edge["source"])
        for edge in result["retrieved_edges"]
        if edge["relation"] == "inherits_method"
    }
    assert {"children.Child", "children.GrandChild"} <= inherited
    assert "children.Override" not in inherited
    assert "children.BelowOverride" not in inherited
    assert "<inherited_method_consumers>" in result["rag_context"]
    assert "children.GrandChild" in result["rag_context"]


def test_unbound_method_inline_projection_drops_explicit_receiver(tmp_path):
    base = tmp_path / "base.py"
    base.write_text(
        "def leaf(value):\n"
        "    return value + 1\n\n"
        "class Base:\n"
        "    def target(self, value):\n"
        "        return leaf(value)\n"
    )
    (tmp_path / "child.py").write_text(
        "from base import Base\n\n"
        "class Child(Base):\n"
        "    def target(self, value):\n"
        "        return Base.target(self, value)\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
        cwd=tmp_path,
        check=True,
    )
    base.write_text(base.read_text().replace("return leaf(value)", "return leaf(value) + 1"))

    definitions, edges, _ = build_graph(tmp_path, max_files=10)
    result = retrieve(
        tmp_path,
        definitions,
        edges,
        seed_terms=[],
        max_seeds=2,
        max_chars=20_000,
    )

    assert "Parameter substitution: value <- value" in result["rag_context"]
    assert "Parameter substitution: value <- self" not in result["rag_context"]
