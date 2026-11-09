"""Node-role classification for stream events.

`streaming.AgentStreamProcessor` used to decide "is this node the main agent, a
sub-agent, or an internal SQL-subgraph node" from three hardcoded name sets. The
tests below lock the *legacy* behaviour (so the refactor is provably behaviour
preserving) and pin down the four cases that used to fail silently.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from openchatbi.node_roles import (
    DEFAULT_SQL_NODES,
    DEFAULT_SUB_AGENT_TOOLS,
    NodeRole,
    NodeRoleTable,
    UnclassifiedNodeError,
)
from openchatbi.streaming import AgentStreamProcessor, StreamStep, StreamToken

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

SQL_LAYER_LABEL = "use_tool"


def _ns(depth: int) -> tuple:
    """Fake astream namespace of a given depth (node-name prefix kept)."""
    return tuple(f"{SQL_LAYER_LABEL}:uuid-{i}" for i in range(depth))


def _msg_event(node: str | None, text: str = "hi", *, depth: int = 0, streaming: bool = True) -> tuple:
    chunk = AIMessageChunk(content=text)
    metadata = {"langgraph_node": node, "streaming_tokens": streaming}
    return (_ns(depth), "messages", (chunk, metadata))


def _tool_call_event(node: str, tool_name: str, *, depth: int = 0) -> tuple:
    message = AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": {"reasoning": "because"}, "id": "call_1"}],
    )
    return (_ns(depth), "updates", {node: {"messages": [message]}})


# ---------------------------------------------------------------------------
# 1. golden test: the default table must reproduce the legacy behaviour exactly
# ---------------------------------------------------------------------------

# (node, depth) -> (level, label, suppress_tokens)
# Expected values are transcribed by hand from the pre-refactor code in
# streaming.py (_process_message lines 244-259, _process_updates lines 266-273)
# so that a behavioural drift shows up here rather than in a UI screenshot.
LEGACY_EXPECTATIONS: list[tuple[str, int, int, str, bool]] = [
    # main-graph nodes, depth 0
    ("llm_node", 0, 0, "main", False),
    ("use_tool", 0, 0, "main", False),
    # ask_human is in both sets; the token stream was suppressed because the
    # SQL set was checked before the main set in _process_message.
    ("ask_human", 0, 0, "main", True),
    # deepagents nodes flattened onto depth 0 -> data_analysis sub-agent layer
    ("model", 0, 1, "data_analysis", False),
    ("tools", 0, 1, "data_analysis", False),
] + [
    # SQL subgraph nodes, nested at depth 1 -> label mirrors the namespace.
    # `ask_human` is also in this set but has its own row below.
    (node, 1, 1, SQL_LAYER_LABEL, True) for node in sorted(DEFAULT_SQL_NODES) if node != "ask_human"
] + [
    # a main-graph node name seen nested is just another nested layer
    ("llm_node", 1, 1, SQL_LAYER_LABEL, False),
    ("model", 1, 1, SQL_LAYER_LABEL, False),
    # Known edge: `ask_human` is pinned by an explicit override, and an explicit
    # label wins over the namespace path. It never actually nests (it is a
    # main-graph node), so this only documents the precedence rule.
    ("ask_human", 1, 1, "main", True),
]


@pytest.mark.parametrize(("node", "depth", "level", "label", "suppress"), LEGACY_EXPECTATIONS)
def test_default_table_matches_legacy_behaviour(
    node: str, depth: int, level: int, label: str, suppress: bool
) -> None:
    table = NodeRoleTable()
    layer_label = SQL_LAYER_LABEL if depth else "main"

    role = table.resolve(node, depth, layer_label)

    assert (role.level, role.label, role.suppress_tokens) == (level, label, suppress)
    # Nothing should be flagged as unclassifiable for the graphs we ship with.
    assert table.unclassified_depth0 == set()


def test_default_table_only_marks_the_llm_node_as_answer_node() -> None:
    table = NodeRoleTable()
    assert table.resolve("llm_node", 0, "main").answer_node is True
    for node in ("use_tool", "ask_human", "model", "generate_sql"):
        assert table.resolve(node, 0, "main").answer_node is False


# ---------------------------------------------------------------------------
# 2. defect 1: a new main-graph node must not be rendered as a sub-agent
# ---------------------------------------------------------------------------


def test_new_main_node_is_classified_as_main() -> None:
    table = NodeRoleTable(unknown_depth0="main")

    role = table.resolve("patrol_status", 0, "main")

    assert (role.level, role.label) == (0, "main")
    assert table.unclassified_depth0 == {"patrol_status"}


def test_new_main_node_step_is_not_labelled_data_analysis() -> None:
    processor = AgentStreamProcessor(NodeRoleTable(unknown_depth0="main"))

    events = processor.process((), "updates", {"patrol_status": {"open_alerts": 3}})

    # No dedicated handler -> generic fallback, but at level 0 / label "main".
    assert all(event.level == 0 and event.label == "main" for event in events)


# ---------------------------------------------------------------------------
# 3./4. the legacy fallback stays, but becomes observable and opt-out-able
# ---------------------------------------------------------------------------


def test_unknown_depth0_node_is_recorded_and_falls_back_to_legacy() -> None:
    processor = AgentStreamProcessor()

    events = processor.process(*_tool_call_event("patrol_status", "search_schema"))

    assert "patrol_status" in processor.unclassified_nodes
    assert len(events) == 1
    assert events[0].level == 1
    assert events[0].label == "data_analysis"


def test_unknown_depth0_error_mode_raises() -> None:
    table = NodeRoleTable(unknown_depth0="error")

    with pytest.raises(UnclassifiedNodeError, match="patrol_status"):
        table.resolve("patrol_status", 0, "main")


def test_unknown_depth0_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="unknown_depth0"):
        NodeRoleTable(unknown_depth0="nope")


def test_nested_unknown_node_never_counts_as_unclassified() -> None:
    table = NodeRoleTable()

    role = table.resolve("validate_sql", 2, "use_tool › tools")

    assert (role.level, role.label) == (2, "use_tool › tools")
    assert table.unclassified_depth0 == set()


# ---------------------------------------------------------------------------
# 5. defect 2: renaming the LLM node used to empty the final answer
# ---------------------------------------------------------------------------


def test_renamed_llm_node_still_marks_final_answer() -> None:
    processor = AgentStreamProcessor(NodeRoleTable(answer_node="agent_llm"))

    events = processor.process(*_msg_event("agent_llm", "the answer"))

    assert len(events) == 1
    token = events[0]
    assert isinstance(token, StreamToken)
    assert token.is_final is True
    assert token.level == 0
    assert processor.final_response == "the answer"


def test_stale_llm_node_name_no_longer_emits_final_tokens() -> None:
    """Guards the mirror image of defect 2: the override must actually take over."""
    processor = AgentStreamProcessor(NodeRoleTable(answer_node="agent_llm"))

    events = processor.process(*_msg_event("llm_node", "the answer"))

    assert events == []
    assert processor.final_response == ""


def test_answer_node_requires_streaming_tokens_flag() -> None:
    processor = AgentStreamProcessor()

    events = processor.process(*_msg_event("llm_node", "x", streaming=False))

    assert events == []
    assert processor.final_response == ""


# ---------------------------------------------------------------------------
# 6. defect 3: a new SQL-subgraph node must keep its token stream suppressed
# ---------------------------------------------------------------------------


def test_new_sql_node_tokens_are_suppressed() -> None:
    table = NodeRoleTable(sql_nodes=DEFAULT_SQL_NODES | {"validate_sql"})
    processor = AgentStreamProcessor(table)

    events = processor.process(*_msg_event("validate_sql", "SELECT", depth=1))

    assert events == []


def test_undeclared_sql_node_leaks_tokens_until_it_is_declared() -> None:
    """Documents why the sql_nodes set has to stay in sync with the subgraph."""
    processor = AgentStreamProcessor()

    events = processor.process(*_msg_event("validate_sql", "SELECT", depth=1))

    assert len(events) == 1
    assert isinstance(events[0], StreamToken)
    assert events[0].level == 1


# ---------------------------------------------------------------------------
# 7. defect 4: a third sub-agent tool must be reported as a sub-agent
# ---------------------------------------------------------------------------


def test_third_sub_agent_tool_is_reported_as_sub_agent() -> None:
    processor = AgentStreamProcessor(NodeRoleTable(sub_agent_tools=DEFAULT_SUB_AGENT_TOOLS | {"web_research"}))

    events = processor.process(*_tool_call_event("llm_node", "web_research"))

    assert len(events) == 1
    step = events[0]
    assert isinstance(step, StreamStep)
    assert step.kind == "sub_agent"
    assert "Running sub agent: web_research" in step.text


def test_undeclared_sub_agent_tool_is_reported_as_plain_tool() -> None:
    """Mirror image of the previous test, pinning the legacy default."""
    processor = AgentStreamProcessor()

    events = processor.process(*_tool_call_event("llm_node", "web_research"))

    assert events[0].kind == "tool"
    assert "Using tools: web_research" in events[0].text


# ---------------------------------------------------------------------------
# 8. introspection: derive the table from the graph instead of hardcoding it
# ---------------------------------------------------------------------------


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _compiled_graph():
    def agent_llm(state: _State) -> dict:
        return {}

    def use_tool(state: _State) -> dict:
        return {}

    graph = StateGraph(_State)
    graph.add_node("agent_llm", agent_llm)
    graph.add_node("use_tool", use_tool)
    graph.add_edge(START, "agent_llm")
    graph.add_edge("use_tool", "agent_llm")
    return graph.compile()


def test_from_graph_infers_nodes_and_answer_node() -> None:
    table = NodeRoleTable.from_graph(_compiled_graph())

    assert table.main_nodes == frozenset({"agent_llm", "use_tool"})
    assert table.answer_node == "agent_llm"
    # Without an explicit sql_graph the historical SQL node list is kept.
    assert table.sql_nodes == DEFAULT_SQL_NODES


def test_from_graph_suppresses_sentinels_and_accepts_sql_graph() -> None:
    sql_graph = _compiled_graph()

    table = NodeRoleTable.from_graph(_compiled_graph(), sql_graph=sql_graph)

    assert "__start__" not in table.main_nodes
    assert "__end__" not in table.main_nodes
    assert table.sql_nodes == frozenset({"agent_llm", "use_tool"})


def test_from_graph_answer_node_drives_final_answer() -> None:
    processor = AgentStreamProcessor(NodeRoleTable.from_graph(_compiled_graph()))

    events = processor.process(*_msg_event("agent_llm", "derived"))

    assert isinstance(events[0], StreamToken)
    assert events[0].is_final is True
    assert processor.final_response == "derived"


def test_node_role_is_immutable() -> None:
    role = NodeRole(level=2, label="x")
    with pytest.raises(FrozenInstanceError):
        role.level = 3  # type: ignore[misc]
