"""Declarative node-role table used by :mod:`openchatbi.streaming`.

``streaming.AgentStreamProcessor`` has to answer one question for every node
name it sees in an astream triple: *what is this node, and how should its events
be rendered?* The answer used to be three module-level sets hardcoded in
``streaming.py``, plus an implicit "anything else at depth 0 is the data_analysis
sub-agent" rule. All four failure modes of that approach are silent:

1. a new main-graph node renders as a nested sub-agent (level 1, ``data_analysis``);
2. renaming the LLM node kills ``is_final`` token detection and the final answer
   silently becomes ``""``;
3. a new SQL-subgraph node stops having its token stream suppressed, leaking raw
   SQL deltas into the "thinking" channel;
4. a new sub-agent tool is reported as an ordinary tool.

``NodeRoleTable()`` reproduces today's behaviour exactly (see the golden test
``test_default_table_matches_legacy_behaviour``), while every assumption is
overridable and unclassified depth-0 nodes are recorded instead of guessed away.

A note on ``unknown_depth0``, measured rather than assumed: in practice every
sub-agent node arrives *nested* (``depth=1``), so the depth-0 fallback is a
safety net that a real turn never reaches - ``unclassified_depth0`` stayed empty
across text2sql, python-analysis and data_analysis turns. It is kept because the
alternative (raising or mislabelling) is worse than recording, not because it
carries the main path.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

_GRAPH_SENTINELS = frozenset({"__start__", "__end__"})


@dataclass(frozen=True)
class NodeRole:
    """How a single graph node's stream events should be classified."""

    level: int
    """0 = top-level agent, >0 = nested sub-agent / subgraph layer."""

    label: str | None = None
    """Display label. ``None`` means "derive it from the astream namespace"."""

    suppress_tokens: bool = False
    """Drop this node's LLM token deltas (``updates`` handlers render them)."""

    answer_node: bool = False
    """This node streams the top-level agent's answer tokens."""


MAIN = NodeRole(level=0, label="main")
ANSWER = NodeRole(level=0, label="main", answer_node=True)
#: Role for a sub-agent node seen at depth 0. In practice sub-agent nodes arrive
#: nested, so this only backs the defensive depth-0 fallback.
FLATTENED_SUB_AGENT = NodeRole(level=1, label="data_analysis")
SQL_SUBGRAPH = NodeRole(level=1, suppress_tokens=True)


class UnclassifiedNodeError(RuntimeError):
    """Raised in ``unknown_depth0="error"`` mode for an undeclared depth-0 node."""


# --- historical defaults: keep these in sync with the graphs they describe ----

DEFAULT_MAIN_NODES = frozenset({"llm_node", "use_tool", "ask_human"})

DEFAULT_SQL_NODES = frozenset(
    {
        "search_knowledge",
        "ask_human",
        "information_extraction",
        "table_selection",
        "generate_sql",
        "execute_sql",
        "regenerate_sql",
        "generate_visualization",
        "score_sql",
        "confidence_gate",
    }
)

# deepagents' create_deep_agent nodes.
#
# Measured against a real turn (LangGraph 1.x, `subgraphs=True`), these arrive
# *nested* - `depth=1`, namespace `('use_tool:<uuid>',)` - not flattened onto
# depth 0. The comment in the pre-refactor code claimed otherwise, and that
# claim is what made the "anything unknown at depth 0 is the data_analysis
# sub-agent" rule look load-bearing.
#
# They stay declared anyway: the depth-0 fallback is cheap insurance against a
# future LangGraph/deepagents version that does flatten them, and `model`/`tools`
# are still the names to match if it happens.
DEFAULT_SUB_AGENT_NODES = frozenset({"model", "tools"})

DEFAULT_ANSWER_NODE = "llm_node"

DEFAULT_SUB_AGENT_TOOLS = frozenset({"data_analysis", "text2sql"})

DEFAULT_SUB_AGENT_LABEL = "data_analysis"

_UNKNOWN_DEPTH0_MODES = frozenset({"sub_agent", "main", "error"})


class NodeRoleTable:
    """Maps ``(node_name, depth)`` to a :class:`NodeRole`."""

    def __init__(
        self,
        *,
        main_nodes: Iterable[str] = DEFAULT_MAIN_NODES,
        sql_nodes: Iterable[str] = DEFAULT_SQL_NODES,
        sub_agent_nodes: Iterable[str] = DEFAULT_SUB_AGENT_NODES,
        answer_node: str = DEFAULT_ANSWER_NODE,
        sub_agent_tools: Iterable[str] = DEFAULT_SUB_AGENT_TOOLS,
        overrides: Mapping[str, NodeRole] | None = None,
        unknown_depth0: str = "sub_agent",
    ) -> None:
        if unknown_depth0 not in _UNKNOWN_DEPTH0_MODES:
            raise ValueError(
                f"unknown_depth0 must be one of {sorted(_UNKNOWN_DEPTH0_MODES)}, got {unknown_depth0!r}"
            )
        self.main_nodes = frozenset(main_nodes)
        self.sql_nodes = frozenset(sql_nodes)
        self.sub_agent_nodes = frozenset(sub_agent_nodes)
        self.answer_node = answer_node
        self.sub_agent_tools = frozenset(sub_agent_tools)
        self.unknown_depth0 = unknown_depth0
        self._overrides = dict(overrides or {})
        # `ask_human` is the one node that legitimately belongs to both the main
        # graph and the SQL subgraph. Pin its role so precedence is explicit
        # instead of depending on which set a given function happens to check
        # first (that used to differ between _process_message and _process_updates).
        self._overrides.setdefault(
            "ask_human", NodeRole(level=0, label="main", suppress_tokens=True)
        )
        #: depth-0 node names this table could not classify (observability hook).
        self.unclassified_depth0: set[str] = set()

    # -- resolution -----------------------------------------------------------

    def resolve(self, node_name: str | None, depth: int, layer_label: str) -> NodeRole:
        """Return the role for ``node_name`` seen at namespace ``depth``."""
        if node_name is None:
            # No `langgraph_node` metadata (e.g. a bare model stream).
            return NodeRole(level=depth, label=layer_label)
        override = self._overrides.get(node_name)
        if override is not None:
            return self._at_depth(override, depth, layer_label)
        if node_name in self.sql_nodes:
            return self._at_depth(SQL_SUBGRAPH, depth, layer_label)
        if depth > 0:
            # Nested event: the namespace already tells us the layer, and the
            # label should mirror the namespace path.
            return NodeRole(level=depth, label=layer_label)
        if node_name == self.answer_node:
            return ANSWER
        if node_name in self.main_nodes:
            return MAIN
        if node_name in self.sub_agent_nodes:
            return FLATTENED_SUB_AGENT
        return self._unclassified(node_name)

    def _at_depth(self, role: NodeRole, depth: int, layer_label: str) -> NodeRole:
        level = role.level if depth == 0 else max(role.level, depth)
        if role.label is not None:
            label = role.label
        else:
            # A label-less role normally mirrors the namespace, but depth 0 has
            # no namespace path -- fall back to the sub-agent label, which is
            # what the legacy code produced in that (practically unreachable)
            # case.
            label = layer_label if depth > 0 else DEFAULT_SUB_AGENT_LABEL
        if level == role.level and label == role.label:
            return role
        return replace(role, level=level, label=label)

    def _unclassified(self, node_name: str) -> NodeRole:
        if self.unknown_depth0 == "error":
            raise UnclassifiedNodeError(
                f"node {node_name!r} appeared at depth 0 but is not declared in the "
                "NodeRoleTable; add it to main_nodes/sub_agent_nodes or pass overrides="
            )
        self.unclassified_depth0.add(node_name)
        return FLATTENED_SUB_AGENT if self.unknown_depth0 == "sub_agent" else MAIN

    # -- introspection --------------------------------------------------------

    @classmethod
    def from_graph(
        cls,
        graph: Any,
        *,
        sql_graph: Any | None = None,
        sub_agent_nodes: Iterable[str] = DEFAULT_SUB_AGENT_NODES,
        **kwargs: Any,
    ) -> NodeRoleTable:
        """Build a table from a compiled graph instead of the hardcoded defaults.

        ``graph`` is the top-level ``CompiledStateGraph``. ``sql_graph`` is the
        text2sql graph *if* the caller holds a reference to it -- it cannot be
        discovered from ``graph`` because this project wraps it in a
        ``StructuredTool`` and calls ``.invoke()`` inside ``use_tool``, so it
        never shows up as a node of the main graph.

        ``sub_agent_nodes`` stays explicit on purpose: deepagents' internal node
        names are not reachable without an instance of the compiled sub-agent.
        """
        main_nodes, answer_node = _graph_shape(graph)
        sql_nodes = _node_names(sql_graph) if sql_graph is not None else DEFAULT_SQL_NODES
        return cls(
            main_nodes=main_nodes,
            sql_nodes=sql_nodes,
            sub_agent_nodes=sub_agent_nodes,
            answer_node=answer_node,
            **kwargs,
        )


def _node_names(graph: Any) -> frozenset[str]:
    drawable = graph.get_graph()
    return frozenset(name for name in drawable.nodes if name not in _GRAPH_SENTINELS)


def _graph_shape(graph: Any) -> tuple[frozenset[str], str]:
    """Return ``(node names, entry node)`` for a compiled graph.

    The entry node is the edge target of ``__start__``; in this project that is
    always the agent's LLM node, which is what ``answer_node`` needs.
    """
    drawable = graph.get_graph()
    names = frozenset(n for n in drawable.nodes if n not in _GRAPH_SENTINELS)
    entry = next(
        (edge.target for edge in drawable.edges if edge.source == "__start__"),
        DEFAULT_ANSWER_NODE,
    )
    return names, entry
