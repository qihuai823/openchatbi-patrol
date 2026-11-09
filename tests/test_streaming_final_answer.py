"""Tests for how `AgentStreamProcessor.final_response` is accumulated.

A token stream cannot know whether the round it is streaming is the last one, so
every `llm_node` round appends its tokens. That includes preamble such as
"I'll query the top 3 categories ..." which the model emits before a tool call.
Observed on a real DeepSeek turn: the preamble survived into FINAL ANSWER and
the user saw English chatter glued to the Chinese answer.

The `updates` event arrives *after* that round's tokens and does carry the
answer: `tool_calls` present means the round was not terminal. So the fix is to
drop what was accumulated when a tool-requesting round completes.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk

from openchatbi.streaming import AgentStreamProcessor


def _token(text: str, node: str = "llm_node", streaming: bool = True) -> tuple:
    """Build the `(chunk, metadata)` payload of an astream `messages` event."""
    chunk = AIMessageChunk(content=text)
    return (chunk, {"langgraph_node": node, "streaming_tokens": streaming})


def _tool_call(name: str = "text2sql") -> list[dict]:
    return [{"name": name, "args": {}, "id": "call-1", "type": "tool_call"}]


def _round(processor: AgentStreamProcessor, text: str, tool_calls: list[dict] | None = None) -> None:
    """Simulate one llm_node round: stream the tokens, then deliver the update."""
    processor.process((), "messages", _token(text))
    processor.process(
        (),
        "updates",
        {"llm_node": {"messages": [AIMessage(content=text, tool_calls=tool_calls or [])]}},
    )


def test_single_round_answer_is_kept_verbatim() -> None:
    processor = AgentStreamProcessor()

    _round(processor, "共有 998 位客户。")

    assert processor.final_response == "共有 998 位客户。"


def test_preamble_before_a_tool_call_is_discarded() -> None:
    """The regression this module exists for."""
    processor = AgentStreamProcessor()

    _round(processor, "I'll query the top 3 categories by GMV in May 2026.", _tool_call())
    _round(processor, "| 1 | electronics | 468,457.44 |")

    assert processor.final_response == "| 1 | electronics | 468,457.44 |"
    assert "I'll query" not in processor.final_response


def test_every_intermediate_round_is_discarded_not_just_the_first() -> None:
    processor = AgentStreamProcessor()

    _round(processor, "Let me first check the schema.", _tool_call("search_schema"))
    _round(processor, "Now I'll run the query.", _tool_call("text2sql"))
    _round(processor, "最终答案是 3 个品类。")

    assert processor.final_response == "最终答案是 3 个品类。"


def test_accumulates_across_chunks_within_one_round() -> None:
    """Token streaming splits text arbitrarily; a round must survive splitting."""
    processor = AgentStreamProcessor()

    for piece in ["共", "有 998 ", "位客户。"]:
        processor.process((), "messages", _token(piece))
    processor.process((), "updates", {"llm_node": {"messages": [AIMessage(content="共有 998 位客户。")]}})

    assert processor.final_response == "共有 998 位客户。"


def test_empty_answer_after_tool_round_does_not_leak_preamble() -> None:
    """A tool round followed by a silent final round must not leave the preamble."""
    processor = AgentStreamProcessor()

    _round(processor, "I'll check that.", _tool_call())
    processor.process((), "updates", {"llm_node": {"messages": [AIMessage(content="")]}})

    assert processor.final_response == ""


def test_sub_agent_tokens_never_reach_the_final_answer() -> None:
    processor = AgentStreamProcessor()

    # depth=1 model node of the data_analysis sub-agent
    processor.process((("use_tool:abc",),), "messages", _token("thinking about the data", node="model"))
    _round(processor, "结论：electronics 占比最高。")

    assert processor.final_response == "结论：electronics 占比最高。"
