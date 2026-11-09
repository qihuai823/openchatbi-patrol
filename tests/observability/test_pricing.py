"""Tests for cost estimation."""

from openchatbi.observability.pricing import estimate_cost


def test_known_model_cost() -> None:
    # gpt-4o: $2.5/1M input, $10/1M output → 1000 in + 500 out.
    cost = estimate_cost("gpt-4o", 1000, 500)
    assert abs(cost - (1000 / 1_000_000 * 2.5 + 500 / 1_000_000 * 10.0)) < 1e-9


def test_prefix_match_is_case_insensitive() -> None:
    # Provider-prefixed / versioned names resolve via prefix lookup.
    assert estimate_cost("GPT-4o-2024-08-06", 1000, 1000) > 0.0


def test_unknown_model_returns_zero() -> None:
    assert estimate_cost("some-local-ollama-model", 9999, 9999) == 0.0


def test_deepseek_is_priced() -> None:
    """DeepSeek is a first-class provider here; a $0.00 turn would be a lie."""
    cost = estimate_cost("deepseek-chat", 1_000_000, 1_000_000)
    assert cost > 0.0
    assert abs(cost - (0.28 + 0.42)) < 1e-9


def test_deepseek_variants_resolve_by_prefix() -> None:
    assert estimate_cost("deepseek-chat-v3", 1_000_000, 0) > 0.0
    assert estimate_cost("DeepSeek-Reasoner", 0, 1_000_000) > 0.0


def test_deepseek_flash_is_priced() -> None:
    """The API can report a model name the caller never configured.

    Requesting ``deepseek-chat`` returned ``deepseek-flash`` in
    ``response_metadata``, which made every turn report a $0.00 cost until the
    bare ``deepseek`` prefix was added as a fallback.
    """
    assert estimate_cost("deepseek-flash", 1_000_000, 0) > 0.0
    assert estimate_cost("deepseek-something-new", 1_000_000, 0) > 0.0
