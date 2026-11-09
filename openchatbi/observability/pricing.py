"""USD cost estimation for LLM calls.

A tiny static table (USD per 1M tokens) covering the providers OpenChatBI
ships with; unknown models fall back to 0.0 so cost accounting never crashes
on a local/Ollama model. Lookup is case-insensitive longest-prefix so that
versioned names (``gpt-4o-2024-08-06``) resolve to their family price.
"""

from __future__ import annotations

# (input_per_1m, output_per_1m) in USD.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "o3": (2.0, 8.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.8, 4.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-opus-4": (15.0, 75.0),
    # DeepSeek (served through its OpenAI-compatible endpoint). Published rates
    # for cache-miss input / output; they change often, so treat these as an
    # order-of-magnitude estimate rather than a billing figure. The bare
    # `deepseek` entry is a fallback: the API may report a model name that was
    # never configured (e.g. `deepseek-flash` for a `deepseek-chat` request),
    # and an unpriced model would silently report a $0.00 turn.
    "deepseek": (0.28, 0.42),
    "deepseek-chat": (0.28, 0.42),
    "deepseek-flash": (0.28, 0.42),
    "deepseek-reasoner": (0.28, 0.42),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost; unknown models return 0.0."""
    if not model:
        return 0.0
    name = model.lower()
    best: tuple[float, float] | None = None
    best_len = -1
    for prefix, price in _PRICES.items():
        if name.startswith(prefix) and len(prefix) > best_len:
            best, best_len = price, len(prefix)
    if best is None:
        return 0.0
    in_rate, out_rate = best
    return input_tokens / 1_000_000 * in_rate + output_tokens / 1_000_000 * out_rate
