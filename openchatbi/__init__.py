"""OpenChatBI core module initialization.

The global ``config`` object used to be built at import time, which forced every
importer to pay for the whole configuration and LLM stack. That cost is ~9s of
startup, and it is pure waste for the parts of the package that never talk to a
model — most visibly ``openchatbi.patrol``, whose whole point is that a scheduled
run needs no LLM.

So the loader is built lazily now: importing ``openchatbi`` is cheap, and the
cost is paid on first access to ``config``. ``from openchatbi import config``
still works exactly as before for every existing caller.
"""

import os
from typing import Any

__all__ = ["config", "Config", "ConfigLoader", "get_default_graph"]

_config_loader = None


def _build_config_loader():
    """Create and load the global configuration exactly once."""
    from openchatbi.config_loader import Config, ConfigLoader

    loader = ConfigLoader()
    # Skip config loading during documentation build
    if not os.environ.get("SPHINX_BUILD"):
        loader.load()
    else:
        from unittest.mock import MagicMock

        loader._config = Config.from_dict({"default_llm": MagicMock()})
    return loader


def _get_config():
    global _config_loader
    if _config_loader is None:
        _config_loader = _build_config_loader()
    return _config_loader


def __getattr__(name: str) -> Any:
    """Resolve the historically eager module attributes on first use."""
    if name == "config":
        return _get_config()
    if name in ("Config", "ConfigLoader"):
        from openchatbi import config_loader

        return getattr(config_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_default_graph():
    """
    Build the synchronous mode of the agent graph using default catalog in config.

    Returns:
        CompiledStateGraph: Compiled agent graph ready for execution.
    """
    if os.environ.get("SPHINX_BUILD"):
        return None

    from langgraph.checkpoint.memory import MemorySaver

    from openchatbi.agent_graph import build_agent_graph_sync
    from openchatbi.tool.memory import get_sync_memory_store

    checkpointer = MemorySaver()
    return build_agent_graph_sync(
        _get_config().get().catalog_store,
        checkpointer=checkpointer,
        memory_store=get_sync_memory_store(),
    )
