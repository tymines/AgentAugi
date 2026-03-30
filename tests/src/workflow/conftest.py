"""
conftest.py for workflow tests.

Stubs optional third-party sub-modules that are not installed so the
evoagentx.workflow package can be imported during pytest collection.

Uses a DFS import probe: try importing each module; on failure, register
a MagicMock so subsequent imports in the same process succeed.
"""

import sys
import types
from unittest.mock import MagicMock


def _stub(name: str) -> None:
    """Register a MagicMock for *name* and all parent packages if absent."""
    if name in sys.modules:
        return
    # Ensure parent packages exist first
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        key = ".".join(parts[:i])
        if key not in sys.modules:
            mock = MagicMock()
            mock.__path__ = []        # make Python treat it as a package
            mock.__package__ = key
            sys.modules[key] = mock


def _try_import(name: str) -> bool:
    """Return True if *name* can be imported; False if ImportError."""
    try:
        __import__(name)
        return True
    except (ImportError, Exception):
        return False


# Stub only what's not importable
_OPTIONAL_MODULES = [
    "browser_use",
    "playwright",
    "playwright.async_api",
    "replicate",
    "feedparser",
    "llama_index.embeddings.azure_openai",
    "llama_index.llms.azure_openai",
]

for _mod in _OPTIONAL_MODULES:
    if not _try_import(_mod):
        _stub(_mod)
