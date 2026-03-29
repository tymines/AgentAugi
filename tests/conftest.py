"""Root conftest.py — auto-stubs optional third-party tool dependencies.

AgentAugi's tool integrations (Telegram, Gmail, RSS, etc.) are optional.
This conftest installs a lightweight import hook so unit tests for pure-logic
modules (evaluators, core) can be collected and run without every optional
tool integration dependency present.

Tests that exercise the real integrations must install the real packages.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types


# ---------------------------------------------------------------------------
# A stub module whose attribute accesses return more stubs, so that patterns
# like ``from google.auth.transport.requests import Request`` succeed silently.
# ---------------------------------------------------------------------------

class _AnyStub(types.ModuleType):
    """Recursive stub: attribute accesses return child stubs."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__path__ = []
        self.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        # torch's inspect.getfile() path requires __file__ to be set; without it
        # torch._library raises "is a built-in module" TypeError.
        self.__file__ = f"<stub:{name}>"

    def __getattr__(self, item: str):
        child_name = f"{self.__name__}.{item}"
        child = _AnyStub(child_name)
        object.__setattr__(self, item, child)
        return child

    def __call__(self, *args, **kwargs):
        return _AnyStub(self.__name__ + ".__result__")

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False


# ---------------------------------------------------------------------------
# Which packages are optional tool integrations.  We probe the environment
# at startup and add anything that fails to import to the stub list.
# ---------------------------------------------------------------------------

def _probe_optional(pkg: str) -> bool:
    """Return True if pkg needs stubbing (not importable)."""
    if pkg in sys.modules:
        return False
    try:
        __import__(pkg)
        return False
    except (ImportError, ModuleNotFoundError):
        return True


# Always-optional: tool integrations that are never in the core env
_ALWAYS_OPTIONAL = [
    "telethon",
    "feedparser",
    "arxiv",
    "serpapi",
    "youtube_transcript_api",
    "newspaper",
    "tweepy",
    "slack_sdk",
    "pyautogui",
    "pynput",
    "pygetwindow",
    "googleapiclient",
    "google_auth_oauthlib",
    "psycopg2",
    "pymongo",
    "browser_use",
    "playwright",
    "duckduckgo_search",
    "mcp",
    "fastmcp",
    "docker",
    "flux",
    "openai_image",
]

# Conditionally optional: Google client libraries may or may not be installed
_GOOGLE_CHAIN = [
    "google",           # top-level namespace must be stubbable when absent
    "google.auth",
    "google.auth.transport",
    "google.auth.transport.requests",
    "google.oauth2",
    "google.oauth2.credentials",
]

# Conditionally optional: llama_index sub-packages that may be absent
_LLAMA_INDEX_OPTIONAL = [
    "llama_index.embeddings.azure_openai",
    "llama_index.embeddings.openai",
    "llama_index.embeddings.huggingface",
    "llama_index.embeddings.cohere",
    "llama_index.llms.azure_openai",
    "llama_index.llms.anthropic",
    "llama_index.llms.cohere",
    "llama_index.vector_stores.chroma",
    "llama_index.vector_stores.pinecone",
    "llama_index.vector_stores.weaviate",
]

# Build the final set of packages that need stubbing
_OPTIONAL: set[str] = set(_ALWAYS_OPTIONAL)

# Add google chain if google.auth is missing
if _probe_optional("google.auth"):
    _OPTIONAL.update(_GOOGLE_CHAIN)

# Add any llama_index sub-packages that are not installed
for _pkg in _LLAMA_INDEX_OPTIONAL:
    if _probe_optional(_pkg):
        _OPTIONAL.add(_pkg)


_OPTIONAL_TUPLE = tuple(_OPTIONAL)


def _needs_stub(fullname: str) -> bool:
    return any(
        fullname == p or fullname.startswith(p + ".")
        for p in _OPTIONAL_TUPLE
    )


class _StubFinder(importlib.abc.MetaPathFinder):
    """Import finder that returns stubs for optional packages."""

    def find_spec(self, fullname, path, target=None):  # noqa: D102
        if fullname in sys.modules:
            return None
        if _needs_stub(fullname):
            return importlib.machinery.ModuleSpec(
                name=fullname,
                loader=_StubLoader(),
                is_package=True,
            )
        return None


class _StubLoader(importlib.abc.Loader):
    """Creates ``_AnyStub`` modules for optional packages."""

    def create_module(self, spec):  # noqa: D102
        return _AnyStub(spec.name)

    def exec_module(self, module):  # noqa: D102
        pass


# Register early so our stubs take priority for optional packages
sys.meta_path.insert(0, _StubFinder())
