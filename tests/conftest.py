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

class _StubBase:
    """A permissive base class returned by stubs so that subclasses compile.

    Accepts any __init__ arguments so that ``super().__init__(...)`` from
    user-defined subclasses does not raise TypeError.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


class _AnyStub(types.ModuleType):
    """Recursive stub: attribute accesses return more stubs.

    Supports patterns like::

        from google.auth.transport.requests import Request   # module attr
        class Foo(SomeLibraryBaseClass): ...                 # base class

    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__path__ = []
        self.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        self.__file__ = f"<stub:{name}>"
        # Python 3.14 import machinery may access __spec__; provide a no-op spec.
        self.__spec__ = importlib.machinery.ModuleSpec(name=name, loader=None)

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

    def __mro_entries__(self, bases):
        # Allow ``class Foo(stub_attr): ...`` to resolve cleanly using the
        # permissive _StubBase that accepts arbitrary __init__ kwargs.
        return (_StubBase,)

    def __class_getitem__(cls, item):
        return cls

    def __getitem__(self, item):
        return _AnyStub(f"{self.__name__}[{item}]")

    def __setitem__(self, key, value):
        pass

    def __delitem__(self, key):
        pass

    def __len__(self):
        return 0

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        # Allow _AnyStub instances to be used as Pydantic field annotations.
        # Treat them as the 'any' type so validation always passes.
        from pydantic_core import core_schema
        return core_schema.any_schema()


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
    # Model providers / optional AI SDKs
    "dashscope",
    "ollama",
    "voyageai",
    # litellm IS now installed — do not stub it
    # "litellm",
    "textgrad",
    "dspy",
    "anthropic",
    # openai IS installed — do not stub it
    # "openai",
    # Vector stores / search
    # faiss IS now installed — do not stub it
    # "faiss",
    "chromadb",
    "pinecone",
    "weaviate",
    "cohere",
    "sentence_transformers",
    # Graph / Neo4j
    "neo4j",
    # Databases
    "psycopg2",
    "pymongo",
    "motor",
    "bson",
    "supabase",
    # Web / browser
    "selenium",
    "webdriver_manager",
    "browser_use",
    "browser_use_py310x",
    # Data / ML
    "torch",
    # datasets IS now installed — do not stub it
    # "datasets",
    # tiktoken IS installed — do not stub it
    # "tiktoken",
    # networkx IS installed — do not stub it
    # "networkx",
    "tkinter",
    "_tkinter",
    "cloudpickle",
    "optuna",
    # regex IS now installed — do not stub it
    # "regex",
    # Document processing
    "PyPDF2",
    "pymupdf",
    "openpyxl",
    "reportlab",
    # Web / scraping
    "feedparser",
    "googlesearch",
    "html2text",
    "wikipedia",
    "ddgs",
    # Python tools / syntax
    # tree_sitter and tree_sitter_python ARE installed — do not stub them
    # "tree_sitter",
    # "tree_sitter_python",
    # Auth
    "passlib",
    "google_auth_oauthlib",
    "googleapiclient",
    # Other
    "overdue",
    "fastmcp",
    "telethon",
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
    # llama_index.core IS now installed — do not stub it
    # "llama_index",
    # "llama_index.core",
    # "llama_index.core.schema",
    "llama_index.core.node_parser",
    "llama_index.core.indices",
    "llama_index.core.storage",
    "llama_index.core.vector_stores",
    "llama_index.core.retrievers",
    "llama_index.core.query_engine",
    "llama_index.core.response_synthesizers",
    "llama_index.core.embeddings",
    "llama_index.core.llms",
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
