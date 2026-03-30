"""
Streaming output support for LLM responses.

Provides a token-by-token streaming pipeline that wraps any callable or async
generator producing text chunks.  The pipeline supports:

* **Filters** – drop chunks matching a predicate (e.g. empty tokens).
* **Transforms** – map each chunk to a new value (e.g. normalise whitespace).
* **Accumulators** – collect full output for callers that need the complete
  text after streaming finishes.
* **Backpressure** – a bounded async queue buffers tokens so a slow consumer
  does not cause the producer to spin.
* **Callbacks** – called on each token and on stream completion.

Design
------
Everything is built around :class:`StreamPipeline`, an async generator wrapper.
Sync callables that return a generator (e.g. a mock or simple iterator) are
supported via :func:`stream_sync_generator`.

Typical usage
~~~~~~~~~~~~~
::

    from evoagentx.core.streaming import StreamPipeline, StreamConfig

    async def my_llm_call():
        # Simulated token stream from an LLM
        for token in ["Hello", " world", "!"]:
            yield token

    config = StreamConfig(buffer_size=32)
    pipeline = StreamPipeline(config=config)

    async for token in pipeline.stream(my_llm_call()):
        print(token, end="", flush=True)

    print()
    print("Full output:", pipeline.accumulated_text)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Generator, List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StreamConfig:
    """
    Configuration for a :class:`StreamPipeline` instance.

    Attributes
    ----------
    buffer_size:
        Maximum number of tokens held in the internal queue before
        backpressure applies (producer waits).  0 means unbounded.
    filters:
        List of predicate callables.  A token is dropped if *any* filter
        returns ``True`` for it.
    transforms:
        List of transform callables applied in order to each token that
        passes filtering.
    on_token:
        Optional callback invoked with each emitted token.
    on_complete:
        Optional callback invoked with the full accumulated text when the
        stream ends.
    encoding:
        Label describing the token encoding (informational only).
    """
    buffer_size: int = 64
    filters: List[Callable[[str], bool]] = field(default_factory=list)
    transforms: List[Callable[[str], str]] = field(default_factory=list)
    on_token: Optional[Callable[[str], None]] = None
    on_complete: Optional[Callable[[str], None]] = None
    encoding: str = "utf-8"


# ---------------------------------------------------------------------------
# Stream statistics
# ---------------------------------------------------------------------------

@dataclass
class StreamStats:
    """
    Timing and throughput statistics for a completed stream.

    Attributes
    ----------
    token_count:
        Number of tokens emitted (after filtering and transforming).
    dropped_count:
        Number of tokens discarded by filters.
    elapsed:
        Total wall-clock seconds from first token to stream end.
    tokens_per_second:
        Throughput estimate.
    """
    token_count: int = 0
    dropped_count: int = 0
    elapsed: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed <= 0:
            return 0.0
        return self.token_count / self.elapsed


# ---------------------------------------------------------------------------
# Stream pipeline
# ---------------------------------------------------------------------------

class StreamPipeline:
    """
    Filtering, transforming, and accumulating wrapper around a token stream.

    The pipeline can wrap:
    * An ``async`` generator (most LLM SDKs produce these).
    * A regular (sync) generator or iterable via :meth:`stream_sync`.

    After the stream finishes, :attr:`accumulated_text` holds the full
    concatenated output and :attr:`stats` holds timing metrics.

    Parameters
    ----------
    config:
        :class:`StreamConfig` instance controlling behaviour.
    """

    def __init__(self, config: Optional[StreamConfig] = None) -> None:
        self.config: StreamConfig = config or StreamConfig()
        self._chunks: List[str] = []
        self._stats: StreamStats = StreamStats()
        self._done: bool = False

    # ------------------------------------------------------------------
    # Public streaming API
    # ------------------------------------------------------------------

    async def stream(
        self,
        source: AsyncIterator[str],
    ) -> AsyncGenerator[str, None]:
        """
        Wrap *source* as an async generator that filters, transforms, and
        accumulates tokens before yielding them to the caller.

        Parameters
        ----------
        source:
            An ``async`` iterable of string tokens.

        Yields
        ------
        str
            Each processed token in order.
        """
        self._reset()
        start = time.perf_counter()
        queue: asyncio.Queue[Optional[str]] = (
            asyncio.Queue(maxsize=self.config.buffer_size)
            if self.config.buffer_size > 0
            else asyncio.Queue()
        )

        producer_task = asyncio.create_task(
            self._produce(source, queue)
        )

        try:
            while True:
                token = await queue.get()
                if token is None:
                    break
                yield token
        finally:
            producer_task.cancel()
            try:
                await producer_task
            except (asyncio.CancelledError, Exception):
                pass

        self._stats.elapsed = time.perf_counter() - start
        self._done = True
        if self.config.on_complete:
            self.config.on_complete(self.accumulated_text)

    async def stream_sync(
        self,
        source: Generator[str, None, None],
    ) -> AsyncGenerator[str, None]:
        """
        Wrap a synchronous generator *source* as an async stream.

        The generator is consumed in a thread pool executor to avoid
        blocking the event loop.

        Parameters
        ----------
        source:
            A synchronous iterable of string tokens.

        Yields
        ------
        str
            Each processed token in order.
        """
        async def _async_gen() -> AsyncGenerator[str, None]:
            loop = asyncio.get_running_loop()
            iterator = iter(source)
            sentinel = object()
            while True:
                chunk = await loop.run_in_executor(
                    None, lambda: next(iterator, sentinel)
                )
                if chunk is sentinel:
                    break
                yield chunk  # type: ignore[misc]

        async for token in self.stream(_async_gen()):
            yield token

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    @property
    def accumulated_text(self) -> str:
        """Full text assembled from all emitted tokens."""
        return "".join(self._chunks)

    @property
    def stats(self) -> StreamStats:
        """Streaming statistics (only meaningful after stream finishes)."""
        return self._stats

    @property
    def is_done(self) -> bool:
        """True once the stream has finished."""
        return self._done

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    def add_filter(self, fn: Callable[[str], bool]) -> "StreamPipeline":
        """Add a filter predicate.  Returns self for chaining."""
        self.config.filters.append(fn)
        return self

    def add_transform(self, fn: Callable[[str], str]) -> "StreamPipeline":
        """Add a transform function.  Returns self for chaining."""
        self.config.transforms.append(fn)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._chunks = []
        self._stats = StreamStats()
        self._done = False

    async def _produce(
        self,
        source: AsyncIterator[str],
        queue: asyncio.Queue,
    ) -> None:
        """
        Consume *source*, apply filters/transforms, enqueue processed tokens.
        Sends ``None`` as a sentinel when the source is exhausted.
        """
        try:
            async for raw_token in source:
                # Filtering
                if any(f(raw_token) for f in self.config.filters):
                    self._stats.dropped_count += 1
                    continue
                # Transforming
                token: str = raw_token
                for transform in self.config.transforms:
                    token = transform(token)
                # Accumulate
                self._chunks.append(token)
                self._stats.token_count += 1
                # Callback
                if self.config.on_token:
                    self.config.on_token(token)
                # Enqueue (may block on backpressure)
                await queue.put(token)
        finally:
            await queue.put(None)  # sentinel


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def drop_empty_tokens(token: str) -> bool:
    """Filter predicate: drop blank / whitespace-only tokens."""
    return not token.strip()


def strip_whitespace(token: str) -> str:
    """Transform: strip leading/trailing whitespace from each token."""
    return token.strip()


async def collect_stream(
    source: AsyncIterator[str],
    config: Optional[StreamConfig] = None,
) -> str:
    """
    Convenience coroutine: fully consume *source* and return the accumulated text.

    Parameters
    ----------
    source:
        An async iterable of string tokens.
    config:
        Optional pipeline configuration.

    Returns
    -------
    str
        Concatenated output.
    """
    pipeline = StreamPipeline(config=config)
    async for _ in pipeline.stream(source):
        pass
    return pipeline.accumulated_text


async def stream_to_list(
    source: AsyncIterator[str],
    config: Optional[StreamConfig] = None,
) -> List[str]:
    """
    Convenience coroutine: collect all tokens from *source* into a list.

    Parameters
    ----------
    source:
        An async iterable of string tokens.
    config:
        Optional pipeline configuration.

    Returns
    -------
    List[str]
        List of emitted tokens (after filtering and transforming).
    """
    pipeline = StreamPipeline(config=config)
    tokens: List[str] = []
    async for token in pipeline.stream(source):
        tokens.append(token)
    return tokens


__all__ = [
    "StreamPipeline",
    "StreamConfig",
    "StreamStats",
    "collect_stream",
    "stream_to_list",
    "drop_empty_tokens",
    "strip_whitespace",
]
