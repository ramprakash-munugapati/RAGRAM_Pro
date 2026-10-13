"""Bridge helpers for calling RAGRAM's async pipeline from synchronous frameworks.

LangChain and LlamaIndex expose sync retriever entry points while RAGRAM is
async end-to-end. :func:`run_sync` runs a coroutine safely from sync code --
including from inside a running event loop (Jupyter, FastAPI workers, asyncio
apps), where a bare ``asyncio.run`` would raise.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Coroutine
from typing import TypeVar

T = TypeVar("T")

__all__ = ["run_sync"]


def run_sync(coro: Coroutine[object, object, T]) -> T:
    """Await ``coro`` from synchronous code.

    * No running loop -> ``asyncio.run`` (zero overhead).
    * Already inside a loop -> run the coroutine on a private event loop in a
      worker thread, so nested/async environments never crash.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ragram-run-sync") as pool:
        return pool.submit(asyncio.run, coro).result()
