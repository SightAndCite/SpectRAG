"""Bounded concurrent execution for index-time inference.

`ThreadPoolExecutor.map` submits the entire iterable at once, so extracting over
a million chunks materialises a million Futures before any work starts. Both LLM
extractors did that, and the embedder did the opposite — one batch in flight at a
time, so a 3M-vector pass is ~94,000 serialised round-trips and is latency-bound
no matter how fast the server is.

`bounded_map` keeps a fixed number of tasks alive: enough to saturate the
endpoint, never more than the configured ceiling, and results are returned in
input order regardless of completion order.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Outcome:
    """Per-item success/failure record for one inference stage."""
    stage: str
    total: int = 0
    ok: int = 0
    failed_indices: list[int] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return len(self.failed_indices)

    @property
    def failure_rate(self) -> float:
        return self.failed / self.total if self.total else 0.0

    def log(self) -> None:
        if not self.failed:
            logger.info("%s: %d/%d succeeded", self.stage, self.ok, self.total)
            return
        # Gaps used to be indistinguishable from a chunk that genuinely had
        # nothing to extract, because a failure just returned an empty list.
        logger.warning(
            "%s: %d/%d succeeded, %d FAILED (%.2f%%) — those chunks carry no "
            "results for this signal. First failures: %s. They are not cached, so "
            "re-running retries only them.",
            self.stage, self.ok, self.total, self.failed,
            100 * self.failure_rate, self.failed_indices[:10],
        )


def bounded_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    workers: int,
    in_flight: int | None = None,
    stage: str = "inference",
    on_progress: Callable[[int, int], None] | None = None,
    progress_every: int = 50,
) -> tuple[list[R | None], Outcome]:
    """Apply `fn` across `items` with a bounded number of tasks in flight.

    Returns results in input order — a failed item yields None — plus an Outcome
    recording which indices failed. Never holds more than `in_flight` Futures, so
    memory does not scale with corpus size.
    """
    n = len(items)
    outcome = Outcome(stage=stage, total=n)
    results: list[R | None] = [None] * n
    if n == 0:
        return results, outcome

    workers = max(1, workers)
    limit = max(workers, in_flight or workers * 2)
    done = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: dict = {}
        nxt = 0
        while nxt < n or pending:
            while nxt < n and len(pending) < limit:
                pending[pool.submit(fn, items[nxt])] = nxt
                nxt += 1
            finished, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for fut in finished:
                i = pending.pop(fut)
                try:
                    results[i] = fut.result()
                    outcome.ok += 1
                except Exception as exc:  # noqa: BLE001 — one item must not kill the stage
                    outcome.failed_indices.append(i)
                    logger.warning("%s: item %d failed: %s", stage, i, exc)
                with lock:
                    done += 1
                    if on_progress and done % progress_every == 0:
                        on_progress(done, n)

    outcome.failed_indices.sort()
    outcome.log()
    return results, outcome


def bounded_imap_batches(
    fn: Callable[[int, Sequence[T]], R],
    items: Sequence[T],
    *,
    batch_size: int,
    workers: int,
    in_flight: int | None = None,
) -> Iterable[tuple[int, R]]:
    """Yield (start_offset, result) for `fn` over batches, bounded and concurrent.

    Completion order is arbitrary; the offset tells the caller where the result
    belongs, so writing into a preallocated array stays correct without any
    reordering step.
    """
    n = len(items)
    if n == 0:
        return
    workers = max(1, workers)
    limit = max(workers, in_flight or workers * 2)
    starts = list(range(0, n, batch_size))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: dict = {}
        nxt = 0
        while nxt < len(starts) or pending:
            while nxt < len(starts) and len(pending) < limit:
                s = starts[nxt]
                pending[pool.submit(fn, s, items[s : s + batch_size])] = s
                nxt += 1
            finished, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for fut in finished:
                s = pending.pop(fut)
                yield s, fut.result()      # exceptions propagate: a lost batch
                                           # would misalign vectors against chunks
