"""Admission control for query serving.

Every earlier fix bounded what ONE query costs — the corpus scan, the pair
dictionary, the vector duplication, the cold-start builds. None of them bound
the total: the server admitted every request that arrived, so N concurrent
queries meant N times the per-query memory with no ceiling. Overload therefore
showed up as memory growth until the process died, rather than as rejected
requests, which is the worst shape for it to take because nothing upstream can
react to it.

Three bounds here, and they are deliberately about *capacity*, not latency:

  in-flight   how many queries may execute at once
  queue       how many may wait for a slot; past that, reject promptly
  deadline    how long a request may wait before it is not worth serving

A queue that cannot reject is only a slower way to exhaust memory, so the
rejection is the point. Waiting is bounded so a burst is absorbed, and anything
beyond that gets 503 with Retry-After while capacity is still intact.

Native thread pools are capped separately. faiss and BLAS default to one thread
per core each, so ten concurrent queries on a ten-core box can ask for a hundred
threads; they then contend rather than parallelise, and the builder's threads
come on top.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Overloaded(Exception):
    """Raised when a request cannot be admitted within its bounds."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class AdmissionStats:
    admitted: int = 0
    queued: int = 0
    rejected_queue_full: int = 0
    rejected_timeout: int = 0
    peak_in_flight: int = 0
    peak_waiting: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class AdmissionController:
    """Bounds concurrent queries, with a bounded wait and a wait deadline."""

    def __init__(self, max_in_flight: int, max_waiting: int,
                 wait_timeout_s: float) -> None:
        self.max_in_flight = max(1, max_in_flight)
        self.max_waiting = max(0, max_waiting)
        self.wait_timeout_s = max(0.0, wait_timeout_s)
        self._sem = threading.BoundedSemaphore(self.max_in_flight)
        self._lock = threading.Lock()
        self._in_flight = 0
        self._waiting = 0
        self.stats = AdmissionStats()

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def waiting(self) -> int:
        with self._lock:
            return self._waiting

    @contextmanager
    def admit(self):
        """Occupy a slot for the duration of a request.

        Rejects rather than queueing without limit: a request that would have to
        wait behind more than `max_waiting` others is refused while the server is
        still healthy, which is information the caller can act on. A slot is
        always released, including when the handler raises.
        """
        with self._lock:
            if self._waiting >= self.max_waiting and self._in_flight >= self.max_in_flight:
                self.stats.rejected_queue_full += 1
                raise Overloaded(
                    f"Server is at capacity ({self.max_in_flight} in flight, "
                    f"{self._waiting} waiting). Retry shortly.",
                    retry_after=max(1.0, self.wait_timeout_s),
                )
            self._waiting += 1
            self.stats.peak_waiting = max(self.stats.peak_waiting, self._waiting)
            if self._in_flight >= self.max_in_flight:
                self.stats.queued += 1

        acquired = self._sem.acquire(timeout=self.wait_timeout_s or None)
        with self._lock:
            self._waiting -= 1
            if not acquired:
                self.stats.rejected_timeout += 1
                raise Overloaded(
                    f"Waited {self.wait_timeout_s:.1f}s for a slot without one "
                    f"becoming free. Retry shortly.",
                    retry_after=max(1.0, self.wait_timeout_s),
                )
            self._in_flight += 1
            self.stats.admitted += 1
            self.stats.peak_in_flight = max(self.stats.peak_in_flight, self._in_flight)

        try:
            yield
        finally:
            with self._lock:
                self._in_flight -= 1
            self._sem.release()


def cap_native_threads(threads_per_process: int) -> dict[str, int | None]:
    """Cap the thread pools that would otherwise scale per concurrent request.

    faiss and the BLAS libraries each default to one thread per core, so several
    concurrent queries on a many-core box request far more threads than there are
    cores. Past that point they contend rather than parallelise, and the indexing
    thread's own pools come on top. Called once at startup, before the libraries
    size their pools.
    """
    n = max(1, threads_per_process)
    applied: dict[str, int | None] = {}
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n))
        applied[var] = int(os.environ[var])
    try:
        import faiss
        faiss.omp_set_num_threads(n)
        applied["faiss"] = n
    except Exception:  # noqa: BLE001 — capping is best-effort, never fatal
        applied["faiss"] = None
    logger.info("Native thread pools capped at %d per process", n)
    return applied
