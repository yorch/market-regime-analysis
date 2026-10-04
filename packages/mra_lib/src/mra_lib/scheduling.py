"""
Periodic loop with a monotonic cadence, exponential backoff and clean shutdown.

Shared by :meth:`MarketRegimeAnalyzer.run_continuous_monitoring
<mra_lib.analyzer.MarketRegimeAnalyzer.run_continuous_monitoring>` and
:meth:`mra_lib.scanner.Scanner.run`::

    from mra_lib.scheduling import run_periodic

    result = run_periodic(do_work, interval=300, max_iterations=10)

Rules (see :func:`run_periodic`):

- **Cadence.** Successful iterations are scheduled on a fixed monotonic grid
  (``start``, ``start + interval``, ...), so slow iterations do not make the
  schedule drift; ticks missed by a very slow iteration are skipped, not
  replayed.
- **Isolation.** An iteration that raises (or returns ``False``) is logged and
  counted as a failure; the loop keeps going.
- **Backoff.** After ``n`` consecutive failures the next attempt waits
  ``interval * 2 ** (n - 1)``, capped at ``max(max_backoff, interval)``.
- **Stop.** The configured signals (SIGTERM by default) set a stop flag that is
  checked between iterations and while sleeping; a second SIGINT, or SIGINT when
  it is not handled, raises ``KeyboardInterrupt``, which also ends the loop
  cleanly. Previous signal handlers are restored on exit.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

#: Default upper bound in seconds for the retry delay after repeated failures.
DEFAULT_MAX_BACKOFF = 3600.0

#: Longest single sleep slice; bounds how long a stop request can go unnoticed.
_SLEEP_SLICE_SECONDS = 1.0


class StopToken:
    """A stop flag that is safe to set from a signal handler.

    A plain attribute, not a :class:`threading.Event`: setting an Event from a
    signal handler can deadlock if the signal lands while the main thread holds
    the Event's internal lock.
    """

    def __init__(self) -> None:
        self.requested = False

    def request(self) -> None:
        """Ask the loop to stop after the current unit of work."""
        self.requested = True

    def __bool__(self) -> bool:
        return self.requested


class StopReason(StrEnum):
    """Why :func:`run_periodic` returned."""

    MAX_ITERATIONS = "max_iterations"
    """The configured number of iterations ran."""

    STOP_REQUESTED = "stop_requested"
    """A signal (or the caller) set the stop flag."""

    INTERRUPTED = "interrupted"
    """``KeyboardInterrupt`` (Ctrl+C) ended the loop."""


@dataclass(frozen=True)
class LoopResult:
    """Outcome of :func:`run_periodic`.

    Attributes:
        iterations: Iterations started (including failed ones).
        successes: Iterations that completed and reported success.
        failures: Iterations that raised or reported failure.
        stop_reason: Why the loop ended.
    """

    iterations: int
    successes: int
    failures: int
    stop_reason: StopReason


def backoff_delay(failures: int, interval: float, max_backoff: float) -> float:
    """Return the retry delay after ``failures`` consecutive failures.

    Args:
        failures: Consecutive failures so far (``>= 1``).
        interval: The loop interval in seconds (the first retry delay).
        max_backoff: Upper bound in seconds; never below ``interval``.

    Returns:
        ``interval * 2 ** (failures - 1)``, capped at ``max(max_backoff, interval)``.
    """
    exponent = max(failures, 1) - 1
    cap = max(max_backoff, interval)
    # Avoid float overflow for very long failure streaks
    if exponent >= 64:  # noqa: PLR2004 - 2**64 * interval is far beyond any sane cap
        return float(cap)
    return float(min(cap, interval * 2**exponent))


def interruptible_sleep(
    delay: float,
    stop: StopToken,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Sleep ``delay`` seconds in short slices so a stop request is honored promptly.

    Args:
        delay: Seconds to sleep.
        stop: Returns early once ``stop.requested`` is set.
        monotonic: Clock (injectable for tests).
        sleep: Sleep function (injectable for tests).
    """
    deadline = monotonic() + delay
    while not stop.requested:
        remaining = deadline - monotonic()
        if remaining <= 0:
            return
        sleep(min(_SLEEP_SLICE_SECONDS, remaining))


def _next_tick(next_tick: float, interval: float, now: float) -> tuple[float, float]:
    """Advance the cadence grid past ``now``; return ``(next_tick, delay)``.

    Ticks missed by a slow iteration are skipped, not replayed.
    """
    next_tick += interval
    while next_tick <= now:
        next_tick += interval
    return next_tick, next_tick - now


@contextmanager
def stop_on_signals(
    signals: Iterable[signal.Signals],
    stop: StopToken,
    *,
    name: str = "Loop",
    log: logging.Logger | None = None,
) -> Iterator[None]:
    """Make ``signals`` set ``stop`` while the block runs; restore the old handlers after.

    Handlers are only installed from the main thread (Python only delivers
    signals there). A second ``SIGINT`` after a stop request raises
    ``KeyboardInterrupt``, so a stuck iteration can still be aborted.

    Args:
        signals: Signals to handle.
        stop: The flag to set.
        name: Label for the log message.
        log: Logger (default: this module's logger).
    """
    log = log or logger
    previous: dict[signal.Signals, Any] = {}
    if threading.current_thread() is threading.main_thread():

        def _on_signal(signum: int, frame: Any) -> None:
            _ = frame
            if signum == signal.SIGINT and stop.requested:
                raise KeyboardInterrupt
            log.info("%s received, stopping %s", signal.Signals(signum).name, name.lower())
            stop.request()

        for sig in signals:
            previous[sig] = signal.signal(sig, _on_signal)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            if handler is not None:
                signal.signal(sig, handler)


def run_periodic(  # noqa: PLR0913 - keyword-only knobs with defaults
    step: Callable[[int], bool | None],
    interval: float,
    *,
    max_iterations: int | None = None,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
    stop: StopToken | None = None,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    signals: Iterable[signal.Signals] = (signal.SIGTERM,),
    name: str = "Loop",
    log: logging.Logger | None = None,
) -> LoopResult:
    """Call ``step`` every ``interval`` seconds until stopped.

    Args:
        step: Called with the 1-based iteration number. Returning ``False`` or
            raising an ``Exception`` counts as a failure (logged, then retried
            with backoff); returning ``True`` or ``None`` counts as a success.
        interval: Seconds between successful iterations (``> 0``).
        max_iterations: Stop after this many iterations (``None`` = until stopped).
        max_backoff: Upper bound in seconds for the retry delay after failures.
        stop: Stop flag; created if not given. Set it from ``step`` (or another
            thread) to end the loop after the current iteration.
        sleep: ``sleep(seconds)`` between iterations. Defaults to
            :func:`interruptible_sleep` on ``stop``.
        monotonic: Clock used for the cadence (injectable for tests).
        signals: Signals that request a stop. Handlers are only installed from
            the main thread and are restored on exit. With ``SIGINT`` included,
            the first Ctrl+C stops after the current iteration and a second one
            raises ``KeyboardInterrupt``.
        name: Label used in log messages (e.g. ``"Monitoring"``).
        log: Logger to use (default: this module's logger).

    Returns:
        The iteration counts and why the loop ended.

    Raises:
        ValueError: If ``interval`` is not positive or ``max_iterations`` < 1.
    """
    if not interval > 0:
        raise ValueError(f"interval must be positive, got {interval!r}")
    if max_iterations is not None and max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations!r}")
    log = log or logger
    stop = stop if stop is not None else StopToken()
    if sleep is None:
        token = stop

        def sleep(delay: float) -> None:
            interruptible_sleep(delay, token, monotonic=monotonic)

    iterations = successes = failures = streak = 0
    reason = StopReason.STOP_REQUESTED
    next_tick = monotonic()
    with stop_on_signals(signals, stop, name=name, log=log):
        try:
            while not stop.requested:
                iterations += 1
                try:
                    ok = step(iterations) is not False
                    error = None if ok else "iteration reported failure"
                except Exception as e:  # noqa: BLE001 - resilient loop: log, back off, retry
                    ok, error = False, str(e) or type(e).__name__

                if ok:
                    successes += 1
                    streak = 0
                    next_tick, delay = _next_tick(next_tick, interval, monotonic())
                else:
                    failures += 1
                    streak += 1
                    delay = backoff_delay(streak, interval, max_backoff)
                    log.error(
                        "%s iteration %d failed (%d in a row): %s; retrying in %.0fs",
                        name,
                        iterations,
                        streak,
                        error,
                        delay,
                    )
                    next_tick = monotonic() + delay

                if max_iterations is not None and iterations >= max_iterations:
                    reason = StopReason.MAX_ITERATIONS
                    break
                if stop.requested:
                    break
                sleep(delay)
        except KeyboardInterrupt:
            log.info("%s stopped by user", name)
            reason = StopReason.INTERRUPTED

    return LoopResult(
        iterations=iterations, successes=successes, failures=failures, stop_reason=reason
    )


__all__ = [
    "DEFAULT_MAX_BACKOFF",
    "LoopResult",
    "StopReason",
    "StopToken",
    "backoff_delay",
    "interruptible_sleep",
    "run_periodic",
    "stop_on_signals",
]
