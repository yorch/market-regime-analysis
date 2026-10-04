"""Tests for the shared periodic loop (mra_lib.scheduling); no real sleeping."""

import signal

import pytest

from mra_lib.scheduling import (
    StopReason,
    StopToken,
    backoff_delay,
    interruptible_sleep,
    run_periodic,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


def test_backoff_delay_grows_and_caps():
    assert [backoff_delay(n, 10, 25) for n in (1, 2, 3, 4)] == [10, 20, 25, 25]
    assert backoff_delay(1, 100, 25) == 100  # never below the interval
    assert backoff_delay(10_000, 1, 3600) == 3600  # no overflow


def test_cadence_skips_missed_ticks():
    clock = Clock()

    def step(i):
        clock.now += 25 if i == 1 else 1  # first iteration overruns two ticks

    result = run_periodic(
        step, 10, max_iterations=3, sleep=clock.sleep, monotonic=clock.monotonic, signals=()
    )
    assert result.successes == 3
    assert clock.sleeps == [5, 9]  # next grid points: 30, then 40


def test_failures_counted_and_backed_off():
    clock = Clock()
    outcomes = iter([False, ValueError("boom"), True])

    def step(i):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    result = run_periodic(
        step, 10, max_iterations=3, sleep=clock.sleep, monotonic=clock.monotonic, signals=()
    )
    assert (result.iterations, result.successes, result.failures) == (3, 1, 2)
    assert clock.sleeps == [10, 20]
    assert result.stop_reason is StopReason.MAX_ITERATIONS


def test_stop_token_ends_loop():
    stop = StopToken()

    def step(i):
        if i == 2:
            stop.request()

    result = run_periodic(step, 1, stop=stop, sleep=lambda d: None, signals=())
    assert result.iterations == 2
    assert result.stop_reason is StopReason.STOP_REQUESTED


def test_sigterm_stops_and_handler_restored():
    before = signal.getsignal(signal.SIGTERM)
    result = run_periodic(
        lambda i: signal.raise_signal(signal.SIGTERM),
        1,
        sleep=lambda d: pytest.fail("must not sleep after a stop request"),
    )
    assert result.iterations == 1 and result.stop_reason is StopReason.STOP_REQUESTED
    assert signal.getsignal(signal.SIGTERM) == before


def test_keyboard_interrupt_is_clean():
    def step(i):
        raise KeyboardInterrupt

    result = run_periodic(step, 1, signals=())
    assert result.stop_reason is StopReason.INTERRUPTED


def test_invalid_arguments():
    with pytest.raises(ValueError, match="interval"):
        run_periodic(lambda i: None, 0)
    with pytest.raises(ValueError, match="max_iterations"):
        run_periodic(lambda i: None, 1, max_iterations=0)


def test_interruptible_sleep_slices_and_honors_stop():
    clock = Clock()
    stop = StopToken()
    interruptible_sleep(2.5, stop, monotonic=clock.monotonic, sleep=clock.sleep)
    assert clock.sleeps == [1.0, 1.0, 0.5]

    def sleep_then_stop(d):
        clock.sleep(d)
        stop.request()

    clock.sleeps.clear()
    interruptible_sleep(10, stop, monotonic=clock.monotonic, sleep=sleep_then_stop)
    assert clock.sleeps == [1.0]
