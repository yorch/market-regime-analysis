"""Tests for MarketRegimeAnalyzer.run_continuous_monitoring (offline, mock provider)."""

import signal
from unittest.mock import patch

import pytest

from mra_lib import MarketRegimeAnalyzer


@pytest.fixture(scope="module")
def analyzer() -> MarketRegimeAnalyzer:
    return MarketRegimeAnalyzer("SPY", periods={"1D": "1y"}, provider_flag="mock")


def test_monitoring_survives_transient_failures(analyzer):
    calls = {"n": 0}
    real_load = analyzer._load_data

    def flaky_load():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient")
        real_load()

    updates: list[str] = []
    with patch.object(analyzer, "_load_data", side_effect=flaky_load):
        successes = analyzer.run_continuous_monitoring(
            0.01, max_iterations=3, on_update=lambda tf, a: updates.append(tf)
        )

    # Iteration 1 reuses constructor data, 2 fails, 3 recovers
    assert successes == 2
    assert updates == ["1D", "1D"]


def test_monitoring_returns_zero_when_every_iteration_fails(analyzer):
    def boom(tf, analysis):
        raise RuntimeError("render failed")

    assert analyzer.run_continuous_monitoring(0.01, max_iterations=2, on_update=boom) == 0


def test_monitoring_backoff_grows_and_caps(analyzer):
    waits: list[float] = []

    with (
        patch.object(analyzer, "_load_data", side_effect=ConnectionError("down")),
        patch.object(
            MarketRegimeAnalyzer, "_monitor_sleep", side_effect=lambda d, stop: waits.append(d)
        ),
    ):
        analyzer.run_continuous_monitoring(10, max_iterations=4, max_backoff=25)

    # Iteration 1 skips the reload (fresh constructor data) and succeeds
    assert waits[0] == pytest.approx(10, abs=0.5)
    assert waits[1:] == [10, 20]


def test_monitoring_restores_sigterm_handler(analyzer):
    before = signal.getsignal(signal.SIGTERM)
    analyzer.run_continuous_monitoring(0.01, max_iterations=1, on_update=lambda tf, a: None)
    assert signal.getsignal(signal.SIGTERM) == before


def test_monitoring_stops_on_sigterm(analyzer):
    def send_sigterm(tf, analysis):
        signal.raise_signal(signal.SIGTERM)

    # Without the handler the test process would be terminated
    successes = analyzer.run_continuous_monitoring(0.01, on_update=send_sigterm)
    assert successes == 1
