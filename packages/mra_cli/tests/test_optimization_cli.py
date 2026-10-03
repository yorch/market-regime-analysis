"""Tests for the mra-optimize script (mra_cli.optimization)."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from mra_cli import optimization


def _synthetic(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=pd.bdate_range("2018-01-01", periods=n),
    )


@pytest.fixture
def fast_settings(monkeypatch):
    monkeypatch.setitem(optimization.WF_SETTINGS, "n_hmm_states", 2)
    monkeypatch.setitem(optimization.WF_SETTINGS, "hmm_n_iter", 5)
    monkeypatch.setitem(optimization.WF_SETTINGS, "retrain_frequency", 63)
    monkeypatch.setattr(optimization, "load_data", lambda *a, **k: _synthetic())


def test_random_mode_writes_output_with_holdout(fast_settings, tmp_path, monkeypatch, capsys):
    out = tmp_path / "res.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mra-optimize",
            "--mode",
            "random",
            "--iterations",
            "2",
            "--seed",
            "3",
            "--quiet",
            "--output",
            str(out),
        ],
    )
    optimization.main()

    data = json.loads(out.read_text())
    assert data["mode"] == "random"
    assert data["seed"] == 3
    assert data["search"]["n_trials"] >= 1
    assert data["search"]["holdout_bars"] == 100
    assert data["holdout"] is not None
    assert "in_sample" in data
    printed = capsys.readouterr().out
    assert "HOLDOUT (OUT-OF-SAMPLE)" in printed
    assert "IN-SAMPLE" in printed


def test_load_data_requires_key_from_env(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(ValueError, match="POLYGON_API_KEY"):
        optimization.load_data("SPY", "polygon")


def test_load_data_passes_env_key(monkeypatch):
    captured = {}

    class FakeProvider:
        def fetch(self, symbol, period, interval):
            return _synthetic(10)

    def fake_create(name, **kwargs):
        captured.update(kwargs, name=name)
        return FakeProvider()

    monkeypatch.setenv("POLYGON_API_KEY", "k123")
    monkeypatch.setattr(optimization.MarketDataProvider, "create_provider", fake_create)
    df = optimization.load_data("SPY", "polygon")
    assert len(df) == 10
    assert captured == {"name": "polygon", "api_key": "k123"}
