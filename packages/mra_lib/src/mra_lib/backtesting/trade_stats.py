"""
Shared trade-level statistics.

Single implementation of win rate / average win / average loss / profit factor
used by :class:`PerformanceMetrics`, :class:`WalkForwardValidator` and the
regime multiplier calibrator, so the three never disagree.

Profit factor convention
------------------------
``profit_factor = gross_wins / gross_losses`` where ``gross_losses`` is the
absolute sum of losing trade P&L.

* ``gross_losses > 0``: ordinary ratio.
* ``gross_losses == 0`` and ``gross_wins > 0``: ``math.inf`` (no losing trades).
* ``gross_losses == 0`` and ``gross_wins == 0`` (no trades, or only breakeven
  trades): ``math.nan`` (undefined).

Consumers that need a finite number (ranking, JSON output) should use
:func:`finite_profit_factor`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

#: Cap applied by :func:`finite_profit_factor` to an infinite profit factor.
PROFIT_FACTOR_CAP = 100.0


def compute_trade_stats(trades: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """
    Compute trade statistics from a list of trade dicts (each with a ``pnl`` key).

    Args:
        trades: Iterable of trade dictionaries containing net ``pnl``.

    Returns:
        Dictionary with ``total_trades``, ``winning_trades``, ``losing_trades``,
        ``win_rate``, ``avg_win`` (positive), ``avg_loss`` (positive magnitude),
        ``avg_trade``, ``expectancy``, ``total_wins``, ``total_losses`` (positive
        magnitude) and ``profit_factor`` (see module docstring for sentinels).
    """
    pnls = [float(t["pnl"]) for t in trades]
    return compute_pnl_stats(pnls)


def compute_pnl_stats(pnls: list[float]) -> dict[str, float | int]:
    """Compute trade statistics from a plain list of per-trade net P&L values."""
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_wins = float(sum(wins))
    total_losses = float(abs(sum(losses)))

    win_rate = len(wins) / n if n else 0.0
    avg_win = total_wins / len(wins) if wins else 0.0
    avg_loss = total_losses / len(losses) if losses else 0.0
    avg_trade = float(sum(pnls)) / n if n else 0.0

    if total_losses > 0:
        profit_factor = total_wins / total_losses
    elif total_wins > 0:
        profit_factor = math.inf
    else:
        profit_factor = math.nan

    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    return {
        "total_trades": n,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_trade": avg_trade,
        "expectancy": expectancy,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "profit_factor": profit_factor,
    }


def finite_profit_factor(profit_factor: float, cap: float = PROFIT_FACTOR_CAP) -> float:
    """
    Map a profit factor onto a finite value for ranking or serialization.

    ``nan`` (undefined) becomes ``0.0`` and ``inf`` (no losers) becomes ``cap``.
    """
    if math.isnan(profit_factor):
        return 0.0
    return min(profit_factor, cap)
