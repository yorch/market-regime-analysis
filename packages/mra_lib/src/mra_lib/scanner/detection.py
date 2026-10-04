"""
Regime-change detection and the alert policy.

:func:`detect_change` is pure: given the previously stored record, the new
record, the multi-timeframe confirmation and a slice of stored history, it
returns a :class:`ChangeDecision`. No I/O, no clock, no logging.

Detection rules
---------------
1. **Baseline.** No previous record for (symbol, timeframe): the new record is
   the baseline. Never an alert.
2. **Same bar.** The new record is for a bar that is already stored (a re-scan
   before the next bar closed). It is compared with the newest record *before*
   that bar, and is a change only if its regime differs from that record *and*
   from the stored record for the same bar (``replaced``). So re-scanning never
   repeats an alert, but a regime that flips while the bar is still forming
   (e.g. an hourly scan of a daily bar) is detected once instead of being lost
   when the re-scan overwrites the stored record. **Stale bar**: an older
   ``bar_time`` than the newest stored record (e.g. a provider lagging behind):
   not a change, and the scanner does not save it.
3. **Unchanged.** Newer bar, same regime: not a change.
4. **Changed.** Newer bar, different regime: a change. It becomes an alert only
   if the :class:`AlertPolicy` allows it, checked in this order (the first
   failing check is reported as the :class:`SuppressReason`):

   - the timeframe is watched (``timeframes_to_watch``, default ``1D``);
   - the new regime is classified (not ``Unknown``);
   - the new regime's confidence is at least ``min_confidence`` (default 0.6);
   - with ``require_confirmation`` (default on): the multi-timeframe
     confirmation is ``confirmed`` *and* its direction equals the new regime's
     bias (so transitions into neutral regimes only alert with confirmation off);
   - the (symbol, timeframe) is not in cooldown.

Cooldown (anti-flapping)
------------------------
Cooldown is derived from the stored history, not kept in memory, so a restart
makes exactly the same decisions. The previous regime's *run* is the stretch of
consecutive stored bars, ending at the previous record, that carry the previous
regime. If the stored history shows the transition into that run (an older
record with a different regime), the change is suppressed when the run is
shorter than the cooldown:

- ``cooldown_bars`` (default 1): suppressed if the run has ``<= cooldown_bars``
  bars. With the default, a regime that lasted a single bar is treated as noise:
  ``A A A B`` alerts, but the immediate flip back ``A A A B A`` does not, and
  neither does ``... B C`` when ``B`` lasted one bar. ``0`` disables it.
- ``cooldown`` (a :class:`~datetime.timedelta`, default off): suppressed if the
  run started less than ``cooldown`` before the new bar.

A run whose start is not visible in the stored history (the scanner's very first
observations, or a run longer than the history slice read) is not in cooldown.
Bars here are stored bars (one per distinct ``bar_time``), so bars the scanner
never saw (downtime) do not count.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from mra_lib.config.enums import DirectionalBias, MarketRegime
from mra_lib.config.regime_tables import REGIME_BIAS
from mra_lib.config.timeframes import TIMEFRAMES
from mra_lib.errors import InvalidParametersError
from mra_lib.signals.confirmation import TimeframeConfirmation
from mra_lib.storage import MAX_HISTORY_LIMIT, RegimeRecord

#: Timeframes alerted on by default (the confirmation's primary timeframe).
DEFAULT_WATCH_TIMEFRAMES: tuple[str, ...] = ("1D",)
#: Default minimum regime confidence for an alert.
DEFAULT_MIN_CONFIDENCE = 0.6
#: Default cooldown in stored bars.
DEFAULT_COOLDOWN_BARS = 1


class ChangeKind(StrEnum):
    """How a new record relates to the previously stored one."""

    BASELINE = "baseline"
    """No previous record: first observation, recorded as the baseline."""

    SAME_BAR = "same_bar"
    """Same ``bar_time`` as the previous record (a re-scan)."""

    STALE_BAR = "stale_bar"
    """Older ``bar_time`` than the previous record."""

    UNCHANGED = "unchanged"
    """Newer bar, same regime."""

    CHANGED = "changed"
    """Newer bar, different regime."""


class SuppressReason(StrEnum):
    """Why a detected change did not become an alert (first failing check)."""

    UNWATCHED = "unwatched"
    UNKNOWN_REGIME = "unknown_regime"
    LOW_CONFIDENCE = "low_confidence"
    NOT_CONFIRMED = "not_confirmed"
    DIRECTION_MISMATCH = "direction_mismatch"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class AlertPolicy:
    """Which regime changes become alerts.

    Attributes:
        timeframes_to_watch: Timeframes whose changes may alert (default ``("1D",)``).
        require_confirmation: Only alert when the multi-timeframe confirmation is
            ``confirmed`` and its direction matches the new regime's bias.
        min_confidence: Minimum regime confidence (0-1) of the new record.
        cooldown_bars: Suppress a change if the previous regime lasted at most this
            many stored bars after a visible transition (``0`` disables).
        cooldown: Suppress a change if the previous regime started less than this
            long before the new bar (``None`` disables). At most
            ``MAX_HISTORY_LIMIT`` (1000) stored bars are read to find the run's
            start, so on intraday timeframes a window longer than 1000 bars is
            effectively capped.

    Raises:
        InvalidParametersError: On an unknown timeframe or an out-of-range value.
    """

    timeframes_to_watch: tuple[str, ...] = DEFAULT_WATCH_TIMEFRAMES
    require_confirmation: bool = True
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    cooldown_bars: int = DEFAULT_COOLDOWN_BARS
    cooldown: timedelta | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the fields."""
        watch = (
            (self.timeframes_to_watch,)
            if isinstance(self.timeframes_to_watch, str)
            else tuple(dict.fromkeys(self.timeframes_to_watch))
        )
        unknown = [tf for tf in watch if tf not in TIMEFRAMES]
        if not watch or unknown:
            raise InvalidParametersError(
                f"timeframes_to_watch must be a non-empty subset of {list(TIMEFRAMES)}, "
                f"got {list(watch)}"
            )
        object.__setattr__(self, "timeframes_to_watch", watch)
        conf = self.min_confidence
        if (
            isinstance(conf, bool)
            or not isinstance(conf, int | float)
            or not math.isfinite(conf)
            or not 0 <= conf <= 1
        ):
            raise InvalidParametersError(f"min_confidence must be in [0, 1], got {conf!r}")
        bars = self.cooldown_bars
        if isinstance(bars, bool) or not isinstance(bars, int) or bars < 0:
            raise InvalidParametersError(f"cooldown_bars must be an integer >= 0, got {bars!r}")
        if self.cooldown is not None and (
            not isinstance(self.cooldown, timedelta) or self.cooldown < timedelta(0)
        ):
            raise InvalidParametersError("cooldown must be a non-negative timedelta or None")

    @property
    def history_lookback(self) -> int:
        """Stored records (ending at the previous record) needed to evaluate the cooldown."""
        if self.cooldown:
            return MAX_HISTORY_LIMIT
        return self.cooldown_bars + 1

    @property
    def uses_cooldown(self) -> bool:
        """Whether any cooldown is configured."""
        return self.cooldown_bars > 0 or bool(self.cooldown)


@dataclass(frozen=True)
class ChangeDecision:
    """Result of :func:`detect_change`.

    Attributes:
        kind: How the new record relates to the previous one.
        alert: Whether the change should be sent to the notifiers.
        suppressed: For a ``CHANGED`` record that does not alert, the reason.
        previous_regime: The previously stored regime (``None`` for a baseline).
        new_regime: The new record's regime.
    """

    kind: ChangeKind
    alert: bool
    suppressed: SuppressReason | None
    previous_regime: str | None
    new_regime: str

    @property
    def changed(self) -> bool:
        """Whether the regime changed on a newer bar (alerted or not)."""
        return self.kind is ChangeKind.CHANGED


def regime_bias(regime: str) -> DirectionalBias:
    """Return the directional bias of a stored regime value (NEUTRAL if unknown)."""
    try:
        return REGIME_BIAS.get(MarketRegime(regime), DirectionalBias.NEUTRAL)
    except ValueError:
        return DirectionalBias.NEUTRAL


def previous_run(
    previous: RegimeRecord, history: Iterable[RegimeRecord]
) -> tuple[int, datetime | None]:
    """Measure the stored run of ``previous.regime`` that ends at ``previous``.

    Args:
        previous: The most recent stored record before the new one.
        history: Stored records for the same (symbol, timeframe), any order.
            Records newer than ``previous`` are ignored; ``previous`` itself is
            added if missing.

    Returns:
        ``(run_length, run_start)``: the number of consecutive stored bars (newest
        first, starting at ``previous``) with the previous regime, and the
        ``bar_time`` of the run's first bar if the transition into the run is
        visible (an older record with a different regime), else ``None``.
    """
    by_bar = {
        r.bar_time: r
        for r in history
        if r.symbol == previous.symbol
        and r.timeframe == previous.timeframe
        and r.bar_time <= previous.bar_time
    }
    by_bar[previous.bar_time] = previous
    run = 0
    start = previous.bar_time
    for bar_time in sorted(by_bar, reverse=True):
        if by_bar[bar_time].regime != previous.regime:
            return run, start
        run += 1
        start = bar_time
    return run, None


def in_cooldown(
    previous: RegimeRecord,
    current: RegimeRecord,
    history: Iterable[RegimeRecord],
    policy: AlertPolicy,
) -> bool:
    """Whether a change from ``previous`` to ``current`` falls in the cooldown.

    See the module docstring for the rules.
    """
    if not policy.uses_cooldown:
        return False
    run, start = previous_run(previous, history)
    if start is None:
        return False
    if policy.cooldown_bars and run <= policy.cooldown_bars:
        return True
    window = policy.cooldown
    return window is not None and window > timedelta(0) and current.bar_time - start < window


def detect_change(  # noqa: PLR0911, PLR0912, PLR0913 - one early return per rule, in order
    previous: RegimeRecord | None,
    current: RegimeRecord,
    *,
    policy: AlertPolicy | None = None,
    confirmation: TimeframeConfirmation | None = None,
    history: Sequence[RegimeRecord] = (),
    replaced: RegimeRecord | None = None,
) -> ChangeDecision:
    """Classify ``current`` against ``previous`` and decide whether to alert.

    Pure function; see the module docstring for the rules.

    Args:
        previous: The stored record with the newest ``bar_time`` for the same
            (symbol, timeframe), read *before* ``current`` was saved; ``None`` if
            there was none.
        current: The new record.
        policy: Alert policy (default :class:`AlertPolicy()`).
        confirmation: Multi-timeframe confirmation computed in the same scan
            (needed when ``policy.require_confirmation``; ``None`` never confirms).
        history: Stored records for the same (symbol, timeframe) up to
            ``previous`` (e.g. ``store.history(symbol, timeframe,
            until=previous.bar_time, limit=policy.history_lookback)``), used for
            the cooldown. Empty means no visible transition (no cooldown).
        replaced: For a re-scan of an already stored bar: the stored record for
            ``current.bar_time`` (which ``current`` replaces). ``previous`` must
            then be the newest record *older* than that bar. A re-scan is a
            change only if ``current`` differs from both ``previous`` and
            ``replaced``, so a regime that flips while a bar is still forming is
            detected exactly once and never lost to the upsert.

    Returns:
        The decision.

    Raises:
        InvalidParametersError: If ``previous`` and ``current`` are for different
            (symbol, timeframe) pairs.
    """
    policy = policy if policy is not None else AlertPolicy()
    for other in (previous, replaced):
        if other is not None and (other.symbol, other.timeframe) != (
            current.symbol,
            current.timeframe,
        ):
            raise InvalidParametersError(
                "previous, replaced and current records must have the same symbol and timeframe"
            )
    if replaced is not None and replaced.bar_time != current.bar_time:
        raise InvalidParametersError("replaced must be the stored record for current.bar_time")
    if previous is None:
        kind = ChangeKind.BASELINE if replaced is None else ChangeKind.SAME_BAR
        return ChangeDecision(kind, False, None, None, current.regime)

    def decision(kind: ChangeKind, suppressed: SuppressReason | None = None) -> ChangeDecision:
        alert = kind is ChangeKind.CHANGED and suppressed is None
        return ChangeDecision(kind, alert, suppressed, previous.regime, current.regime)

    if current.bar_time == previous.bar_time:
        return decision(ChangeKind.SAME_BAR)
    if current.bar_time < previous.bar_time:
        return decision(ChangeKind.STALE_BAR)
    if replaced is not None and replaced.regime == current.regime:
        return decision(ChangeKind.SAME_BAR)  # this bar's regime was already evaluated
    if current.regime == previous.regime:
        return decision(ChangeKind.UNCHANGED if replaced is None else ChangeKind.SAME_BAR)

    changed = ChangeKind.CHANGED
    if current.timeframe not in policy.timeframes_to_watch:
        return decision(changed, SuppressReason.UNWATCHED)
    if current.regime == MarketRegime.UNKNOWN.value:
        return decision(changed, SuppressReason.UNKNOWN_REGIME)
    if current.confidence < policy.min_confidence:
        return decision(changed, SuppressReason.LOW_CONFIDENCE)
    if policy.require_confirmation:
        if confirmation is None or not confirmation.confirmed:
            return decision(changed, SuppressReason.NOT_CONFIRMED)
        if confirmation.direction is not regime_bias(current.regime):
            return decision(changed, SuppressReason.DIRECTION_MISMATCH)
    if in_cooldown(previous, current, history, policy):
        return decision(changed, SuppressReason.COOLDOWN)
    return decision(changed)


__all__ = [
    "DEFAULT_COOLDOWN_BARS",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_WATCH_TIMEFRAMES",
    "AlertPolicy",
    "ChangeDecision",
    "ChangeKind",
    "SuppressReason",
    "detect_change",
    "in_cooldown",
    "previous_run",
    "regime_bias",
]
