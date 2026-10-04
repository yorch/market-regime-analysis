"""
Scheduled watchlist scanner with regime-change alerts.

Analyze a watchlist on a schedule, save every result to the regime history
store, detect regime changes and send alerts::

    from mra_lib.scanner import AlertPolicy, LogNotifier, Scanner, notifiers_from_env
    from mra_lib.storage import default_store

    scanner = Scanner(
        ["SPY", "QQQ"],
        store=default_store(),
        provider="mock",
        notifiers=[LogNotifier(), *notifiers_from_env()],
        policy=AlertPolicy(timeframes_to_watch=("1D",), min_confidence=0.6),
    )
    report = scanner.scan_once()          # one pass over the watchlist
    summary = scanner.run(interval=3600)  # until SIGINT/SIGTERM

Modules:

- :mod:`~mra_lib.scanner.detection`: :func:`detect_change` (pure) and
  :class:`AlertPolicy`.
- :mod:`~mra_lib.scanner.events`: :class:`RegimeChangeEvent`.
- :mod:`~mra_lib.scanner.notifiers`: the :class:`Notifier` protocol, the log,
  webhook, Telegram and Discord channels, and :func:`notifiers_from_env`.
- :mod:`~mra_lib.scanner.scanner`: :class:`Scanner` and its reports.
"""

from .detection import (
    DEFAULT_COOLDOWN_BARS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_WATCH_TIMEFRAMES,
    AlertPolicy,
    ChangeDecision,
    ChangeKind,
    SuppressReason,
    detect_change,
    in_cooldown,
    previous_run,
)
from .events import RegimeChangeEvent
from .notifiers import (
    ALERT_SECRET_ENV_VARS,
    DiscordNotifier,
    LogNotifier,
    Notifier,
    TelegramNotifier,
    WebhookNotifier,
    mask_secrets,
    notifiers_from_env,
    validate_notifier_url,
)

__all__ = [
    "ALERT_SECRET_ENV_VARS",
    "DEFAULT_COOLDOWN_BARS",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_WATCH_TIMEFRAMES",
    "AlertPolicy",
    "ChangeDecision",
    "ChangeKind",
    "DiscordNotifier",
    "LogNotifier",
    "Notifier",
    "RegimeChangeEvent",
    "SuppressReason",
    "TelegramNotifier",
    "WebhookNotifier",
    "detect_change",
    "in_cooldown",
    "mask_secrets",
    "notifiers_from_env",
    "previous_run",
    "validate_notifier_url",
]
