"""
Alert notifiers: deliver :class:`~mra_lib.scanner.events.RegimeChangeEvent` objects.

Every notifier implements the :class:`Notifier` protocol (``send(event)``) and
raises :class:`~mra_lib.errors.NotifierError` when delivery fails; the scanner
catches, logs and counts those failures, so a broken channel never stops a scan.

Channels:

- :class:`LogNotifier`: logs the message (always available; used by dry runs).
- :class:`WebhookNotifier`: POSTs :meth:`RegimeChangeEvent.to_dict` as JSON.
- :class:`TelegramNotifier`: Bot API ``sendMessage`` (plain text).
- :class:`DiscordNotifier`: Discord webhook (``content`` + one embed).

:func:`notifiers_from_env` builds the HTTP channels from ``ALERT_WEBHOOK_URL``,
``TELEGRAM_BOT_TOKEN`` + ``TELEGRAM_CHAT_ID`` and ``DISCORD_WEBHOOK_URL``.

Secret handling: webhook URLs and the Telegram bot token are credentials (the
token is part of the Bot API URL path). They are never logged, never put in
exception messages (errors carry only the channel name, HTTP status, and a
masked, truncated response excerpt), and never shown by ``repr``. Each
notifier registers its secrets with :func:`register_secret`, and
:func:`mask_secrets` replaces every registered value; the scanner applies it to
any error text it logs. ``mra_web`` additionally scrubs the env values listed
in :data:`ALERT_SECRET_ENV_VARS` from server logs.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

import requests

from mra_lib.data_providers._http import redact
from mra_lib.errors import NotifierConfigError, NotifierError

from .detection import regime_bias
from .events import RegimeChangeEvent

logger = logging.getLogger(__name__)

# ── environment ───────────────────────────────────────────────────────────

ALERT_WEBHOOK_URL_ENV = "ALERT_WEBHOOK_URL"
TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"
DISCORD_WEBHOOK_URL_ENV = "DISCORD_WEBHOOK_URL"

#: Env vars whose values are secrets (scrubbed from logs by ``mra_web``).
ALERT_SECRET_ENV_VARS: tuple[str, ...] = (
    ALERT_WEBHOOK_URL_ENV,
    TELEGRAM_BOT_TOKEN_ENV,
    DISCORD_WEBHOOK_URL_ENV,
)

# ── HTTP delivery defaults ────────────────────────────────────────────────

#: Seconds before an alert request times out (connect and read).
DEFAULT_TIMEOUT = 10.0
#: Retries after the first attempt, for network errors, 429 and 5xx.
DEFAULT_RETRIES = 2
#: Base delay before the first retry; doubles after each attempt.
_BACKOFF_SECONDS = 1.0
#: Longest server-requested ``Retry-After`` that is honored (seconds).
_MAX_RETRY_AFTER_SECONDS = 30.0
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
#: Response excerpt length included in error messages.
_EXCERPT = 120

TELEGRAM_API_BASE = "https://api.telegram.org"
#: Telegram ``sendMessage`` text limit (characters).
TELEGRAM_MAX_TEXT = 4096
#: Discord ``content`` limit (characters).
DISCORD_MAX_CONTENT = 2000

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_MASK = "***"
#: Ignore very short secrets; they would mask unrelated text.
_MIN_SECRET_LENGTH = 6

# ── secret masking ────────────────────────────────────────────────────────

# Telegram bot tokens look like "123456789:AAH..." (bot id, colon, 30+ url-safe chars)
_TELEGRAM_TOKEN = re.compile(r"(?<![0-9])\d{5,}:[A-Za-z0-9_-]{20,}")

_secrets: set[str] = set()
_secrets_lock = threading.Lock()


def register_secret(value: str | None) -> None:
    """Register a secret value so :func:`mask_secrets` hides it.

    For a URL, its path (with and without the query) is registered too, because
    HTTP client errors often print the path without the scheme and host.

    Args:
        value: The secret; empty or very short values are ignored.
    """
    if not value or len(value) < _MIN_SECRET_LENGTH:
        return
    values = {value}
    if "://" in value:
        try:
            parts = urlsplit(value)
        except ValueError:
            parts = None
        if parts is not None:
            path_query = parts.path + (f"?{parts.query}" if parts.query else "")
            values.update(v for v in (parts.path, path_query) if len(v) >= _MIN_SECRET_LENGTH)
    with _secrets_lock:
        _secrets.update(values)


def mask_secrets(text: str) -> str:
    """Replace registered secrets, Telegram bot tokens and credential query values with ``***``."""
    if not text:
        return text
    with _secrets_lock:
        secrets = sorted(_secrets, key=len, reverse=True)
    for value in secrets:
        text = text.replace(value, _MASK)
    text = _TELEGRAM_TOKEN.sub(_MASK, text)
    return redact(text)


def safe_url(url: str) -> str:
    """Return ``scheme://host/***``: enough to identify a channel, without the secret path."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or "?"
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return _MASK
    return f"{parts.scheme}://{host}{port}/{_MASK}"


def validate_notifier_url(url: str, *, name: str = "URL") -> str:
    """Validate an alert URL: ``https://`` only, or ``http://`` to localhost for testing.

    Args:
        url: The URL to check (surrounding whitespace is stripped).
        name: Label for error messages (e.g. the env var name). The URL itself is
            never included in the message.

    Returns:
        The stripped URL.

    Raises:
        NotifierConfigError: If the URL is empty, malformed, not https (except
            http to localhost), or carries user credentials.
    """
    if not isinstance(url, str) or not url.strip():
        raise NotifierConfigError(f"{name} must be a non-empty URL")
    url = url.strip()
    try:
        parts = urlsplit(url)
        host = parts.hostname
        _ = parts.port  # raises ValueError on a malformed port
    except ValueError:
        raise NotifierConfigError(f"{name} is not a valid URL") from None
    if any(ch.isspace() for ch in url) or not host:
        raise NotifierConfigError(f"{name} is not a valid URL")
    if parts.username is not None or parts.password is not None:
        raise NotifierConfigError(f"{name} must not contain user credentials")
    scheme = parts.scheme.lower()
    if scheme == "https" or (scheme == "http" and host.lower() in _LOCAL_HOSTS):
        return url
    raise NotifierConfigError(f"{name} must use https:// (http:// is only allowed for localhost)")


# ── HTTP helper ───────────────────────────────────────────────────────────


def post_json(  # noqa: PLR0913 - keyword-only knobs with defaults
    url: str,
    payload: Mapping[str, Any],
    *,
    channel: str,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> requests.Response:
    """POST ``payload`` as JSON, retrying network errors, 429 and 5xx.

    Args:
        url: Target URL (a secret: never logged or put in errors).
        payload: JSON body.
        channel: Channel label for error messages (e.g. ``"Telegram"``).
        timeout: Per-request timeout in seconds.
        retries: Retries after the first attempt.
        sleep: Sleep function between retries (injectable for tests).

    Returns:
        The successful (2xx) response.

    Raises:
        NotifierError: When every attempt failed. The message names the channel
            and the HTTP status or exception type only (plus a masked excerpt of
            the response body); the original exception is not chained, because
            its text can contain the URL.
    """
    delay = _BACKOFF_SECONDS
    failure = "no attempt made"
    for attempt in range(retries + 1):
        last = attempt == retries
        try:
            # No redirects: following one would re-send the alert to another
            # (possibly plain-http) location, bypassing the https-only check
            response = requests.post(
                url, json=dict(payload), timeout=timeout, allow_redirects=False
            )
        except requests.RequestException as e:
            failure = f"request failed ({type(e).__name__})"
            if not last:
                sleep(delay)
                delay *= 2
                continue
            break
        if 200 <= response.status_code < 300:  # noqa: PLR2004 - 2xx only (3xx = redirect)
            return response
        # Mask before truncating, so a cut can never leave part of a secret behind
        excerpt = mask_secrets((response.text or "").replace(url, _MASK))[:_EXCERPT]
        failure = f"HTTP {response.status_code}" + (f": {excerpt}" if excerpt else "")
        if response.status_code in _RETRYABLE_STATUS and not last:
            retry_after = response.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.isdigit() else delay
            if wait > _MAX_RETRY_AFTER_SECONDS:
                break
            sleep(wait)
            delay *= 2
            continue
        break
    raise NotifierError(f"{channel} delivery failed: {failure}")


# ── notifiers ─────────────────────────────────────────────────────────────


@runtime_checkable
class Notifier(Protocol):
    """Delivers regime-change events to one channel.

    Attributes:
        name: Short channel name for logs and summaries (no secrets).
    """

    name: str

    def send(self, event: RegimeChangeEvent) -> None:
        """Deliver ``event``.

        Raises:
            NotifierError: If delivery failed (message free of secrets).
        """
        ...


class LogNotifier:
    """Log each event's message (INFO, logger ``mra_lib.scanner.notifiers``).

    Always available and never fails; the scanner's dry-run channel.

    Args:
        level: Logging level for the messages.
    """

    name = "log"

    def __init__(self, level: int = logging.INFO) -> None:
        self.level = level

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def describe(self) -> str:
        """Safe one-line description of the channel."""
        return "log"

    def send(self, event: RegimeChangeEvent) -> None:
        """Log ``event.format_message()``."""
        logger.log(self.level, "🔔 %s", event.format_message())


class _HttpNotifier:
    """Shared retry/timeout settings for the HTTP channels."""

    name = "http"

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not timeout > 0:
            raise NotifierConfigError("timeout must be positive")
        if retries < 0:
            raise NotifierConfigError("retries must be >= 0")
        self.timeout = timeout
        self.retries = retries
        self._sleep = sleep

    def _post(self, url: str, payload: Mapping[str, Any], channel: str) -> requests.Response:
        return post_json(
            url,
            payload,
            channel=channel,
            timeout=self.timeout,
            retries=self.retries,
            sleep=self._sleep,
        )


class WebhookNotifier(_HttpNotifier):
    """POST each event's :meth:`~RegimeChangeEvent.to_dict` as JSON to a URL.

    Args:
        url: Webhook URL (https; http only for localhost). Treated as a secret.
        timeout: Per-request timeout in seconds.
        retries: Retries for network errors, 429 and 5xx.
        sleep: Sleep between retries (injectable for tests).

    Raises:
        NotifierConfigError: If the URL is invalid.
    """

    name = "webhook"

    def __init__(
        self,
        url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(timeout=timeout, retries=retries, sleep=sleep)
        self._url = validate_notifier_url(url, name="Webhook URL")
        register_secret(self._url)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(url={safe_url(self._url)!r})"

    def describe(self) -> str:
        """Safe one-line description of the channel."""
        return f"webhook ({safe_url(self._url)})"

    def payload(self, event: RegimeChangeEvent) -> dict[str, Any]:
        """The JSON body sent for ``event``."""
        return event.to_dict()

    def send(self, event: RegimeChangeEvent) -> None:
        """POST the event."""
        self._post(self._url, self.payload(event), "Webhook")


class TelegramNotifier(_HttpNotifier):
    """Send each event as a plain-text Telegram message (Bot API ``sendMessage``).

    Plain text (no ``parse_mode``), so regime names and symbols never need
    escaping and cannot inject markup.

    Args:
        bot_token: Bot token from @BotFather (a secret: it is part of the URL path).
        chat_id: Target chat id (e.g. ``"123456789"``, ``"-100..."`` or ``"@channel"``).
        api_base: Bot API base URL (override for tests; https or localhost only).
        timeout: Per-request timeout in seconds.
        retries: Retries for network errors, 429 and 5xx.
        sleep: Sleep between retries (injectable for tests).

    Raises:
        NotifierConfigError: If the token, chat id or API base is invalid.
    """

    name = "telegram"

    def __init__(  # noqa: PLR0913 - keyword-only knobs with defaults
        self,
        bot_token: str,
        chat_id: str,
        *,
        api_base: str = TELEGRAM_API_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(timeout=timeout, retries=retries, sleep=sleep)
        token = bot_token.strip() if isinstance(bot_token, str) else ""
        if not token or any(ch.isspace() or ch in "/?#" for ch in token):
            raise NotifierConfigError("Telegram bot token is empty or malformed")
        chat = str(chat_id).strip() if chat_id is not None else ""
        if not chat or any(ch.isspace() for ch in chat):
            raise NotifierConfigError("Telegram chat id is empty or malformed")
        base = validate_notifier_url(api_base, name="Telegram API base URL").rstrip("/")
        self._token = token
        self.chat_id = chat
        self._url = f"{base}/bot{token}/sendMessage"
        register_secret(token)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(chat_id={self.chat_id!r}, bot_token={_MASK!r})"

    def describe(self) -> str:
        """Safe one-line description of the channel."""
        return f"telegram (chat {self.chat_id})"

    def payload(self, event: RegimeChangeEvent) -> dict[str, Any]:
        """The ``sendMessage`` body sent for ``event``."""
        return {
            "chat_id": self.chat_id,
            "text": event.format_message()[:TELEGRAM_MAX_TEXT],
            "disable_web_page_preview": True,
        }

    def send(self, event: RegimeChangeEvent) -> None:
        """Send the message; a Bot API ``{"ok": false}`` reply counts as a failure."""
        response = self._post(self._url, self.payload(event), "Telegram")
        try:
            body = response.json()
        except ValueError:
            return
        if isinstance(body, dict) and body.get("ok") is False:
            description = mask_secrets(str(body.get("description", "")))[:_EXCERPT]
            raise NotifierError(f"Telegram delivery failed: {description or 'ok=false'}")


# Embed colors by the new regime's bias
_DISCORD_COLORS = {"bullish": 0x2ECC71, "bearish": 0xE74C3C, "neutral": 0x95A5A6}


class DiscordNotifier(_HttpNotifier):
    """Post each event to a Discord channel webhook (``content`` plus one embed).

    Mentions are disabled (``allowed_mentions: {"parse": []}``), so message text
    can never ping ``@everyone``.

    Args:
        webhook_url: Discord webhook URL (a secret: it embeds the webhook token).
        timeout: Per-request timeout in seconds.
        retries: Retries for network errors, 429 and 5xx.
        sleep: Sleep between retries (injectable for tests).

    Raises:
        NotifierConfigError: If the URL is invalid.
    """

    name = "discord"

    def __init__(
        self,
        webhook_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(timeout=timeout, retries=retries, sleep=sleep)
        self._url = validate_notifier_url(webhook_url, name="Discord webhook URL")
        register_secret(self._url)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(webhook_url={safe_url(self._url)!r})"

    def describe(self) -> str:
        """Safe one-line description of the channel."""
        return f"discord ({safe_url(self._url)})"

    def payload(self, event: RegimeChangeEvent) -> dict[str, Any]:
        """The webhook body sent for ``event``."""
        fields = [
            {"name": "Confidence", "value": f"{event.confidence:.0%}", "inline": True},
            {"name": "Strategy", "value": event.recommended_strategy, "inline": True},
            {"name": "Confirmation", "value": event.confirmation_summary(), "inline": False},
            {"name": "Bar (UTC)", "value": event.bar_time.isoformat(sep=" "), "inline": True},
        ]
        if event.close is not None:
            fields.append({"name": "Close", "value": f"{event.close:.2f}", "inline": True})
        return {
            "content": f"Regime change: {event.title}"[:DISCORD_MAX_CONTENT],
            "embeds": [
                {
                    "title": event.title[:256],
                    "description": f"Previous bar {event.previous_bar_time.isoformat(sep=' ')} UTC",
                    "color": _DISCORD_COLORS[regime_bias(event.new_regime).value],
                    "fields": fields,
                    "timestamp": event.detected_at.isoformat(),
                }
            ],
            "allowed_mentions": {"parse": []},
        }

    def send(self, event: RegimeChangeEvent) -> None:
        """POST the event to the webhook."""
        self._post(self._url, self.payload(event), "Discord")


def notifiers_from_env(env: Mapping[str, str] | None = None) -> list[Notifier]:
    """Build the HTTP notifiers configured in the environment.

    A channel is enabled only when its variables are set (non-blank):

    - ``ALERT_WEBHOOK_URL`` → :class:`WebhookNotifier`
    - ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` → :class:`TelegramNotifier`
    - ``DISCORD_WEBHOOK_URL`` → :class:`DiscordNotifier`

    :class:`LogNotifier` is not included; add it explicitly if wanted.

    Args:
        env: Variables to read (default: ``os.environ``).

    Returns:
        The enabled notifiers (possibly empty).

    Raises:
        NotifierConfigError: If a set variable is invalid (e.g. a non-https URL)
            or only one of the Telegram pair is set. Messages name the variable,
            never its value.
    """
    import os  # noqa: PLC0415

    source: Mapping[str, str] = os.environ if env is None else env

    def get(name: str) -> str:
        return (source.get(name) or "").strip()

    notifiers: list[Notifier] = []
    if url := get(ALERT_WEBHOOK_URL_ENV):
        register_secret(url)
        notifiers.append(WebhookNotifier(_env_url(url, ALERT_WEBHOOK_URL_ENV)))
    token, chat = get(TELEGRAM_BOT_TOKEN_ENV), get(TELEGRAM_CHAT_ID_ENV)
    if token or chat:
        register_secret(token)
        if not (token and chat):
            raise NotifierConfigError(
                f"{TELEGRAM_BOT_TOKEN_ENV} and {TELEGRAM_CHAT_ID_ENV} must both be set"
            )
        try:
            notifiers.append(TelegramNotifier(token, chat))
        except NotifierConfigError as e:
            raise NotifierConfigError(
                f"{TELEGRAM_BOT_TOKEN_ENV}/{TELEGRAM_CHAT_ID_ENV}: {e}"
            ) from None
    if url := get(DISCORD_WEBHOOK_URL_ENV):
        register_secret(url)
        notifiers.append(DiscordNotifier(_env_url(url, DISCORD_WEBHOOK_URL_ENV)))
    return notifiers


def _env_url(url: str, name: str) -> str:
    return validate_notifier_url(url, name=name)


def describe_notifier(notifier: Notifier) -> str:
    """Safe description of a notifier: its ``describe()`` if defined, else its name."""
    describe = getattr(notifier, "describe", None)
    if callable(describe):
        return str(describe())
    return str(getattr(notifier, "name", type(notifier).__name__))


__all__ = [
    "ALERT_SECRET_ENV_VARS",
    "ALERT_WEBHOOK_URL_ENV",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "DISCORD_WEBHOOK_URL_ENV",
    "TELEGRAM_BOT_TOKEN_ENV",
    "TELEGRAM_CHAT_ID_ENV",
    "DiscordNotifier",
    "LogNotifier",
    "Notifier",
    "TelegramNotifier",
    "WebhookNotifier",
    "describe_notifier",
    "mask_secrets",
    "notifiers_from_env",
    "post_json",
    "register_secret",
    "safe_url",
    "validate_notifier_url",
]
