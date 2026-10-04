"""
Secret scrubbing for server-side logs.

Provider SDKs put credentials in request URLs (e.g. Alpha Vantage ``apikey=``), and
those URLs end up in exception messages. This module redacts such values before a
log record is written. Clients never receive exception text at all; this protects
the server's own logs.
"""

import logging
import numbers
import os
import re

from mra_lib.data_providers.credentials import PROVIDER_ENV_PAIRS, PROVIDER_ENV_VARS
from mra_lib.scanner.notifiers import ALERT_SECRET_ENV_VARS

REDACTED = "***"

# key=value pairs in URLs, query strings, and repr() output.
# key=value pairs in URLs/query strings, and 'key': 'value' in dict reprs.
_KV_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|apikey|access[_-]?token|token|secret|password|apca-api-secret-key"
    r"|apca-api-key-id)(['\"]?\s*[=:]\s*['\"]?)([^&\s'\",;)}\]]+)"
)
# Authorization headers ("Bearer <jwt>", Tiingo's "Token <key>").
_BEARER_PATTERN = re.compile(r"(?i)\b((?:bearer|token)\s+)([A-Za-z0-9._~+/=-]{8,})")

# Ignore very short env values; they would redact unrelated text.
_MIN_LITERAL_LENGTH = 8


def _literal_secrets() -> list[str]:
    """Return configured secret values that must never appear in logs."""
    names = {"JWT_SECRET", "API_KEYS", *ALERT_SECRET_ENV_VARS}
    for env_vars in PROVIDER_ENV_VARS.values():
        names.update(env_vars)
    for pair in PROVIDER_ENV_PAIRS.values():
        names.update(pair)

    values: list[str] = []
    for name in names:
        raw = os.getenv(name)
        if not raw:
            continue
        parts = raw.split(",") if name == "API_KEYS" else [raw]
        values.extend(p.strip() for p in parts if len(p.strip()) >= _MIN_LITERAL_LENGTH)
    # Replace longer values first so a prefix never leaves a suffix behind.
    return sorted(values, key=len, reverse=True)


def scrub_secrets(text: str) -> str:
    """Redact credentials (``apikey=``, ``token=``, bearer tokens, known secrets)."""
    if not text:
        return text
    for value in _literal_secrets():
        text = text.replace(value, REDACTED)
    text = _KV_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    return _BEARER_PATTERN.sub(lambda m: f"{m.group(1)}{REDACTED}", text)


def _scrub_arg(arg: object) -> object:
    # Keep the original object unless its text contains a secret, so numeric
    # format specifiers (%d, %.2f with numpy/Decimal values) and formatters that
    # unpack typed args (uvicorn's access log) keep working.
    if arg is None or isinstance(arg, numbers.Number):
        return arg
    text = str(arg)
    scrubbed = scrub_secrets(text)
    return arg if scrubbed == text else scrubbed


class SecretScrubFilter(logging.Filter):
    """Logging filter that redacts secrets from the message, args and traceback.

    The ``args`` structure is preserved (only string-converted and scrubbed) so
    formatters that unpack it keep working.
    """

    _formatter = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.args:
            record.msg = scrub_secrets(str(record.msg))
        # With args, msg is a %-format template from code; scrubbing it could eat a
        # placeholder (e.g. "token=%s"), so only the interpolated values are scrubbed.
        elif isinstance(record.args, tuple):
            record.args = tuple(_scrub_arg(a) for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: _scrub_arg(v) for k, v in record.args.items()}
        if record.exc_info and not record.exc_text:
            record.exc_text = self._formatter.formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = scrub_secrets(record.exc_text)
        return True


_SCRUBBED_LOGGERS = ("", "uvicorn", "uvicorn.error", "uvicorn.access")


def install_log_scrubber() -> None:
    """Attach a SecretScrubFilter to the root and uvicorn handlers (idempotent)."""
    for name in _SCRUBBED_LOGGERS:
        for handler in logging.getLogger(name).handlers:
            if not any(isinstance(f, SecretScrubFilter) for f in handler.filters):
                handler.addFilter(SecretScrubFilter())
