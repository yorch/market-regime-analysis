"""Tests for the alert notifiers (requests is mocked; no network)."""

import json
import logging
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

from mra_lib.config.enums import DirectionalBias
from mra_lib.errors import NotifierConfigError, NotifierError
from mra_lib.scanner import (
    DiscordNotifier,
    LogNotifier,
    RegimeChangeEvent,
    TelegramNotifier,
    WebhookNotifier,
    mask_secrets,
    notifiers_from_env,
    validate_notifier_url,
)
from mra_lib.signals.confirmation import ConfirmationReason, TimeframeConfirmation

TOKEN = "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"
WEBHOOK = "https://hooks.example.com/services/T000/B000/s3cretWebhookPath"
DISCORD = "https://discord.com/api/webhooks/1234567890/dIsCoRdWeBhOoKtOkEn_secret"
POST = "mra_lib.scanner.notifiers.requests.post"


def response(status: int = 200, body: object | None = None, headers=None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.ok = 200 <= status < 300
    resp.text = json.dumps(body) if body is not None else ""
    resp.headers = headers or {}
    if body is None:
        resp.json.side_effect = ValueError("no json")
    else:
        resp.json.return_value = body
    return resp


@pytest.fixture
def event() -> RegimeChangeEvent:
    confirmation = TimeframeConfirmation(
        direction=DirectionalBias.BULLISH,
        agreement=0.72,
        confirmed=True,
        primary_timeframe="1D",
        aligned_timeframes=("1D", "1H"),
        conflicting_timeframes=("15m",),
        unavailable_timeframes=(),
        risk_timeframes=(),
        confidence=0.85,
        threshold=0.6,
        reason=ConfirmationReason.CONFIRMED,
    )
    return RegimeChangeEvent(
        symbol="SPY",
        timeframe="1D",
        previous_regime="Bear Trending",
        new_regime="Bull Trending",
        confidence=0.81,
        bar_time=datetime(2026, 10, 2),
        previous_bar_time=datetime(2026, 10, 1),
        recommended_strategy="Trend Following",
        confirmation=confirmation,
        detected_at=datetime(2026, 10, 2, 21, 5, tzinfo=UTC),
        provider="yfinance",
        close=571.23,
    )


def no_sleep(_: float) -> None:
    return None


class TestEvent:
    def test_to_dict_is_json_safe(self, event):
        data = event.to_dict()
        assert json.loads(json.dumps(data, allow_nan=False)) == data
        assert data["event"] == "regime_change"
        assert data["bar_time"] == "2026-10-02T00:00:00"
        assert data["confirmation"]["direction"] == "bullish"
        assert data["confirmation"]["confirmed"] is True

    def test_message(self, event):
        msg = event.format_message()
        assert msg.splitlines()[0] == "Regime change: SPY 1D: Bear Trending -> Bull Trending"
        assert "Confidence 81%" in msg and "close 571.23" in msg
        assert "bullish, confirmed (agreement 72%; 1D, 1H)" in msg

    def test_non_finite_close_is_null(self, event):
        from dataclasses import replace

        assert replace(event, close=float("nan")).to_dict()["close"] is None


class TestWebhook:
    def test_posts_event_json_with_timeout(self, event):
        with patch(POST, return_value=response(204)) as post:
            WebhookNotifier(WEBHOOK, sleep=no_sleep).send(event)
        post.assert_called_once()
        args, kwargs = post.call_args
        assert args == (WEBHOOK,)
        assert kwargs["json"] == event.to_dict()
        assert kwargs["timeout"] == 10.0

    def test_retries_then_raises_without_secret(self, event, caplog):
        sleeps: list[float] = []
        with (
            patch(POST, return_value=response(503, {"error": "down " + WEBHOOK})) as post,
            caplog.at_level(logging.DEBUG),
            pytest.raises(NotifierError) as info,
        ):
            WebhookNotifier(WEBHOOK, retries=2, sleep=sleeps.append).send(event)
        assert post.call_count == 3
        assert sleeps == [1.0, 2.0]
        assert "HTTP 503" in str(info.value)
        assert WEBHOOK not in str(info.value) and "s3cretWebhookPath" not in str(info.value)
        assert "s3cretWebhookPath" not in caplog.text

    def test_network_error_message_has_no_url(self, event):
        error = requests.ConnectionError(f"Max retries exceeded with url: {WEBHOOK}")
        with patch(POST, side_effect=error), pytest.raises(NotifierError) as info:
            WebhookNotifier(WEBHOOK, retries=1, sleep=no_sleep).send(event)
        text = str(info.value)
        assert "ConnectionError" in text and "s3cretWebhookPath" not in text
        assert info.value.__cause__ is None and info.value.__context__ is None

    def test_client_error_not_retried(self, event):
        with (
            patch(POST, return_value=response(400, {"e": 1})) as post,
            pytest.raises(NotifierError),
        ):
            WebhookNotifier(WEBHOOK, sleep=no_sleep).send(event)
        assert post.call_count == 1

    def test_long_retry_after_fails_fast(self, event):
        with (
            patch(POST, return_value=response(429, {}, {"Retry-After": "120"})) as post,
            pytest.raises(NotifierError, match="HTTP 429"),
        ):
            WebhookNotifier(WEBHOOK, sleep=no_sleep).send(event)
        assert post.call_count == 1

    def test_redirects_not_followed(self, event):
        with (
            patch(POST, return_value=response(302, None, {"Location": "http://evil"})) as post,
            pytest.raises(NotifierError, match="HTTP 302"),
        ):
            WebhookNotifier(WEBHOOK, sleep=no_sleep).send(event)
        assert post.call_args.kwargs["allow_redirects"] is False

    def test_secret_masked_before_truncation(self, event):
        body = {"error": "x" * 90 + WEBHOOK}  # the 120-char cut lands inside the URL
        with patch(POST, return_value=response(400, body)), pytest.raises(NotifierError) as info:
            WebhookNotifier(WEBHOOK, sleep=no_sleep).send(event)
        assert "hooks" not in str(info.value) and "***" in str(info.value)

    def test_repr_hides_path(self):
        notifier = WebhookNotifier(WEBHOOK)
        assert "s3cret" not in repr(notifier) and "s3cret" not in notifier.describe()
        assert "hooks.example.com" in repr(notifier)


class TestTelegram:
    def test_send_message_payload(self, event):
        with patch(POST, return_value=response(200, {"ok": True})) as post:
            TelegramNotifier(TOKEN, "-100123", sleep=no_sleep).send(event)
        args, kwargs = post.call_args
        assert args == (f"https://api.telegram.org/bot{TOKEN}/sendMessage",)
        assert kwargs["json"] == {
            "chat_id": "-100123",
            "text": event.format_message(),
            "disable_web_page_preview": True,
        }
        assert "parse_mode" not in kwargs["json"]
        assert kwargs["timeout"] == 10.0

    def test_ok_false_is_failure_and_token_masked(self, event):
        body = {"ok": False, "description": f"Unauthorized for bot{TOKEN}"}
        with patch(POST, return_value=response(200, body)), pytest.raises(NotifierError) as info:
            TelegramNotifier(TOKEN, "1", sleep=no_sleep).send(event)
        assert TOKEN not in str(info.value)
        assert "Unauthorized" in str(info.value)

    def test_http_error_hides_token(self, event, caplog):
        error = requests.ConnectTimeout(f"https://api.telegram.org/bot{TOKEN}/sendMessage")
        with (
            patch(POST, side_effect=error),
            caplog.at_level(logging.DEBUG),
            pytest.raises(NotifierError) as info,
        ):
            TelegramNotifier(TOKEN, "1", retries=0).send(event)
        assert TOKEN not in str(info.value)
        assert TOKEN not in caplog.text

    def test_truncates_long_text(self, event):
        from dataclasses import replace

        long_event = replace(event, recommended_strategy="x" * 5000)
        payload = TelegramNotifier(TOKEN, "1").payload(long_event)
        assert len(payload["text"]) == 4096

    @pytest.mark.parametrize("token", ["", "  ", "abc/def", "has space"])
    def test_invalid_token(self, token):
        with pytest.raises(NotifierConfigError) as info:
            TelegramNotifier(token, "1")
        assert token.strip() == "" or token not in str(info.value)

    def test_repr_hides_token(self):
        notifier = TelegramNotifier(TOKEN, "42")
        assert TOKEN not in repr(notifier) and "AAH" not in repr(notifier)
        assert notifier.describe() == "telegram (chat 42)"


class TestDiscord:
    def test_payload_shape(self, event):
        with patch(POST, return_value=response(204)) as post:
            DiscordNotifier(DISCORD, sleep=no_sleep).send(event)
        args, kwargs = post.call_args
        assert args == (DISCORD,)
        body = kwargs["json"]
        assert body["content"] == "Regime change: SPY 1D: Bear Trending -> Bull Trending"
        assert body["allowed_mentions"] == {"parse": []}
        (embed,) = body["embeds"]
        assert embed["title"] == "SPY 1D: Bear Trending -> Bull Trending"
        assert embed["color"] == 0x2ECC71
        assert {f["name"] for f in embed["fields"]} >= {"Confidence", "Strategy", "Confirmation"}
        assert kwargs["timeout"] == 10.0

    def test_failure_hides_webhook_token(self, event):
        with (
            patch(POST, return_value=response(404, {"message": "Unknown Webhook"})),
            pytest.raises(NotifierError) as info,
        ):
            DiscordNotifier(DISCORD, sleep=no_sleep).send(event)
        assert "dIsCoRd" not in str(info.value) and "Unknown Webhook" in str(info.value)


class TestLogNotifier:
    def test_logs_message(self, event, caplog):
        with caplog.at_level(logging.INFO, logger="mra_lib.scanner"):
            LogNotifier().send(event)
        assert "SPY 1D: Bear Trending -> Bull Trending" in caplog.text


class TestUrlValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/hook",
            "http://localhost:8080/hook",
            "http://127.0.0.1/hook",
            "http://[::1]:9000/x",
        ],
    )
    def test_valid(self, url):
        assert validate_notifier_url(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "",
            "http://example.com/hook",
            "ftp://example.com",
            "https://",
            "https://user:pass@example.com/x",
            "javascript:alert(1)",
            "https://exa mple.com",
            "https://example.com:99999/",
        ],
    )
    def test_invalid_never_echoes_url(self, url):
        with pytest.raises(NotifierConfigError) as info:
            validate_notifier_url(url, name="ALERT_WEBHOOK_URL")
        assert "ALERT_WEBHOOK_URL" in str(info.value)
        if url:
            assert url not in str(info.value)


class TestFromEnv:
    def test_nothing_configured(self):
        assert notifiers_from_env({}) == []

    def test_all_channels(self):
        notifiers = notifiers_from_env(
            {
                "ALERT_WEBHOOK_URL": WEBHOOK,
                "TELEGRAM_BOT_TOKEN": TOKEN,
                "TELEGRAM_CHAT_ID": "42",
                "DISCORD_WEBHOOK_URL": DISCORD,
            }
        )
        assert [n.name for n in notifiers] == ["webhook", "telegram", "discord"]

    def test_blank_values_are_unset(self):
        assert notifiers_from_env({"ALERT_WEBHOOK_URL": "  ", "TELEGRAM_CHAT_ID": ""}) == []

    def test_telegram_pair_required(self):
        with pytest.raises(NotifierConfigError, match="must both be set") as info:
            notifiers_from_env({"TELEGRAM_BOT_TOKEN": TOKEN})
        assert TOKEN not in str(info.value)
        with pytest.raises(NotifierConfigError, match="must both be set"):
            notifiers_from_env({"TELEGRAM_CHAT_ID": "42"})

    def test_insecure_url_rejected_by_name(self):
        url = "http://hooks.example.com/s3cret"
        with pytest.raises(NotifierConfigError, match="DISCORD_WEBHOOK_URL") as info:
            notifiers_from_env({"DISCORD_WEBHOOK_URL": url})
        assert "s3cret" not in str(info.value)

    def test_localhost_http_allowed(self):
        (notifier,) = notifiers_from_env({"ALERT_WEBHOOK_URL": "http://localhost:9999/hook"})
        assert notifier.name == "webhook"

    def test_reads_os_environ(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", WEBHOOK)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        assert [n.name for n in notifiers_from_env()] == ["webhook"]


class TestSecretsNeverLogged:
    def test_url_path_is_masked_too(self):
        WebhookNotifier(WEBHOOK)
        text = mask_secrets("host='hooks.example.com' url: /services/T000/B000/s3cretWebhookPath")
        assert "s3cretWebhookPath" not in text

    def test_mask_secrets(self):
        TelegramNotifier(TOKEN, "1")
        WebhookNotifier(WEBHOOK)
        text = mask_secrets(f"failed {WEBHOOK} and bot{TOKEN} and 987654321:{'x' * 30}")
        assert WEBHOOK not in text and TOKEN not in text and "x" * 30 not in text
