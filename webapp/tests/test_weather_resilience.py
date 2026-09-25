"""Network failures must never crash a page.

A raw ``requests.exceptions.ReadTimeout`` escaping ``geocode_city()`` took down
the whole Dataset Upload page with a traceback. Every call in ``core.weather``
now goes through ``_get_json()``, which retries transient failures and converts
anything else into a catchable ``WeatherServiceError``.
"""
from datetime import datetime
from unittest.mock import patch

import pytest
import requests

from core import weather as W


class _Response:
    """Minimal stand-in for a successful requests.Response."""

    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


GEO_PAYLOAD = {
    "results": [{
        "name": "Laubsdorf", "country": "Germany", "admin1": "Brandenburg",
        "latitude": 51.687, "longitude": 14.414, "elevation": 84,
        "timezone": "Europe/Berlin",
    }]
}


@pytest.fixture(autouse=True)
def _no_real_sleep():
    """Keep the retry backoff from slowing the suite down."""
    with patch.object(W.time, "sleep", lambda *_: None):
        yield


# ── The bug that crashed the page ────────────────────────────────────────────

def test_geocode_returns_empty_list_on_timeout_instead_of_raising():
    with patch.object(W.requests, "get", side_effect=requests.exceptions.ReadTimeout("x")):
        assert W.geocode_city("Laubsdorf") == []


def test_geocode_strict_raises_a_catchable_error_not_a_requests_exception():
    with patch.object(W.requests, "get", side_effect=requests.exceptions.ReadTimeout("x")):
        with pytest.raises(W.WeatherServiceError):
            W.geocode_city_strict("Laubsdorf")


# A date well inside the covered window, whenever the suite happens to run.
_IN_WINDOW = datetime(2024, 6, 15, 12, 0)


@pytest.mark.parametrize("func, args", [
    ("fetch_weather", (51.7, 14.4, _IN_WINDOW)),
    ("fetch_forecast", (51.7, 14.4)),
])
def test_weather_calls_raise_weather_service_error(func, args):
    with patch.object(W.requests, "get", side_effect=requests.exceptions.ConnectionError("x")):
        with pytest.raises(W.WeatherServiceError):
            getattr(W, func)(*args)


# ── Retry behaviour ──────────────────────────────────────────────────────────

def test_a_transient_timeout_is_retried_and_recovers():
    calls = {"n": 0}

    def flaky(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ReadTimeout("transient")
        return _Response(GEO_PAYLOAD)

    with patch.object(W.requests, "get", side_effect=flaky):
        results = W.geocode_city("Laubsdorf")

    assert calls["n"] == 2, "should have retried exactly once"
    assert results and results[0]["latitude"] == pytest.approx(51.687)


def test_http_errors_are_not_retried():
    """A 500 will not fix itself — retrying just makes the user wait longer."""
    class Err500:
        status_code = 500

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

        def json(self):
            return {}

    calls = {"n": 0}

    def always_500(*_a, **_k):
        calls["n"] += 1
        return Err500()

    with patch.object(W.requests, "get", side_effect=always_500):
        with pytest.raises(W.WeatherServiceError, match="HTTP 500"):
            W.geocode_city_strict("Laubsdorf")

    assert calls["n"] == 1


def test_malformed_json_is_reported_clearly():
    class BadBody:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("not json")

    with patch.object(W.requests, "get", return_value=BadBody()):
        with pytest.raises(W.WeatherServiceError, match="malformed"):
            W.geocode_city_strict("Laubsdorf")


def test_successful_lookup_still_works():
    with patch.object(W.requests, "get", return_value=_Response(GEO_PAYLOAD)):
        results = W.geocode_city("Laubsdorf")
    assert len(results) == 1
    assert "Laubsdorf" in results[0]["display"]
