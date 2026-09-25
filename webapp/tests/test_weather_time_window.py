"""The weather client must be honest about *when* it can answer.

Before this, ``fetch_weather()`` always asked for today and silently snapped to
the nearest available hour, so a prediction for 2024-06-15 was really today's
weather under yesterday's heading. Now the date picks the endpoint (ERA5
archive for the past, forecast for today + 15 days), a date outside that window
raises before any request is sent, and a missing hour is an error rather than a
neighbouring value.
"""
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
import requests

from core import weather as W


TODAY = date(2026, 9, 17)   # frozen "today" for every test in this module


@pytest.fixture(autouse=True)
def _frozen_today(monkeypatch):
    monkeypatch.setattr(W, "_today", lambda: TODAY)


@pytest.fixture(autouse=True)
def _no_real_sleep():
    with patch.object(W.time, "sleep", lambda *_: None):
        yield


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _day_payload(day: date, *, hours=range(24), ghi=None, tz="Europe/Berlin"):
    """A minimal Open-Meteo hourly payload for one local day."""
    times = [f"{day:%Y-%m-%d}T{h:02d}:00" for h in hours]
    n = len(times)
    if ghi is None:
        ghi = [max(0.0, 100.0 * (12 - abs(12 - h))) for h in hours]
    return {
        "timezone": tz,
        "utc_offset_seconds": 7200,
        "hourly": {
            "time": times,
            "shortwave_radiation":  list(ghi),
            "temperature_2m":       [18.0] * n,
            "relative_humidity_2m": [65.0] * n,
            "dew_point_2m":         [8.0] * n,
            "wind_speed_10m":       [3.0] * n,
            "wind_direction_10m":   [180.0] * n,
            "precipitation":        [0.0] * n,
        },
    }


@pytest.fixture
def capture_get():
    """Patch requests.get, record every (url, params), return a fixed payload."""
    calls = []

    def _install(payload):
        def _get(url, params=None, timeout=None):
            calls.append((url, params or {}))
            return _Response(payload)

        patcher = patch.object(W.requests, "get", side_effect=_get)
        patcher.start()
        return calls

    yield _install
    patch.stopall()


# ── The covered window ───────────────────────────────────────────────────────

def test_available_window_spans_era5_to_fifteen_days_ahead():
    win = W.available_window()
    assert win.min_date == date(1940, 1, 1)
    assert win.today    == TODAY
    assert win.max_date == date(2026, 10, 2)          # 17 Sep + 15 days


@pytest.mark.parametrize("day, expected", [
    (date(1940, 1, 1),  "archive"),                   # first ERA5 day
    (date(2024, 6, 15), "archive"),
    (TODAY - timedelta(days=1), "archive"),
    (TODAY, "forecast"),
    (TODAY + timedelta(days=1), "forecast"),
    (TODAY + timedelta(days=15), "forecast"),         # last forecast day
])
def test_resolve_source_picks_the_right_endpoint(day, expected):
    assert W.resolve_source(day) == expected


@pytest.mark.parametrize("day", [
    date(1939, 12, 31),
    TODAY + timedelta(days=16),
    date(2100, 1, 1),
])
def test_dates_outside_the_window_are_refused(day):
    with pytest.raises(W.DateOutOfRangeError):
        W.resolve_source(day)


def test_out_of_range_error_is_catchable_as_a_weather_service_error():
    """Pages that only catch WeatherServiceError must not break."""
    err = W.DateOutOfRangeError(date(2030, 1, 1), date(1940, 1, 1), TODAY)
    assert isinstance(err, W.WeatherServiceError)
    assert err.requested == date(2030, 1, 1)
    assert "2030-01-01" in str(err) and "1940-01-01" in str(err)


def test_out_of_range_raises_before_any_network_call():
    def _explode(*_a, **_k):
        raise AssertionError("no request should be sent for an unusable date")

    with patch.object(W.requests, "get", side_effect=_explode):
        with pytest.raises(W.DateOutOfRangeError):
            W.fetch_weather(51.7, 14.4, datetime(2026, 12, 25, 12))


# ── Which endpoint, and with which parameters ────────────────────────────────

def test_past_date_queries_the_archive_with_a_single_day_range(capture_get):
    day = date(2024, 6, 15)
    calls = capture_get(_day_payload(day))

    W.fetch_weather(51.7, 14.4, datetime(2024, 6, 15, 12))

    url, params = calls[0]
    assert url == W._ARCHIVE_URL
    assert params["start_date"] == params["end_date"] == "2024-06-15"
    assert params["timezone"] == "auto"
    # start_date and forecast_days together are rejected by the API
    assert "forecast_days" not in params


@pytest.mark.parametrize("offset", [0, 3, 15])
def test_today_and_future_dates_query_the_forecast_endpoint(capture_get, offset):
    day = TODAY + timedelta(days=offset)
    calls = capture_get(_day_payload(day))

    W.fetch_weather(51.7, 14.4, datetime.combine(day, datetime.min.time()).replace(hour=12))

    url, params = calls[0]
    assert url == W._FORECAST_URL
    assert params["start_date"] == params["end_date"] == day.isoformat()


# ── No silent substitution ───────────────────────────────────────────────────

def test_a_missing_hour_raises_instead_of_snapping_to_a_neighbour(capture_get):
    """The regression test for the deleted nearest-timestamp fallback."""
    day = date(2024, 3, 31)
    # A spring daylight-saving day: 02:00 does not exist locally.
    hours = [h for h in range(24) if h != 2]
    capture_get(_day_payload(day, hours=hours))

    with pytest.raises(W.WeatherServiceError, match="02:00"):
        W.fetch_weather(51.7, 14.4, datetime(2024, 3, 31, 2, 30))


def test_a_day_without_any_radiation_data_raises(capture_get):
    day = date(2024, 6, 15)
    capture_get(_day_payload(day, ghi=[None] * 24))

    with pytest.raises(W.WeatherServiceError, match="radiation"):
        W.fetch_weather(51.7, 14.4, datetime(2024, 6, 15, 12))


def test_a_null_ghi_at_the_requested_hour_raises(capture_get):
    day = date(2024, 6, 15)
    ghi = [200.0] * 24
    ghi[14] = None
    capture_get(_day_payload(day, ghi=ghi))

    with pytest.raises(W.WeatherServiceError, match="radiation value"):
        W.fetch_weather(51.7, 14.4, datetime(2024, 6, 15, 14))


def test_a_missing_non_radiation_value_is_substituted_and_reported(capture_get):
    day = date(2024, 6, 15)
    payload = _day_payload(day)
    payload["hourly"]["temperature_2m"][14] = None
    capture_get(payload)

    result = W.fetch_weather(51.7, 14.4, datetime(2024, 6, 15, 14))

    assert result["Temp_WS"] == 15.0            # the documented default
    assert result["_missing"] == ["temperature"]


# ── Provenance reported back to the page ─────────────────────────────────────

def test_past_date_reports_archive_provenance(capture_get):
    day = date(2026, 9, 10)                     # one week before frozen today
    capture_get(_day_payload(day))

    result = W.fetch_weather(51.7, 14.4, datetime(2026, 9, 10, 14))

    assert result["_source"] == "archive"
    assert "ERA5" in result["_source_label"]
    assert result["_matched_time"] == "2026-09-10T14:00"
    assert result["_horizon_days"] == -7
    assert result["_date"] == "2026-09-10"
    assert result["_missing"] == []


def test_future_date_reports_the_forecast_horizon(capture_get):
    day = TODAY + timedelta(days=3)
    capture_get(_day_payload(day))

    result = W.fetch_weather(51.7, 14.4, datetime.combine(day, datetime.min.time()).replace(hour=12))

    assert result["_source"] == "forecast"
    assert "+3 days" in result["_source_label"]
    assert result["_horizon_days"] == 3


def test_today_is_labelled_without_a_horizon(capture_get):
    capture_get(_day_payload(TODAY))

    result = W.fetch_weather(51.7, 14.4, datetime(2026, 9, 17, 12))

    assert result["_horizon_days"] == 0
    assert "Today" in result["_source_label"]


# ── Accepted input types ─────────────────────────────────────────────────────

@pytest.mark.parametrize("given", [
    datetime(2024, 6, 15, 14, 37),              # minutes are floored away
    pd.Timestamp("2024-06-15 14:00"),
    "2024-06-15T14:00",
])
def test_the_datetime_argument_accepts_several_forms(capture_get, given):
    capture_get(_day_payload(date(2024, 6, 15)))
    result = W.fetch_weather(51.7, 14.4, given)
    assert result["_matched_time"] == "2024-06-15T14:00"


def test_a_missing_datetime_is_a_programming_error_not_a_service_error():
    def _explode(*_a, **_k):
        raise AssertionError("no request should be sent")

    with patch.object(W.requests, "get", side_effect=_explode):
        with pytest.raises(ValueError):
            W.fetch_weather(51.7, 14.4, None)


# ── The day series used by the chart and the DLI ─────────────────────────────

def test_the_day_series_covers_the_whole_day_and_keeps_gaps_as_nan(capture_get):
    day = date(2024, 6, 15)
    ghi = [200.0] * 24
    ghi[3] = None
    capture_get(_day_payload(day, ghi=ghi))

    series = W.fetch_weather(51.7, 14.4, datetime(2024, 6, 15, 12))["_day_series"]

    assert list(series.columns) == ["time", "GHI", "temperature", "precipitation"]
    assert len(series) == 24
    # NaN leaves a visible gap in the chart; a zero would claim darkness.
    assert series["GHI"].isna().sum() == 1


def test_fetch_forecast_defaults_to_today(capture_get):
    calls = capture_get(_day_payload(TODAY))

    df = W.fetch_forecast(51.7, 14.4)

    url, params = calls[0]
    assert url == W._FORECAST_URL
    assert params["start_date"] == TODAY.isoformat()
    assert len(df) == 24


def test_fetch_forecast_honours_an_explicit_past_date(capture_get):
    calls = capture_get(_day_payload(date(2024, 6, 15)))

    W.fetch_forecast(51.7, 14.4, datetime(2024, 6, 15, 8))

    url, params = calls[0]
    assert url == W._ARCHIVE_URL
    assert params["start_date"] == "2024-06-15"


# ── Error reporting ──────────────────────────────────────────────────────────

def test_the_api_explanation_for_a_rejected_range_is_shown(capture_get):
    """Open-Meteo puts the real cause in a JSON 'reason' field."""
    class Err400:
        status_code = 400

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

        def json(self):
            return {"error": True,
                    "reason": "Value 'start_date' is out of allowed range"}

    with patch.object(W.requests, "get", return_value=Err400()):
        with pytest.raises(W.WeatherServiceError, match="out of allowed range"):
            W.fetch_weather(51.7, 14.4, datetime(2024, 6, 15, 12))


# ── Units ────────────────────────────────────────────────────────────────────

def test_wind_is_requested_in_metres_per_second(capture_get):
    """Open-Meteo defaults wind to km/h; the model was trained on m/s sensors.

    Without this parameter a calm 4.8 m/s afternoon arrives as 17.4, which is
    above the training range (0–5.33 m/s) and raised a false out-of-domain
    warning on the prediction pages.
    """
    calls = capture_get(_day_payload(date(2026, 9, 18)))
    W.fetch_weather(51.6872, 14.4143, datetime(2026, 9, 18, 12, 0))
    assert calls, "no request was sent"
    _, params = calls[-1]
    assert params.get("wind_speed_unit") == "ms"


def test_every_other_variable_is_left_at_the_unit_the_model_expects(capture_get):
    """°C, %, mm, W/m² and degrees are already Open-Meteo's defaults — asking
    for them again would be noise, and a wrong override would be silent."""
    calls = capture_get(_day_payload(date(2026, 9, 18)))
    W.fetch_weather(51.6872, 14.4143, datetime(2026, 9, 18, 12, 0))
    _, params = calls[-1]
    assert "temperature_unit" not in params
    assert "precipitation_unit" not in params
