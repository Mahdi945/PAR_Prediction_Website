"""
core/weather.py
───────────────
Open-Meteo API client for PAR Predictor.

Provides:
  fetch_weather(lat, lon, dt)  →  dict of weather variables at a specific time
  fetch_forecast(lat, lon)     →  DataFrame of today's hourly forecast
  geocode_city(name)           →  list of matching locations with lat/lon
"""

from __future__ import annotations

import time

import numpy as np
import requests
import pandas as pd
from datetime import datetime, timedelta


# ── API endpoints ────────────────────────────────────────────────────────────
_WEATHER_URL  = "https://api.open-meteo.com/v1/forecast"
_GEO_URL      = "https://geocoding-api.open-meteo.com/v1/search"
_TIMEOUT      = 12  # seconds per attempt
_RETRIES      = 3   # transient timeouts are common on shared hosting
_BACKOFF      = 1.5 # seconds, doubled after each failed attempt


class WeatherServiceError(RuntimeError):
    """Open-Meteo could not be reached, or returned something unusable.

    Every network call in this module raises this instead of letting a raw
    requests exception escape — an unhandled ReadTimeout crashes the whole
    Streamlit page with a traceback.
    """


def _get_json(url: str, params: dict, *, what: str, retries: int = _RETRIES) -> dict:
    """GET JSON with retries and a clear, catchable error.

    Retries only on transient failures (timeout / connection reset). An HTTP
    error or malformed body fails immediately — retrying would not help.
    """
    last_exc: Exception | None = None

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()

        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(_BACKOFF * (2 ** attempt))

        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            raise WeatherServiceError(
                f"{what} failed: the service replied with HTTP {code}."
            ) from exc

        except ValueError as exc:                       # body was not valid JSON
            raise WeatherServiceError(
                f"{what} failed: the service returned a malformed response."
            ) from exc

    raise WeatherServiceError(
        f"{what} failed: no response after {retries} attempts "
        f"({type(last_exc).__name__}). Check the internet connection, or enter "
        f"the values manually."
    ) from last_exc

# Open-Meteo hourly variable names we need
_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "shortwave_radiation",
]


# ── Public helpers ────────────────────────────────────────────────────────────

def fetch_weather(lat: float, lon: float, dt: datetime) -> dict:
    """
    Fetch weather conditions from Open-Meteo for *lat/lon* at datetime *dt*.

    Returns a flat dict with keys matching training feature names:
        GHI_RC_01, Temp_WS, RH_WS, DWP_WS, WS_WS, WD_WS,
        PREC_INT_WS, _timezone
    """
    params = {
        "latitude":     round(lat, 4),
        "longitude":    round(lon, 4),
        "hourly":       ",".join(_HOURLY_VARS),
        "timezone":     "auto",
        "forecast_days": 1,
    }

    data = _get_json(_WEATHER_URL, params, what="Weather fetch")

    # Robust time matching — pure Python datetime, no pandas version dependency.
    # Open-Meteo time strings are always "YYYY-MM-DDTHH:MM" (hourly, UTC-based).
    time_strs = data["hourly"]["time"]   # e.g. ["2026-08-04T00:00", ...]
    _fmt = "%Y-%m-%dT%H:%M"

    # Convert input dt to a naive Python datetime, floored to the hour
    if isinstance(dt, datetime):
        target_naive = dt.replace(minute=0, second=0, microsecond=0,
                                  tzinfo=None)
    else:
        target_naive = datetime.strptime(str(dt)[:16], _fmt)

    # Fast path: exact hour match
    target_str = target_naive.strftime(_fmt)
    if target_str in time_strs:
        idx = time_strs.index(target_str)
    else:
        # Fallback: closest timestamp by absolute difference in seconds
        times_native = [datetime.strptime(t, _fmt) for t in time_strs]
        diffs = [abs((t - target_naive).total_seconds()) for t in times_native]
        idx = int(np.argmin(diffs))

    def _safe(key: str, default: float = 0.0) -> float:
        val = data["hourly"].get(key, [None])[idx]
        return float(val) if val is not None else default

    return {
        "GHI_RC_01":   max(0.0, _safe("shortwave_radiation")),
        "Temp_WS":     _safe("temperature_2m",        15.0),
        "RH_WS":       _safe("relative_humidity_2m",  60.0),
        "DWP_WS":      _safe("dew_point_2m",          10.0),
        "WS_WS":       _safe("wind_speed_10m",         2.0),
        "WD_WS":       _safe("wind_direction_10m",   180.0),
        "PREC_INT_WS": max(0.0, _safe("precipitation", 0.0)),
        # metadata
        "_timezone":   data.get("timezone", "UTC"),
        "_utc_offset": data.get("utc_offset_seconds", 0),
        "_times":      time_strs,           # raw strings, no pandas needed
        "_ghi_series": [
            max(0.0, v) if v is not None else 0.0
            for v in data["hourly"]["shortwave_radiation"]
        ],
        "_temp_series": data["hourly"].get("temperature_2m", []),
        "_prec_series": data["hourly"].get("precipitation", []),
    }


def fetch_forecast(
    lat: float,
    lon: float,
    dt: datetime | None = None,
) -> pd.DataFrame:
    """
    Return today's hourly forecast as a DataFrame with columns:
        time, GHI, temperature, precipitation.

    The optional *dt* argument is accepted for API compatibility with
    fetch_weather() and normal-mode page calls, but the full daily forecast
    is still returned so the chart and DLI summary remain intact.
    """
    params = {
        "latitude":     round(lat, 4),
        "longitude":    round(lon, 4),
        "hourly":       "shortwave_radiation,temperature_2m,precipitation",
        "timezone":     "auto",
        "forecast_days": 1,
    }
    data = _get_json(_WEATHER_URL, params, what="Forecast fetch")

    df = pd.DataFrame({
        "time":         pd.to_datetime(data["hourly"]["time"]),
        "GHI":          [max(0.0, v or 0.0) for v in data["hourly"]["shortwave_radiation"]],
        "temperature":  data["hourly"]["temperature_2m"],
        "precipitation": data["hourly"]["precipitation"],
    })
    return df


def geocode_city(name: str, max_results: int = 5) -> list[dict]:
    """
    Search for a city by name.  Returns a list of dicts:
        {name, country, admin1, latitude, longitude, elevation}

    Best-effort: returns an empty list if the lookup fails or times out, since
    every caller offers editable coordinates as a fallback. Use
    ``geocode_city_strict()`` when the caller needs to report the failure.
    """
    try:
        return geocode_city_strict(name, max_results)
    except WeatherServiceError:
        return []


def geocode_city_strict(name: str, max_results: int = 5) -> list[dict]:
    """Same as geocode_city() but raises WeatherServiceError on failure."""
    params = {
        "name":     name,
        "count":    max_results,
        "language": "en",
        "format":   "json",
    }
    # One retry only: this is a convenience lookup, not worth a long stall.
    results = _get_json(_GEO_URL, params, what="City lookup", retries=2).get("results", [])

    out = []
    for r in results:
        out.append({
            "name":      r.get("name", ""),
            "country":   r.get("country", ""),
            "admin1":    r.get("admin1", ""),
            "latitude":  r.get("latitude", 0.0),
            "longitude": r.get("longitude", 0.0),
            "elevation": r.get("elevation", 0.0),
            "display":   f"{r.get('name','')}, {r.get('admin1','')} – {r.get('country','')}",
        })
    return out
