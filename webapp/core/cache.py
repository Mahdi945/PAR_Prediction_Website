"""
core/cache.py
─────────────
Streamlit-cached front for the Open-Meteo client.

``core.weather`` stays plain Python (tests call it directly). The pages import
*these* wrappers, so repeating a prediction — a rerun, a second visitor asking
about the same place and hour — costs no network call and no Open-Meteo quota
(10,000 requests/day on the free tier).

    fetch_weather(lat, lon, dt) → dict          same return as core.weather.fetch_weather
    geocode_city(name)          → list[dict]    same return as core.weather.geocode_city

TTLs follow how fast the underlying data can change:
    archive  (ERA5, past dates)   24 h  — reanalysis is final once published
    forecast (today → +15 days)    1 h  — Open-Meteo refreshes its model runs hourly
    geocoding                      7 d  — place names do not move
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from . import weather as W
from .weather import DateOutOfRangeError, WeatherServiceError  # re-exported for the pages

__all__ = ["fetch_weather", "geocode_city", "DateOutOfRangeError", "WeatherServiceError"]

_COORD_DECIMALS = 4      # ~11 m; Open-Meteo's grid is far coarser than that


def _coords(lat: float, lon: float) -> tuple[float, float]:
    return round(float(lat), _COORD_DECIMALS), round(float(lon), _COORD_DECIMALS)


@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=1024)
def _fetch_archive(lat: float, lon: float, dt_iso: str) -> dict:
    return W.fetch_weather(lat, lon, datetime.fromisoformat(dt_iso))


@st.cache_data(ttl=3600, show_spinner=False, max_entries=1024)
def _fetch_forecast(lat: float, lon: float, dt_iso: str) -> dict:
    return W.fetch_weather(lat, lon, datetime.fromisoformat(dt_iso))


def fetch_weather(lat: float, lon: float, dt) -> dict:
    """Cached ``core.weather.fetch_weather``. Errors are never cached: a failed
    request is retried on the next call."""
    when = W._normalise_dt(dt)                    # floors to the hour → stable cache key
    source = W.resolve_source(when.date())        # DateOutOfRangeError before touching the cache
    lat, lon = _coords(lat, lon)
    fn = _fetch_archive if source == "archive" else _fetch_forecast
    return fn(lat, lon, when.isoformat())


@st.cache_data(ttl=3600, show_spinner=False, max_entries=512)
def _fetch_ensemble(lat: float, lon: float, dt_iso: str) -> dict | None:
    return W.fetch_ensemble(lat, lon, datetime.fromisoformat(dt_iso))


def fetch_ensemble(lat: float, lon: float, dt) -> dict | None:
    """Cached ``core.weather.fetch_ensemble``.

    Same one-hour TTL as the forecast it accompanies, and ``None`` is a valid
    cached answer: when the ensemble has nothing for an hour it will not have
    anything a second later either, and the page works without it.
    """
    when = W._normalise_dt(dt)
    lat, lon = _coords(lat, lon)
    return _fetch_ensemble(lat, lon, when.isoformat())


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False, max_entries=2048)
def geocode_city(name: str, max_results: int = 5) -> list[dict]:
    """Cached ``core.weather.geocode_city`` (returns [] on failure, like the original)."""
    return W.geocode_city(name.strip(), max_results)
