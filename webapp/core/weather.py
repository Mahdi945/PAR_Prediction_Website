"""
core/weather.py
───────────────
Open-Meteo API client for PAR Predictor.

Two endpoints are used, chosen automatically from the requested date:

    date <  today   →  archive  (ERA5 reanalysis, back to 1940-01-01)
    date >= today   →  forecast (up to 15 days ahead)

Anything outside that window raises ``DateOutOfRangeError`` **before** any
network call. Nothing is ever silently substituted: if the requested hour or
its radiation value is missing, the caller gets an error, not a neighbouring
value dressed up as the answer.

Provides:
  available_window()             →  (min_date, max_date, today)
  resolve_source(date)           →  "archive" | "forecast"
  fetch_weather(lat, lon, dt)    →  dict of weather variables at a specific hour
  fetch_forecast(lat, lon, dt)   →  DataFrame of that local day's hourly series
  geocode_city(name)             →  list of matching locations with lat/lon
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Literal, NamedTuple

import requests
import pandas as pd


# ── API endpoints ────────────────────────────────────────────────────────────
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_WEATHER_URL  = _FORECAST_URL   # historical alias, kept bound for other callers
_ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
# Tried in order. ICON is the better forecast over Europe and carries more
# members, but its ensemble stops at about 7 days; GFS reaches 15, which is
# exactly the window this app offers. Without the second one the error bar
# would quietly vanish for half the range.
_ENSEMBLE_MODELS = ("icon_seamless", "gfs_seamless")
_GEO_URL      = "https://geocoding-api.open-meteo.com/v1/search"
_TIMEOUT      = 12  # seconds per attempt
_RETRIES      = 3   # transient timeouts are common on shared hosting
_BACKOFF      = 1.5 # seconds, doubled after each failed attempt

# ── Coverage of the two endpoints ────────────────────────────────────────────
ARCHIVE_START       = date(1940, 1, 1)   # ERA5 reanalysis begins here
FORECAST_DAYS_AHEAD = 15                 # today + 15 = 16 forecast days
_TIME_FMT           = "%Y-%m-%dT%H:%M"   # Open-Meteo's hourly time format


class WeatherServiceError(RuntimeError):
    """Open-Meteo could not be reached, or returned something unusable.

    Every network call in this module raises this instead of letting a raw
    requests exception escape — an unhandled ReadTimeout crashes the whole
    Streamlit page with a traceback.
    """


class DateOutOfRangeError(WeatherServiceError):
    """The requested date lies outside what Open-Meteo can serve.

    Subclasses WeatherServiceError so existing handlers keep catching it, but
    pages can catch it first and show the covered window instead of an API
    error message.
    """

    def __init__(self, requested: date, min_date: date, max_date: date):
        self.requested = requested
        self.min_date  = min_date
        self.max_date  = max_date
        super().__init__(
            f"No weather data for {requested:%Y-%m-%d}. Open-Meteo covers "
            f"{min_date:%Y-%m-%d} (ERA5 reanalysis archive) to "
            f"{max_date:%Y-%m-%d} (today + {FORECAST_DAYS_AHEAD}-day forecast). "
            f"Choose a date inside that range."
        )


def _today() -> date:
    """Calendar 'today' used for the archive/forecast split.

    UTC, matching how Open-Meteo computes its own allowed ranges server-side.
    """
    return datetime.now(timezone.utc).date()


class AvailabilityWindow(NamedTuple):
    min_date: date
    max_date: date
    today:    date


def available_window() -> AvailabilityWindow:
    """The date range both pages may offer. Recomputed on every call."""
    t = _today()
    return AvailabilityWindow(ARCHIVE_START, t + timedelta(days=FORECAST_DAYS_AHEAD), t)


def resolve_source(d: date) -> Literal["archive", "forecast"]:
    """Which endpoint serves *d*. Raises DateOutOfRangeError outside the window."""
    win = available_window()
    if d < win.min_date or d > win.max_date:
        raise DateOutOfRangeError(d, win.min_date, win.max_date)
    return "archive" if d < win.today else "forecast"


def source_label(d: date) -> tuple[str, int]:
    """Human description of where *d*'s data comes from, and its horizon in days."""
    horizon = (d - _today()).days
    if horizon < 0:
        return "Historical — ERA5 reanalysis (Open-Meteo archive)", horizon
    if horizon == 0:
        return "Today — Open-Meteo weather model", 0
    plural = "s" if horizon > 1 else ""
    return f"Forecast +{horizon} day{plural} — Open-Meteo", horizon


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
            code   = exc.response.status_code if exc.response is not None else "?"
            reason = _http_reason(exc)
            detail = f" — {reason}" if reason else ""
            raise WeatherServiceError(
                f"{what} failed: the service replied with HTTP {code}{detail}."
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


def _http_reason(exc: requests.HTTPError) -> str:
    """Open-Meteo explains 4xx errors in a JSON 'reason' field — surface it."""
    try:
        return str(exc.response.json().get("reason", "")).strip()
    except Exception:
        return ""


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

# Series returned for the chart and the daily light integral
_SERIES_VARS = ["shortwave_radiation", "temperature_2m", "precipitation"]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalise_dt(dt) -> datetime:
    """Accept datetime / pd.Timestamp / 'YYYY-MM-DDTHH:MM' → naive, floored hour.

    An aware datetime is read as *local wall time* at the requested location,
    the same convention compute_features() uses.
    """
    if dt is None:
        raise ValueError("fetch_weather() needs a date and time, got None.")
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if isinstance(dt, datetime):
        return dt.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    return datetime.strptime(str(dt)[:13] + ":00", _TIME_FMT)


def _fetch_day(
    lat: float,
    lon: float,
    day: date,
    hourly_vars: list[str],
    *,
    what: str,
) -> tuple[dict, str]:
    """Download one local day from whichever endpoint covers it.

    Returns (payload, source). Raises before any network call if *day* is
    outside the covered window.
    """
    source = resolve_source(day)
    url    = _ARCHIVE_URL if source == "archive" else _FORECAST_URL

    # start_date/end_date must never be combined with forecast_days —
    # the API rejects the pair.
    params = {
        "latitude":   round(lat, 4),
        "longitude":  round(lon, 4),
        "hourly":     ",".join(hourly_vars),
        "timezone":   "auto",
        "start_date": day.isoformat(),
        "end_date":   day.isoformat(),
        # Open-Meteo defaults wind to km/h. The station sensors the model was
        # trained on report m/s, so ask for m/s explicitly — otherwise every
        # wind reading arrives 3.6x too large and looks out of range.
        # Every other variable already matches: C, %, mm, W/m2, degrees.
        "wind_speed_unit": "ms",
    }

    data   = _get_json(url, params, what=what)
    hourly = data.get("hourly") or {}
    times  = hourly.get("time") or []

    if not times:
        raise WeatherServiceError(
            f"{what} failed: Open-Meteo returned no hourly data for "
            f"{day:%Y-%m-%d} at this location."
        )

    ghi = hourly.get("shortwave_radiation") or []
    if all(v is None for v in ghi):
        raise WeatherServiceError(
            f"Open-Meteo has no radiation data for {day:%Y-%m-%d} at this "
            f"location ({source} source). Try another date, or enter the "
            f"values manually in Expert Mode."
        )

    return data, source


def _build_day_df(data: dict) -> pd.DataFrame:
    """The selected local day as a DataFrame. Nulls stay NaN, never 0.

    A NaN leaves a visible gap in the chart and is skipped by .sum() for the
    daily light integral — a zero would silently claim darkness.
    """
    hourly = data["hourly"]
    n      = len(hourly["time"])

    def _col(key: str) -> list:
        values = hourly.get(key) or [None] * n
        return [float(v) if v is not None else float("nan") for v in values]

    ghi = [max(0.0, v) if v == v else v for v in _col("shortwave_radiation")]

    return pd.DataFrame({
        "time":          pd.to_datetime(hourly["time"]),
        "GHI":           ghi,
        "temperature":   _col("temperature_2m"),
        "precipitation": _col("precipitation"),
    })


# ── Public helpers ────────────────────────────────────────────────────────────

def fetch_weather(lat: float, lon: float, dt: datetime) -> dict:
    """
    Fetch weather conditions from Open-Meteo for *lat/lon* at datetime *dt*.

    The endpoint is chosen from the date: ERA5 reanalysis for the past, the
    forecast model for today and the next 15 days.

    Returns a flat dict with keys matching training feature names:
        GHI_RC_01, Temp_WS, RH_WS, DWP_WS, WS_WS, WD_WS, PREC_INT_WS
    plus metadata prefixed with '_' (source, matched hour, day series…).

    Raises:
        DateOutOfRangeError  — date outside 1940-01-01 … today+15
        WeatherServiceError  — network failure, missing hour, or missing GHI

    Note: on the spring daylight-saving transition the local day has 23 hours.
    Asking for the hour that does not exist raises rather than snapping to a
    neighbour.
    """
    target = _normalise_dt(dt)
    day    = target.date()

    data, source = _fetch_day(lat, lon, day, _HOURLY_VARS, what="Weather fetch")

    hourly     = data["hourly"]
    time_strs  = hourly["time"]
    target_str = target.strftime(_TIME_FMT)

    # Exact hour only — no nearest-neighbour fallback.
    if target_str not in time_strs:
        raise WeatherServiceError(
            f"Open-Meteo has no hourly record for {target_str} at this location "
            f"(the local day has {len(time_strs)} hours — this happens at a "
            f"daylight-saving transition). Choose another hour."
        )
    idx = time_strs.index(target_str)

    ghi_raw = (hourly.get("shortwave_radiation") or [None])[idx]
    if ghi_raw is None:
        raise WeatherServiceError(
            f"Open-Meteo has no radiation value for {target_str} at this "
            f"location. PAR cannot be predicted without GHI — choose another "
            f"hour, or enter the values manually in Expert Mode."
        )

    missing: list[str] = []

    def _safe(key: str, label: str, default: float) -> float:
        """Substitute a typical value for a missing sensor, and say so."""
        val = hourly.get(key, [None] * len(time_strs))[idx]
        if val is None:
            missing.append(label)
            return default
        return float(val)

    label, horizon = source_label(day)

    return {
        "GHI_RC_01":   max(0.0, float(ghi_raw)),
        "Temp_WS":     _safe("temperature_2m",       "temperature",   15.0),
        "RH_WS":       _safe("relative_humidity_2m", "humidity",      60.0),
        "DWP_WS":      _safe("dew_point_2m",         "dew point",     10.0),
        "WS_WS":       _safe("wind_speed_10m",       "wind speed",     2.0),
        "WD_WS":       _safe("wind_direction_10m",   "wind direction", 180.0),
        "PREC_INT_WS": max(0.0, _safe("precipitation", "precipitation", 0.0)),
        # provenance — shown to the user so the number is never anonymous
        "_source":       source,
        "_source_label": label,
        "_horizon_days": horizon,
        "_matched_time": target_str,
        "_date":         day.isoformat(),
        "_missing":      missing,
        "_day_series":   _build_day_df(data),
        # metadata
        "_timezone":   data.get("timezone", "UTC"),
        "_utc_offset": data.get("utc_offset_seconds", 0),
        "_times":      time_strs,
        "_ghi_series": [
            max(0.0, v) if v is not None else 0.0
            for v in hourly["shortwave_radiation"]
        ],
        "_temp_series": hourly.get("temperature_2m", []),
        "_prec_series": hourly.get("precipitation", []),
    }


def fetch_forecast(
    lat: float,
    lon: float,
    dt: datetime | None = None,
) -> pd.DataFrame:
    """
    Return the hourly series for the local day containing *dt*, with columns:
        time, GHI, temperature, precipitation.

    *dt* defaults to today. Past days come from the ERA5 archive, so the name
    is historical — ``fetch_day_series`` is the accurate alias.

    Normal Mode no longer calls this: fetch_weather() already returns the same
    day as ``_day_series``, which saves a second request and guarantees the
    point value and the chart share one source.
    """
    day = _normalise_dt(dt).date() if dt is not None else _today()
    data, _ = _fetch_day(lat, lon, day, _SERIES_VARS, what="Day series fetch")
    return _build_day_df(data)


fetch_day_series = fetch_forecast


def fetch_ensemble(lat: float, lon: float, dt) -> dict | None:
    """The spread of GHI across ensemble members for one hour.

    A deterministic forecast is the model run once. An ensemble runs it many
    times from slightly different starting states; where the members agree the
    atmosphere is predictable, where they scatter it is not. That scatter is a
    measurement of the forecast's own uncertainty, from the same provider and
    at no extra cost.

    Returns ``{"members": [...GHI...], "n": int, "hour": datetime}`` or ``None``
    when the ensemble has nothing for this hour — a past date, a location it
    does not cover, or the service being down. The caller shows a plain
    prediction in that case: an unavailable error bar must never cost the user
    their prediction.

    Note what this does and does not measure. It is the uncertainty in the
    *weather input*, not in the model, and the two add on top of each other.
    Ensembles also tend to be under-dispersive, so treat the spread as a floor
    rather than a full account of what could happen.
    """
    when = _normalise_dt(dt)
    day = when.date()

    if resolve_source(day) != "forecast":
        return None                      # the past is not forecast; it happened

    stamp = when.strftime(_TIME_FMT)

    for model in _ENSEMBLE_MODELS:
        try:
            data = _get_json(
                _ENSEMBLE_URL,
                {
                    "latitude":  round(lat, 4),
                    "longitude": round(lon, 4),
                    "hourly":    "shortwave_radiation",
                    "models":     model,
                    "timezone":   "auto",
                    "start_date": day.isoformat(),
                    "end_date":   day.isoformat(),
                },
                what="Ensemble forecast",
                retries=1,           # a missing error bar is not worth a retry storm
            )
        except (WeatherServiceError, requests.RequestException):
            continue

        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        if stamp not in times:
            continue
        i = times.index(stamp)

        # Every member arrives as its own column: shortwave_radiation,
        # shortwave_radiation_member01, _member02 ...
        members = [
            float(hourly[k][i])
            for k in hourly
            if k.startswith("shortwave_radiation")
            and i < len(hourly[k])
            and hourly[k][i] is not None
        ]
        if len(members) >= 2:
            return {"members": members, "n": len(members),
                    "hour": when, "model": model}

    return None


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
        # admin1 is the state/region, admin2 the district. Both are carried so
        # the picker can tell two places of the same name apart — there is a
        # Cottbus in Brandenburg and another in Missouri.
        parts = [r.get("name", ""), r.get("admin2", ""), r.get("admin1", ""), r.get("country", "")]
        out.append({
            "name":       r.get("name", ""),
            "country":    r.get("country", ""),
            "country_code": r.get("country_code", ""),
            "admin1":     r.get("admin1", ""),
            "admin2":     r.get("admin2", ""),
            "latitude":   r.get("latitude", 0.0),
            "longitude":  r.get("longitude", 0.0),
            "elevation":  r.get("elevation", 0.0),
            # The IANA zone of the place itself — what "local time there" means,
            # and what Expert Mode needs for its solar geometry.
            "timezone":   r.get("timezone", "") or "",
            "population": r.get("population") or 0,
            "display":    ", ".join(p for p in parts if p),
        })
    return out
