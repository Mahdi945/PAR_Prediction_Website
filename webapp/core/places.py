"""
core/places.py
──────────────
Find a place, and know what time it is there.

Typing coordinates by hand is the most error-prone step in the app: one wrong
digit in the latitude moves the sun by degrees and the prediction with it.
This module lets a visitor search for a city or region instead, and fills the
coordinates, the altitude and the timezone from the match they pick.

    search(query)                → list of candidate places (cached)
    reference_timezone(fallback) → the zone "now" should be read in
    local_now(tz) / local_hour   → the current wall time there
    place_picker(...)            → the Streamlit widget both pages use

One widget, live. Matches appear while the visitor types — each keystroke,
after a 250 ms pause, queries the geocoder and refreshes the dropdown; there is
nothing to press. Choosing a match fills the coordinates, altitude, timezone
and clock. If the searchbox component is missing the picker falls back to a
plain selectbox that needs Enter, so the page still works.

Accuracy. Coordinates come from Open-Meteo's geocoding service, built on the
GeoNames database: each hit is that settlement's official centroid, with its
terrain elevation and IANA timezone. Places sharing a name are separated by
district, region, country and population, all shown in the list — "Villingen,
Regierungsbezirk Gießen, Hesse" is not "Villingen-Schwenningen, Baden-
Wurttemberg", and the exact latitude, longitude, elevation and zone of whatever
was applied stay on screen underneath, so a wrong pick is visible.

A centroid is a town, not a field. For a specific plot, pick the town and then
nudge the coordinates — the number inputs stay editable on purpose.
"""

from __future__ import annotations

from datetime import datetime, time as dtime
from functools import lru_cache
from zoneinfo import ZoneInfo, available_timezones

import streamlit as st

from . import weather as W

try:
    # Live type-ahead needs a component: Streamlit's own text_input and
    # selectbox only rerun on Enter or blur, and selectbox filters just the
    # options already loaded, so neither can query as the visitor types.
    from streamlit_searchbox import st_searchbox
    HAS_SEARCHBOX = True
except ImportError:                                  # pragma: no cover
    st_searchbox = None
    HAS_SEARCHBOX = False

__all__ = ["search", "reverse", "identify", "reference_timezone", "local_now",
           "local_hour", "local_clock", "interpret", "place_picker", "KNOWN_SITES",
           "timezone_choices"]

MAX_RESULTS = 8

# Open-Meteo's geocoder only goes name -> coordinates. For the other direction
# BigDataCloud's reverse-geocode-client endpoint is free, needs no key, and
# returns English locality names plus the IANA zone. Best effort only: if it is
# unreachable the app is unchanged, because the coordinates are what count.
_REVERSE_URL = "https://api.bigdatacloud.net/data/reverse-geocode-client"

# The two stations the model was trained on. Recognised directly so the default
# location is named for what it is, rather than as the municipality a reverse
# lookup would return (the Laubsdorf mast reverse-geocodes to Neuhausen/Spree).
KNOWN_SITES = [
    {"name": "Laubsdorf", "admin1": "Brandenburg", "admin2": "", "country": "Germany",
     "country_code": "DE", "latitude": 51.68718711157154, "longitude": 14.414256224166728,
     "elevation": 84.0, "timezone": "Europe/Berlin", "population": 0,
     "display": "Laubsdorf, Brandenburg, Germany — training station"},
    {"name": "Nebelin", "admin1": "Brandenburg", "admin2": "Prignitz", "country": "Germany",
     "country_code": "DE", "latitude": 53.11833921379164, "longitude": 11.746088890223463,
     "elevation": 50.0, "timezone": "Europe/Berlin", "population": 0,
     "display": "Nebelin, Prignitz, Brandenburg, Germany — training station"},
]
KNOWN_SITE_RADIUS_KM = 2.0


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False, max_entries=2048)
def search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """Candidate places for a free-text query. Never raises; [] means no match."""
    q = (query or "").strip()
    if len(q) < 2:
        return []
    return W.geocode_city(q, max_results)


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance, used only to recognise the known stations."""
    import math
    r, p1, p2 = 6371.0088, math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


@st.cache_data(ttl=30 * 24 * 3600, show_spinner=False, max_entries=4096)
def reverse(lat: float, lon: float) -> dict | None:
    """Coordinates -> the settlement they fall in. None when it cannot be resolved.

    Shaped like a forward geocoding hit so both directions can be handled the
    same way, except that `elevation` is absent: this answers "where is this
    point", it does not re-measure the terrain.
    """
    try:
        r = W.requests.get(_REVERSE_URL, timeout=10,
                           params={"latitude": float(lat), "longitude": float(lon),
                                   "localityLanguage": "en"})
        r.raise_for_status()
        d = r.json()
    except Exception:                       # offline, rate-limited, anything
        return None

    locality = (d.get("locality") or "").strip()
    city     = (d.get("city") or "").strip()
    region   = (d.get("principalSubdivision") or "").strip()
    country  = (d.get("countryName") or "").strip()
    if not any((locality, city, region, country)):
        return None                         # open ocean

    tz = ""
    for item in (d.get("localityInfo") or {}).get("informative", []):
        if item.get("description") == "time zone" and item.get("name"):
            tz = item["name"]
            break

    parts, seen = [], set()
    for part in (locality, city, region, country):
        if part and part not in seen:
            seen.add(part)
            parts.append(part)
    return {
        "name": locality or city or region or country,
        "admin1": region, "admin2": city if city != locality else "",
        "country": country, "country_code": d.get("countryCode", "") or "",
        "latitude": float(lat), "longitude": float(lon),
        "elevation": None, "timezone": tz, "population": 0,
        "display": ", ".join(parts),
    }


def identify(lat: float, lon: float) -> dict | None:
    """Name the point: a training station if it is one, otherwise a reverse lookup."""
    for site in KNOWN_SITES:
        if _km(lat, lon, site["latitude"], site["longitude"]) <= KNOWN_SITE_RADIUS_KM:
            return dict(site)
    return reverse(round(float(lat), 3), round(float(lon), 3))   # ~110 m, so the cache bites


# ── Time ─────────────────────────────────────────────────────────────────────

def browser_timezone() -> str | None:
    """The visitor's own IANA zone, as reported by their browser.

    Without this the app defaults to the *server's* clock. On Streamlit Cloud
    that is UTC, so a visitor in Germany opening the page at 16:00 CEST was
    offered 14:00 — an hour label that matched nobody.
    """
    try:
        tz = st.context.timezone
    except Exception:                       # older Streamlit, or no browser context
        return None
    return tz if tz and _is_valid(tz) else None


def _is_valid(tz: str) -> bool:
    try:
        ZoneInfo(tz)
        return True
    except Exception:
        return False


def reference_timezone(preferred: str | None = None, fallback: str = "UTC") -> str:
    """Which clock "now" should be read on.

    The place's own zone when one has been chosen, otherwise the visitor's
    browser zone, otherwise `fallback`.
    """
    if preferred and _is_valid(preferred):
        return preferred
    return browser_timezone() or fallback


def local_now(tz: str | None = None) -> datetime:
    """Current wall time in `tz` (naive, so it drops straight into a widget)."""
    zone = tz if tz and _is_valid(tz) else reference_timezone()
    return datetime.now(ZoneInfo(zone)).replace(tzinfo=None)


def local_hour(tz: str | None = None) -> dtime:
    """Current local hour, floored — the weather API is hourly anyway."""
    return dtime(local_now(tz).hour, 0)


@lru_cache(maxsize=1)
def _all_timezones() -> tuple[str, ...]:
    """Every IANA zone this interpreter knows, sorted. ~600 of them."""
    try:
        return tuple(sorted(available_timezones()))
    except Exception:                       # pragma: no cover - no tz database
        return ()


def timezone_choices(current: str | None = None) -> list[str]:
    """Options for a timezone picker, guaranteed to contain `current`.

    A Streamlit selectbox raises if the value held in session state is not one
    of its options, and the zone can arrive from a reverse lookup (which has
    returned things like ``Etc/GMT+2`` over open water). Anything unexpected is
    kept and listed first rather than silently dropped.
    """
    options = list(_all_timezones())
    if current and current not in options:
        options.insert(0, current)
    return options


def local_clock(tz: str | None = None) -> dtime:
    """Current local time *as a clock shows it*, minutes included.

    This is what goes in the "local time at that location" field. The floored
    hour is what the weather request ends up using, but showing 16:00 beside a
    caption reading 16:34 looks like two different times — so the field states
    the real one and the page says which hour is fetched.
    """
    now = local_now(tz)
    return dtime(now.hour, now.minute)


# ── The picker ───────────────────────────────────────────────────────────────

def _label(p: dict) -> str:
    """The text shown in the dropdown. Must be unique per place, and must
    change when the search changes — Streamlit keeps a widget's rendered
    options when they compare equal, so positional indices as options made two
    different searches look identical and left a stale name on screen."""
    pop = f"  ·  {p['population']:,} people" if p.get("population") else ""
    return f"{p['display']}{pop}"


def interpret(typed: str | None, offered: list[dict]) -> tuple[str, object, str]:
    """What a value coming out of the picker means.

    Returns ``(action, payload, note)`` where action is one of:

        "ignore"  nothing selected
        "pick"    `payload` is the place the visitor chose from the list
        "single"  `payload` is a one-item result list — apply it, nothing to choose
        "many"    `payload` is the result list — the visitor picks next
        "none"    the search found nothing; `payload` is []

    Kept free of Streamlit so it can be tested directly: ``AppTest`` cannot
    drive ``accept_new_options``, so the widget wiring is verified in a browser
    but the decision it encodes is verified here.
    """
    if not typed or not str(typed).strip():
        return "ignore", None, ""
    by_label = {_label(p): p for p in offered}
    if typed in by_label:
        return "pick", by_label[typed], ""

    query = str(typed).strip()
    hits = search(query)
    if not hits:
        return "none", [], (f"No place matched “{query}”. Check the spelling, "
                            f"or type the coordinates below.")
    if len(hits) == 1:
        return "single", hits, ""
    return "many", hits, f"{len(hits)} places match “{query}” — open the list and pick one."


def place_picker(
    *,
    key: str,
    lat_key: str,
    lon_key: str,
    alt_key: str,
    tz_key: str | None = None,
    date_key: str | None = None,
    time_key: str | None = None,
    label: str = "Search a city or region",
    default: dict | None = None,
) -> None:
    """One autocomplete box: type a place, press Enter, pick a match.

    A single ``st.selectbox`` with ``accept_new_options=True`` does both jobs.
    Anything typed that is not already on the list is treated as a new search;
    the matches it returns become the list, and choosing one fills the
    coordinate fields. A lone match is applied straight away, because there is
    nothing to disambiguate.

    Optional `tz_key`, `date_key` and `time_key` also move the timezone field
    and the clock to the chosen place — which is what "local time at that
    location" is supposed to mean.
    """
    opts_key, sel_key, note_key = f"{key}_options", f"{key}_select", f"{key}_note"

    # First render: show the place the coordinate fields already point at, so
    # the box and the numbers below it agree instead of starting blank.
    if opts_key not in st.session_state:
        st.session_state[opts_key] = [default] if default else []
        if default:
            st.session_state[f"{key}_chosen"] = default
            if not HAS_SEARCHBOX:
                # Only the fallback selectbox keeps its value under this key.
                # st_searchbox stores a dict of its own there and chokes on a
                # plain string ("string indices must be integers").
                st.session_state[sel_key] = _label(default)

    places: list[dict] = st.session_state.get(opts_key, [])

    def _apply(place: dict) -> None:
        st.session_state[lat_key] = float(place["latitude"])
        st.session_state[lon_key] = float(place["longitude"])
        st.session_state[alt_key] = float(place["elevation"] or 0.0)
        tz = place.get("timezone") or ""
        if tz_key and tz:
            st.session_state[tz_key] = tz
        if tz and (date_key or time_key):
            there = local_now(tz)
            if date_key:
                st.session_state[date_key] = there.date()
            if time_key:
                st.session_state[time_key] = dtime(there.hour, there.minute)
        st.session_state[f"{key}_applied"] = place["display"]
        st.session_state[f"{key}_chosen"] = place
        st.session_state[note_key] = ""

    def _on_change() -> None:
        action, payload, note = interpret(st.session_state.get(sel_key),
                                          st.session_state.get(opts_key, []))
        if action == "ignore":
            return
        if action == "pick":
            _apply(payload)
            return
        # a new search: the hits become the list, and the typed text is dropped
        st.session_state[opts_key] = payload
        st.session_state[sel_key] = None
        st.session_state[note_key] = note
        if action == "single":
            _apply(payload[0])

    _help = ("Start typing; matches appear as you go. Coordinates, altitude and "
             "timezone come from the GeoNames database via Open-Meteo.")

    if HAS_SEARCHBOX:
        def _suggest(term: str) -> list[tuple[str, dict]]:
            """Called on each keystroke, after the debounce."""
            return [(_label(p), p) for p in search(term)]

        picked = st_searchbox(
            _suggest,
            key=sel_key,
            label=label,
            placeholder="Start typing a place…",
            default_options=[(_label(default), default)] if default else None,
            default=default,
            debounce=250,               # ms of quiet before a lookup is sent
            rerun_on_update=True,
            help=_help,
        )
        if isinstance(picked, dict) and picked.get("display") != st.session_state.get(f"{key}_applied"):
            _apply(picked)
            # the pages read the coordinates at the top of the script, so a
            # rerun keeps the caption and the clock in step with the new point
            st.rerun()
    else:                                            # pragma: no cover
        # Fallback: one selectbox that doubles as a search box. Needs Enter,
        # but keeps the page working if the component is unavailable.
        st.selectbox(
            label,
            options=[_label(p) for p in places],
            index=None,
            key=sel_key,
            accept_new_options=True,
            filter_mode="contains",
            placeholder="Type a place and press Enter…",
            on_change=_on_change,
            help=_help,
        )

    note = st.session_state.get(note_key)
    if note:
        st.caption(note)

    chosen = st.session_state.get(f"{key}_chosen")
    if chosen:
        tz_note = f" · {chosen['timezone']}" if chosen.get("timezone") else ""
        st.caption(
            f"📍 **{chosen['latitude']:.4f}°N, {chosen['longitude']:.4f}°E** · "
            f"{chosen['elevation']:.0f} m{tz_note}"
        )
