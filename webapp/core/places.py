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

Accuracy. Coordinates come from Open-Meteo's geocoding service, which is built
on the GeoNames database: each hit is that settlement's official centroid, with
its terrain elevation and IANA timezone. Two places of the same name are told
apart by district, region, country and population, all of which are shown
before anything is applied — "Cottbus, Brandenburg, Germany (84,754)" versus
"Cottbus, Howell, Missouri, United States". Nothing is written into the form
until the visitor presses the button, and the exact numbers are displayed first,
so a wrong match is visible rather than silent.

A centroid is a town, not a field. For a specific plot, pick the town and then
nudge the coordinates — the number inputs stay editable on purpose.
"""

from __future__ import annotations

from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import streamlit as st

from . import weather as W

__all__ = ["search", "reference_timezone", "local_now", "local_hour", "place_picker"]

MAX_RESULTS = 8


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False, max_entries=2048)
def search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """Candidate places for a free-text query. Never raises; [] means no match."""
    q = (query or "").strip()
    if len(q) < 2:
        return []
    return W.geocode_city(q, max_results)


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


# ── The picker ───────────────────────────────────────────────────────────────

def _label(p: dict) -> str:
    pop = f"  ·  {p['population']:,} people" if p.get("population") else ""
    return f"{p['display']}{pop}"


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
) -> None:
    """Search box → candidate list → "use these coordinates".

    Writes straight into the session-state keys of the coordinate widgets, so
    the form updates in place. `tz_key`, `date_key` and `time_key` are optional:
    when given, the timezone field is set to the place's own zone and the
    date/time are moved to *now there*, which is what "local time at that
    location" means.
    """
    query = st.text_input(
        label,
        key=f"{key}_query",
        placeholder="Cottbus · Brandenburg · Nairobi · 　…",
        help="Type a place and press Enter. Coordinates, altitude and timezone "
             "come from the GeoNames database via Open-Meteo.",
    )
    if not query or not query.strip():
        return

    with st.spinner("Looking up…"):
        results = search(query)

    if not results:
        st.caption(f"No place matched “{query.strip()}”. Check the spelling, or "
                   f"type the coordinates below.")
        return

    choice = st.selectbox(
        f"{len(results)} match{'es' if len(results) != 1 else ''}",
        options=range(len(results)),
        format_func=lambda i: _label(results[i]),
        key=f"{key}_choice",
    )
    place = results[choice]

    tz_note = f" · {place['timezone']}" if place.get("timezone") else ""
    st.caption(
        f"📍 **{place['latitude']:.4f}°N, {place['longitude']:.4f}°E** · "
        f"{place['elevation']:.0f} m{tz_note}"
    )

    if st.button("📍 Use these coordinates", key=f"{key}_apply", use_container_width=True):
        st.session_state[lat_key] = float(place["latitude"])
        st.session_state[lon_key] = float(place["longitude"])
        st.session_state[alt_key] = float(place["elevation"] or 0.0)
        tz = place.get("timezone") or ""
        if tz_key and tz:
            st.session_state[tz_key] = tz
        # "Local time at that location" only means something if the clock moves
        # with the place.
        if tz and (date_key or time_key):
            there = local_now(tz)
            if date_key:
                st.session_state[date_key] = there.date()
            if time_key:
                st.session_state[time_key] = dtime(there.hour, 0)
        st.session_state[f"{key}_applied"] = place["display"]
        st.rerun()
