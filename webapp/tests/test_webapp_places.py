"""Place search and the clock the pages offer by default.

Typing coordinates is the most error-prone step in the app, and the default
hour used to come from the *server's* clock — UTC on Streamlit Cloud, so a
visitor in Germany at 16:00 was shown 14:00.
"""
from datetime import date, datetime, time as dtime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from core import places as PL
from core import weather as W


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


GEO = {"results": [
    {"name": "Cottbus", "country": "Germany", "country_code": "DE", "admin1": "Brandenburg",
     "admin2": "", "latitude": 51.75769, "longitude": 14.32888, "elevation": 80.0,
     "timezone": "Europe/Berlin", "population": 84754},
    {"name": "Cottbus", "country": "United States", "country_code": "US", "admin1": "Missouri",
     "admin2": "Howell", "latitude": 36.64978, "longitude": -91.8482, "elevation": 326.0,
     "timezone": "America/Chicago", "population": 0},
]}


# ── Geocoding carries what the picker needs ──────────────────────────────────

def test_geocoding_returns_timezone_region_and_population():
    with patch.object(W.requests, "get", return_value=_Response(GEO)):
        hits = W.geocode_city("Cottbus")
    assert len(hits) == 2
    de, us = hits
    assert de["timezone"] == "Europe/Berlin" and us["timezone"] == "America/Chicago"
    assert de["population"] == 84754
    assert de["elevation"] == 80.0


def test_display_separates_two_places_of_the_same_name():
    """Without the region a visitor cannot tell the Brandenburg Cottbus from
    the one in Missouri — and would silently predict for the wrong continent."""
    with patch.object(W.requests, "get", return_value=_Response(GEO)):
        de, us = W.geocode_city("Cottbus")
    assert de["display"] == "Cottbus, Brandenburg, Germany"
    assert us["display"] == "Cottbus, Howell, Missouri, United States"
    assert de["display"] != us["display"]


# ── search() ─────────────────────────────────────────────────────────────────

def test_search_ignores_queries_too_short_to_mean_anything():
    """No network call for one stray keystroke."""
    with patch.object(W.requests, "get", side_effect=AssertionError("should not be called")):
        assert PL.search("") == []
        assert PL.search(" ") == []
        assert PL.search("C") == []


def test_search_survives_a_dead_network():
    import requests
    with patch.object(W.requests, "get", side_effect=requests.exceptions.ReadTimeout("x")):
        with patch.object(W.time, "sleep", lambda *_: None):
            assert PL.search("Cottbus zzz unique") == []


# ── The clock ────────────────────────────────────────────────────────────────

def test_reference_timezone_prefers_the_chosen_place():
    assert PL.reference_timezone("Asia/Tokyo") == "Asia/Tokyo"


def test_reference_timezone_ignores_a_nonsense_zone():
    assert PL.reference_timezone("Not/AZone", fallback="Europe/Berlin") in (
        "Europe/Berlin", PL.browser_timezone() or "Europe/Berlin")


def test_reference_timezone_falls_back_when_nothing_is_known(monkeypatch):
    monkeypatch.setattr(PL, "browser_timezone", lambda: None)
    assert PL.reference_timezone(None, fallback="Europe/Berlin") == "Europe/Berlin"
    assert PL.reference_timezone(None) == "UTC"


def test_local_now_reads_the_clock_of_that_zone():
    """The whole point: 'local time at that location' must be local *there*."""
    berlin = PL.local_now("Europe/Berlin")
    tokyo = PL.local_now("Asia/Tokyo")
    assert berlin.tzinfo is None and tokyo.tzinfo is None      # drops into a widget
    offset = (tokyo - berlin).total_seconds() / 3600
    assert 6.9 < offset < 8.1                                   # Tokyo is 7–8 h ahead


def test_local_hour_is_floored():
    h = PL.local_hour("Europe/Berlin")
    assert isinstance(h, dtime) and h.minute == 0 and h.second == 0
    assert h.hour == datetime.now(ZoneInfo("Europe/Berlin")).hour


def test_local_now_does_not_use_the_server_clock(monkeypatch):
    """The bug this replaced: datetime.now() on a UTC server showed 14:00 to a
    visitor whose wall clock said 16:00."""
    monkeypatch.setattr(PL, "browser_timezone", lambda: None)
    utc = PL.local_now("UTC")
    berlin = PL.local_now("Europe/Berlin")
    assert berlin != utc
    assert 0.9 < (berlin - utc).total_seconds() / 3600 < 2.1    # CET/CEST


# ── The autocomplete's decision logic ────────────────────────────────────────
#
# One widget does two jobs: anything typed that is not already on the list is a
# new search, anything on the list is a choice. AppTest cannot drive
# `accept_new_options` (its _widget_state only sends values already present in
# `options`), so the decision is tested here and the wiring in a browser.

VILLINGEN = {"results": [
    {"name": "Villingen", "country": "Germany", "country_code": "DE", "admin1": "Hesse",
     "admin2": "Regierungsbezirk Gießen", "latitude": 50.5045, "longitude": 8.9361,
     "elevation": 163.0, "timezone": "Europe/Berlin", "population": 0},
    {"name": "Villingen-Schwenningen", "country": "Germany", "country_code": "DE",
     "admin1": "Baden-Wurttemberg", "admin2": "Freiburg Region", "latitude": 48.0623,
     "longitude": 8.4936, "elevation": 704.0, "timezone": "Europe/Berlin", "population": 81811},
]}
KOETHEN = {"results": [
    {"name": "Köthen", "country": "Germany", "country_code": "DE", "admin1": "Saxony-Anhalt",
     "admin2": "", "latitude": 51.7519, "longitude": 11.9705, "elevation": 75.0,
     "timezone": "Europe/Berlin", "population": 25716},
]}


@pytest.fixture(autouse=True)
def _fresh_cache():
    PL.search.clear()
    yield
    PL.search.clear()


def _hits(payload):
    with patch.object(W.requests, "get", return_value=_Response(payload)):
        return PL.search("query")


def test_nothing_selected_is_ignored():
    assert PL.interpret(None, [])[0] == "ignore"
    assert PL.interpret("", [])[0] == "ignore"
    assert PL.interpret("   ", [])[0] == "ignore"


def test_typing_a_new_place_becomes_a_search():
    with patch.object(W.requests, "get", return_value=_Response(VILLINGEN)):
        action, hits, note = PL.interpret("villingen", [])
    assert action == "many" and len(hits) == 2
    assert "2 places match" in note


def test_choosing_an_offered_place_applies_it():
    offered = _hits(VILLINGEN)
    label = PL._label(offered[1])
    action, place, _ = PL.interpret(label, offered)
    assert action == "pick"
    assert place["latitude"] == pytest.approx(48.0623)      # the Schwenningen one


def test_a_lone_match_needs_no_second_step():
    with patch.object(W.requests, "get", return_value=_Response(KOETHEN)):
        action, hits, _ = PL.interpret("köthen", [])
    assert action == "single" and len(hits) == 1
    assert hits[0]["latitude"] == pytest.approx(51.7519)


def test_no_match_says_so_without_guessing():
    with patch.object(W.requests, "get", return_value=_Response({"results": []})):
        action, hits, note = PL.interpret("zzzznowhere", [])
    assert action == "none" and hits == []
    assert "No place matched" in note


def test_a_second_search_replaces_the_list_rather_than_reusing_it():
    """The stale-label bug: options were positional indices, so two different
    searches produced an identical options list (0, 1, …) and Streamlit kept
    the previous names on screen — a search for "villingen" showed "Köthen"."""
    first = _hits(VILLINGEN)
    PL.search.clear()
    with patch.object(W.requests, "get", return_value=_Response(KOETHEN)):
        action, second, _ = PL.interpret("köthen", first)
    assert action == "single"
    assert [PL._label(p) for p in first] != [PL._label(p) for p in second]


def test_labels_are_unique_so_a_choice_is_unambiguous():
    offered = _hits(VILLINGEN)
    labels = [PL._label(p) for p in offered]
    assert len(set(labels)) == len(labels)
    assert "Villingen, Regierungsbezirk Gießen, Hesse, Germany" in labels[0]
    assert "Villingen-Schwenningen" in labels[1] and "81,811 people" in labels[1]


def test_local_clock_keeps_the_minutes_the_field_shows():
    """The field says "local time at that location", so it must show the time a
    clock there shows. It used to be floored to the hour while the caption beside
    it printed the minutes — 16:00 next to 16:32, two numbers for one instant."""
    now = PL.local_now("Africa/Tunis")
    shown = PL.local_clock("Africa/Tunis")
    assert (shown.hour, shown.minute) == (now.hour, now.minute)
    assert PL.local_hour("Africa/Tunis") == dtime(now.hour, 0)   # what the API is asked for


def test_the_clock_differs_between_zones_one_hour_apart():
    tunis = PL.local_clock("Africa/Tunis")
    berlin = PL.local_clock("Europe/Berlin")
    assert tunis.minute == berlin.minute
    assert (berlin.hour - tunis.hour) % 24 == 1        # Tunis UTC+1, Berlin UTC+2 in summer


# ── Coordinates → place (the other direction) ────────────────────────────────

REVERSE_PAYLOAD = {
    "locality": "Nebelin", "city": "Karstadt", "principalSubdivision": "Brandenburg",
    "countryName": "Germany", "countryCode": "DE",
    "localityInfo": {"informative": [
        {"name": "Europe", "description": "terrestrial continent"},
        {"name": "Europe/Berlin", "description": "time zone"},
    ]},
}
OCEAN_PAYLOAD = {"locality": "", "city": "", "principalSubdivision": "", "countryName": ""}


def test_a_training_station_is_named_as_one_without_any_lookup():
    """The Laubsdorf mast reverse-geocodes to the municipality Neuhausen/Spree.
    It is our own station, so it is recognised directly — and costs no request."""
    with patch.object(W.requests, "get", side_effect=AssertionError("should not be called")):
        site = PL.identify(51.68718711157154, 14.414256224166728)
    assert site["name"] == "Laubsdorf"
    assert "training station" in site["display"]
    assert site["timezone"] == "Europe/Berlin"


def test_the_other_station_is_recognised_too():
    with patch.object(W.requests, "get", side_effect=AssertionError("should not be called")):
        assert PL.identify(53.11833921379164, 11.746088890223463)["name"] == "Nebelin"


def test_a_point_far_from_a_station_is_reverse_geocoded():
    PL.reverse.clear()
    with patch.object(W.requests, "get", return_value=_Response(REVERSE_PAYLOAD)):
        place = PL.identify(53.5, 11.9)
    assert place["display"] == "Nebelin, Karstadt, Brandenburg, Germany"
    assert place["timezone"] == "Europe/Berlin"
    assert place["latitude"] == pytest.approx(53.5)
    assert place["elevation"] is None          # this answers "where", not "how high"


def test_reverse_returns_nothing_over_open_water():
    PL.reverse.clear()
    with patch.object(W.requests, "get", return_value=_Response(OCEAN_PAYLOAD)):
        assert PL.reverse(0.0, -30.0) is None


def test_reverse_never_breaks_the_page_when_the_lookup_is_down():
    """Best effort only: the coordinates are what the prediction uses."""
    import requests
    PL.reverse.clear()
    with patch.object(W.requests, "get", side_effect=requests.exceptions.ConnectionError("down")):
        assert PL.reverse(52.5, 13.4) is None


def test_lookups_are_rounded_so_the_cache_actually_hits():
    """Nudging a coordinate in the sixth decimal is the same 110 m square."""
    PL.reverse.clear()
    calls = []

    def _get(url, params=None, timeout=None):
        calls.append(params)
        return _Response(REVERSE_PAYLOAD)

    with patch.object(W.requests, "get", side_effect=_get):
        PL.identify(53.500001, 11.900001)
        PL.identify(53.500002, 11.900002)
    assert len(calls) == 1
    assert calls[0]["latitude"] == pytest.approx(53.5)


# ── The timezone dropdown ────────────────────────────────────────────────────

def test_timezone_choices_lists_the_iana_database():
    opts = PL.timezone_choices("Europe/Berlin")
    assert len(opts) > 400
    assert "Europe/Berlin" in opts and "Africa/Tunis" in opts and "Asia/Tokyo" in opts
    assert opts == sorted(opts)


def test_timezone_choices_keeps_a_zone_it_does_not_recognise():
    """A selectbox raises when session state holds a value outside its options,
    and reverse lookups have returned zones like Etc/GMT+2 over open water."""
    opts = PL.timezone_choices("Etc/GMT+2")
    assert "Etc/GMT+2" in opts
    odd = PL.timezone_choices("Somewhere/Invented")
    assert odd[0] == "Somewhere/Invented"


def test_timezone_choices_without_a_current_value():
    opts = PL.timezone_choices(None)
    assert len(opts) > 400 and "UTC" in opts
