"""Out-of-domain inputs must be flagged, in-domain ones left alone.

The model knows two Brandenburg stations, 2024–2025. core.domain compares a
request against that — distance to the nearest station, and every model
feature against its training range — and phrases the result for a visitor.
"""
import numpy as np
import pandas as pd
import pytest

from core import domain as DM
from core import features as F
from core import predict as P

MILD = {"GHI_RC_01": 650.0, "Temp_WS": 22.0, "RH_WS": 55.0, "DWP_WS": 12.0,
        "WS_WS": 3.0, "WD_WS": 225.0, "PREC_INT_WS": 0.0}


# ── Location ──────────────────────────────────────────────────────────────────

def test_training_station_is_inside_the_domain():
    c = DM.check_location(51.687, 14.414)
    assert c.in_domain and c.nearest == "Laubsdorf" and c.distance_km < 1.0


def test_berlin_is_inside_nairobi_is_not():
    assert DM.check_location(52.52, 13.405).in_domain
    far = DM.check_location(-1.29, 36.82)
    assert not far.in_domain and far.distance_km > 6000


def test_haversine_matches_a_known_distance():
    assert DM.haversine_km(52.52, 13.405, 48.8566, 2.3522) == pytest.approx(878, abs=10)   # Berlin–Paris


# ── Features ──────────────────────────────────────────────────────────────────

def test_a_mild_spring_day_at_laubsdorf_raises_no_flag():
    feats, day = F.compute_features(51.687, 14.414, 0.0, "2025-05-15 12:00", MILD, "Europe/Berlin")
    assert day
    assert DM.check_features(feats) == []


def test_cairo_midsummer_noon_leaves_the_training_sun_geometry():
    """The sun never gets that high over Brandenburg: zenith below the training minimum."""
    feats, _ = F.compute_features(30.04, 31.24, 23.0, "2025-06-21 12:30",
                                  {**MILD, "GHI_RC_01": 950.0, "Temp_WS": 30.0}, "Africa/Cairo")
    flagged = {c.feature for c in DM.check_features(feats)}
    assert "zenith" in flagged or "elevation" in flagged


def test_hot_afternoon_flags_temperature_and_says_what_the_model_sees():
    feats, _ = F.compute_features(51.687, 14.414, 0.0, "2025-07-20 15:00",
                                  {**MILD, "Temp_WS": 36.0, "DWP_WS": 12.0}, "Europe/Berlin")
    checks = DM.check_features(feats)
    temp = next(c for c in checks if c.feature == "Temp_WS")
    assert temp.value == 36.0 and temp.high == pytest.approx(P.training_bounds()["Temp_WS"][1])
    sentence = next(s for s in DM.describe(None, checks) if "temperature" in s)
    assert "above the training range" in sentence and "treats it as" in sentence


def test_describe_location_sentence_names_distance_and_station():
    notes = DM.describe(DM.LocationCheck("Nebelin", 812.0, False), [])
    assert len(notes) == 1
    assert "812 km" in notes[0] and "Nebelin" in notes[0] and "extrapolation" in notes[0]


def test_describe_is_silent_when_everything_is_fine():
    assert DM.describe(DM.LocationCheck("Laubsdorf", 0.3, True), []) == []


def test_quiet_channels_are_not_flagged():
    """Heavy rain is ordinary weather, not a reason to doubt the prediction."""
    row = pd.DataFrame([{**P.train_medians(), "PREC_INT_WS": 3.5, "wind_sin": 1.0}])
    assert DM.check_features(row) == []


# ── Batch ─────────────────────────────────────────────────────────────────────

def test_batch_check_reports_the_share_of_rows_outside():
    base = P.train_medians()
    df = pd.DataFrame([dict(base) for _ in range(4)])
    df["Temp_WS"] = [10.0, 40.0, 50.0, 20.0]
    out = DM.check_features_batch(df)
    assert list(out["feature"]) == ["Temp_WS"]
    row = out.iloc[0]
    assert row["n_out"] == 2 and row["pct_out"] == 50.0
    assert row["min"] == 10.0 and row["max"] == 50.0
    assert row["label"] == "air temperature" and row["unit"] == "°C"


def test_batch_check_on_clean_data_is_empty():
    df = pd.DataFrame([dict(P.train_medians()) for _ in range(3)])
    assert DM.check_features_batch(df).empty
    assert DM.check_features_batch(df.iloc[0:0]).empty


# ── The error figure must not outrun what it measures ─────────────────────────
# MAE is the model's error given the weather it is handed. In Normal Mode that
# weather is Open-Meteo's, and PAR follows GHI almost one for one, so a forecast
# day carries the forecast's error on top. The app used to print the same
# "typical error ± 29" for a measured day and for one 15 days out.

def test_error_note_states_what_it_covers():
    note = DM.error_note(29.4)
    assert "29" in note
    assert "for the weather shown" in note


def test_error_note_warns_when_a_forecast_is_involved():
    for horizon in (1, 7, 15):
        note = DM.error_note(29.4, horizon)
        assert "the forecast adds its own error on top" in note, horizon


def test_error_note_is_silent_for_measured_and_past_days():
    """horizon 0 is today, None is a past date or hand-entered readings."""
    for horizon in (0, None):
        assert "forecast" not in DM.error_note(29.4, horizon), horizon
