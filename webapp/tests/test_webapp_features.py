"""The web app's feature pipeline must produce exactly what the model was trained on."""
import math

import numpy as np
import pytest

from _common import MODEL_FEATURES
from core import features as F
from core import predict as P

LAUBSDORF = dict(lat=51.68718711157154, lon=14.414256224166728, alt=0)
WEATHER = {
    "GHI_RC_01": 650.0, "Temp_WS": 22.0, "RH_WS": 55.0, "DWP_WS": 12.0,
    "WS_WS": 3.0, "WD_WS": 225.0, "PREC_INT_WS": 0.0, "PREC_DIFF_WS": 0.0,
    "Temp_RC_01": 35.0,
}


@pytest.fixture(scope="module")
def noon():
    df, is_day = F.compute_features(**LAUBSDORF, dt_str="2024-06-15 12:00:00",
                                    weather=WEATHER, tz_str="Europe/Berlin")
    return df, is_day


def test_all_model_features_are_produced(noon):
    df, _ = noon
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    assert not missing, f"compute_features() is missing model feature(s): {missing}"


def test_model_features_are_finite(noon):
    df, _ = noon
    values = df.loc[:, MODEL_FEATURES].to_numpy(dtype="float64")
    assert np.isfinite(values).all()


def test_summer_noon_is_daytime_with_high_sun(noon):
    df, is_day = noon
    assert bool(is_day)
    assert df["elevation"].iloc[0] > 45          # sun high above Laubsdorf in mid-June
    assert 0 < df["zenith"].iloc[0] < 45
    assert 1.0 <= df["airmass"].iloc[0] < 1.6


def test_wind_encoding_lies_on_unit_circle(noon):
    df, _ = noon
    s, c = df["wind_sin"].iloc[0], df["wind_cos"].iloc[0]
    assert math.isclose(s * s + c * c, 1.0, abs_tol=1e-9)
    # 225 deg -> south-west: both components negative
    assert s < 0 and c < 0


def test_clearness_index_is_bounded(noon):
    df, _ = noon
    assert 0.0 <= df["clearness_kt"].iloc[0] <= 1.0


def test_mccree_baseline_factor():
    assert math.isclose(P.mccree_estimate(1000.0), 2060.0, rel_tol=1e-9)
    assert P.mccree_estimate(-5.0) == 0.0


def test_fallback_feature_list_matches_training():
    assert list(P._FALLBACK_FEATURES) == MODEL_FEATURES


def test_missing_feature_raises_instead_of_being_zero_filled(noon):
    """A silently zero-filled feature yields a confident but wrong prediction."""
    df, _ = noon
    with pytest.raises(KeyError, match="dni"):
        P.predict_par(df.drop(columns=["dni"]))


# ── A measured 0.0 must survive: `value or default` would replace it ──────────

@pytest.mark.parametrize("field", ["Temp_WS", "RH_WS", "DWP_WS", "WS_WS"])
def test_zero_reading_is_not_replaced_by_default(field):
    """0 °C, 0 % RH and 0 m/s are real measurements, not missing values."""
    df, _ = F.compute_features(**LAUBSDORF, dt_str="2025-01-15 12:00:00",
                               weather={**WEATHER, field: 0.0}, tz_str="Europe/Berlin")
    assert df[field].iloc[0] == 0.0


def test_north_wind_stays_north():
    """WD_WS = 0° is north; the old `or 180.0` turned it into a south wind."""
    df, _ = F.compute_features(**LAUBSDORF, dt_str="2025-01-15 12:00:00",
                               weather={**WEATHER, "WD_WS": 0.0}, tz_str="Europe/Berlin")
    assert df["wind_sin"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert df["wind_cos"].iloc[0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_genuinely_missing_value_still_falls_back(missing):
    df, _ = F.compute_features(**LAUBSDORF, dt_str="2025-01-15 12:00:00",
                               weather={**WEATHER, "Temp_WS": missing}, tz_str="Europe/Berlin")
    assert df["Temp_WS"].iloc[0] == 15.0          # documented default


def test_batch_path_preserves_zeros():
    """Page 3 (Dataset Upload) goes through compute_features_batch."""
    import pandas as pd

    raw = pd.DataFrame({
        "timestamp": pd.date_range("2025-06-15 10:00", periods=3, freq="h"),
        "lat": LAUBSDORF["lat"], "lon": LAUBSDORF["lon"], "alt": 0,
        "GHI_RC_01": [300.0, 500.0, 650.0], "Temp_WS": [0.0, 12.0, 20.0],
        "RH_WS": [80.0, 75.0, 55.0], "DWP_WS": [0.0, 8.0, 10.0],
        "WS_WS": [0.0, 2.0, 3.0], "WD_WS": [0.0, 90.0, 180.0],
    })
    feats, is_day = F.compute_features_batch(raw, "Europe/Berlin")
    assert len(feats) == len(raw) == len(is_day)
    assert feats["Temp_WS"].iloc[0] == 0.0
    assert feats["WS_WS"].iloc[0] == 0.0
    assert all(c in feats.columns for c in MODEL_FEATURES)
