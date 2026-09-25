"""The vectorised batch feature builder must agree with the single-row one.

Dataset Upload scores whole files through ``compute_features_batch``; Normal
and Expert Mode go row by row through ``compute_features``. If the two ever
drift, the same weather would give two different PAR values depending on the
page — so the batch path is held to the single-row path to 1e-6 on every one of
the 22 features, across seasons, hours, hemispheres and DST transitions.
"""
import time

import numpy as np
import pandas as pd
import pytest

from _common import MODEL_FEATURES
from core import features as F

SITES = np.array([
    (51.687, 14.414, 0.0),       # Laubsdorf
    (53.118, 11.746, 0.0),       # Nebelin
    (-23.55, -46.63, 760.0),     # São Paulo — southern hemisphere, altitude
    (35.68, 139.65, 40.0),       # Tokyo
])
WEATHER_KEYS = ["GHI_RC_01", "Temp_WS", "RH_WS", "DWP_WS", "WS_WS", "WD_WS",
                "PREC_INT_WS", "PREC_DIFF_WS", "Temp_RC_01"]


def _frame(n: int = 150, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2024-01-01", "2025-12-31 23:00", freq="h")
    stamps = pd.to_datetime(rng.choice(hours.values, n))
    # DST edges and solstices on top of the random draw
    stamps = stamps.append(pd.to_datetime(["2025-03-30 02:30", "2025-10-26 02:30",
                                           "2025-06-21 12:00", "2025-12-21 00:00"]))
    m = len(stamps)
    pick = rng.integers(0, len(SITES), m)
    df = pd.DataFrame({
        "timestamp": stamps, "lat": SITES[pick, 0], "lon": SITES[pick, 1], "alt": SITES[pick, 2],
        "GHI_RC_01": rng.uniform(0, 1000, m), "Temp_WS": rng.uniform(-5, 35, m),
        "RH_WS": rng.uniform(20, 100, m), "DWP_WS": rng.uniform(-10, 20, m),
        "WS_WS": rng.uniform(0, 10, m), "WD_WS": rng.uniform(0, 360, m),
        "PREC_INT_WS": rng.choice([0.0, 0.0, 0.5, 2.0], m),
        "PREC_DIFF_WS": rng.choice([0.0, 0.3], m),
        "Temp_RC_01": rng.uniform(0, 50, m),
    })
    df.loc[::9, "GHI_RC_01"] = 0.0          # dark rows
    df.loc[::13, "Temp_RC_01"] = np.nan     # NOCT fallback rows
    return df


def _single_rows(df: pd.DataFrame, tz: str):
    rows, days = [], []
    for _, r in df.iterrows():
        feats, day = F.compute_features(r["lat"], r["lon"], r["alt"], r["timestamp"],
                                        {k: r[k] for k in WEATHER_KEYS}, tz)
        rows.append(feats.iloc[0])
        days.append(day)
    return pd.DataFrame(rows).reset_index(drop=True), np.asarray(days, dtype=bool)


@pytest.fixture(scope="module")
def both():
    df = _frame()
    batch, batch_day = F.compute_features_batch(df, "Europe/Berlin")
    single, single_day = _single_rows(df, "Europe/Berlin")
    return df, batch.reset_index(drop=True), batch_day, single, single_day


def test_daytime_flag_is_identical(both):
    _, _, batch_day, _, single_day = both
    assert (batch_day == single_day).all()


@pytest.mark.parametrize("feature", F.FEATURE_COLUMNS)
def test_every_feature_matches_the_single_row_path(both, feature):
    _, batch, _, single, _ = both
    np.testing.assert_allclose(batch[feature].to_numpy(dtype="float64"),
                               single[feature].to_numpy(dtype="float64"),
                               atol=1e-6, rtol=0, err_msg=feature)


def test_batch_covers_all_model_features_in_order(both):
    _, batch, _, _, _ = both
    assert list(batch.columns) == F.FEATURE_COLUMNS
    assert all(c in batch.columns for c in MODEL_FEATURES)


def test_batch_is_at_least_twenty_times_faster_than_the_loop():
    """The old implementation looped iterrows → compute_features (≈5 ms/row);
    a 100k-row upload took eight minutes. Measured speed-up is ~150×; 20× is
    the floor this test insists on so a regression to a loop is caught."""
    df = _frame(n=400, seed=2)
    t0 = time.perf_counter(); F.compute_features_batch(df, "Europe/Berlin"); t_batch = time.perf_counter() - t0
    t0 = time.perf_counter(); _single_rows(df.head(80), "Europe/Berlin"); t_loop = (time.perf_counter() - t0) * 5
    assert t_batch * 20 < t_loop, f"batch {t_batch:.3f}s vs loop-equivalent {t_loop:.3f}s"


def test_precomputed_wind_vector_is_used_as_is():
    """Training averaged sin/cos over the minute (resultant length ≤ 1). When the
    upload pipeline hands those in, they must not be recomputed from a bearing."""
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-06-15 12:00")], "lat": [51.687], "lon": [14.414],
                       "alt": [0.0], "GHI_RC_01": [600.0], "WD_WS": [90.0],
                       "wind_sin": [0.3], "wind_cos": [0.4]})
    feats, _ = F.compute_features_batch(df, "Europe/Berlin")
    assert feats["wind_sin"].iloc[0] == pytest.approx(0.3)
    assert feats["wind_cos"].iloc[0] == pytest.approx(0.4)


def test_bearing_is_encoded_when_no_vector_is_given():
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-06-15 12:00")], "lat": [51.687], "lon": [14.414],
                       "alt": [0.0], "GHI_RC_01": [600.0], "WD_WS": [90.0]})
    feats, _ = F.compute_features_batch(df, "Europe/Berlin")
    assert feats["wind_sin"].iloc[0] == pytest.approx(1.0)
    assert feats["wind_cos"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_reference_cell_temperature_fallback_chain():
    """measured Temp_RC_01 → merged Temp_RC → NOCT model, as in notebook 4 + the app."""
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-06-15 12:00")] * 3, "lat": [51.687] * 3, "lon": [14.414] * 3,
        "alt": [0.0] * 3, "GHI_RC_01": [800.0] * 3, "Temp_WS": [20.0] * 3,
        "Temp_RC_01": [30.0, np.nan, np.nan], "Temp_RC": [25.0, 25.0, np.nan],
    })
    feats, _ = F.compute_features_batch(df, "Europe/Berlin")
    assert feats["Temp_RC_merged"].tolist() == pytest.approx([30.0, 25.0, 20.0 + F.NOCT_SLOPE * 800.0])


def test_ambiguous_dst_hour_is_treated_as_night_not_garbage():
    """02:30 on the autumn fall-back night exists twice. Both paths make it NaT →
    no geometry → night. Nothing is guessed."""
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-10-26 02:30")], "lat": [51.687], "lon": [14.414],
                       "alt": [0.0], "GHI_RC_01": [100.0]})
    feats, day = F.compute_features_batch(df, "Europe/Berlin")
    assert not day[0]
    assert feats["airmass"].iloc[0] == F.AIRMASS_NIGHT
    assert feats["clearness_kt"].iloc[0] == 0.0


def test_empty_input_gives_empty_output_with_the_right_columns():
    feats, day = F.compute_features_batch(pd.DataFrame(columns=["timestamp", "lat", "lon", "alt", "GHI_RC_01"]))
    assert list(feats.columns) == F.FEATURE_COLUMNS
    assert len(feats) == 0 and len(day) == 0


def test_missing_weather_columns_take_the_documented_defaults():
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-06-15 12:00")], "lat": [51.687], "lon": [14.414],
                       "alt": [0.0], "GHI_RC_01": [600.0]})
    feats, _ = F.compute_features_batch(df, "Europe/Berlin")
    for k, v in F.DEFAULTS.items():
        assert feats[k].iloc[0] == v, k
