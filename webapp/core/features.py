"""
core/features.py
────────────────
Feature engineering pipeline that mirrors the training notebooks
(4_4_feature_engineering_All_Files.ipynb).

Main entry point:
    compute_features(lat, lon, alt, dt_str, weather, tz_str)
        → (pd.DataFrame of all features, bool is_daytime)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
from datetime import datetime


def compute_features(
    lat: float,
    lon: float,
    alt: float,
    dt_str: str | datetime,
    weather: dict,
    tz_str: str = "UTC",
) -> tuple[pd.DataFrame, bool]:
    """
    Build the complete feature vector for a single observation.

    Parameters
    ----------
    lat, lon, alt : float
        Location coordinates (WGS84) and altitude in metres.
    dt_str : str or datetime
        Local date-time of the observation (naive = localised to tz_str).
    weather : dict
        Raw weather values (keys: GHI_RC_01, Temp_WS, RH_WS, DWP_WS,
        WS_WS, WD_WS, PREC_INT_WS).
    tz_str : str
        IANA timezone string, e.g. "Europe/Berlin".

    Returns
    -------
    (features_df, is_daytime)
        features_df : pd.DataFrame with one row containing all 22 features.
        is_daytime  : True when solar elevation > 0.
    """

    # ── 1. Timezone-aware timestamp ─────────────────────────────────────────
    try:
        ts = pd.DatetimeIndex([pd.Timestamp(dt_str)])
        if ts.tz is None:
            ts = ts.tz_localize(tz_str, ambiguous="NaT",
                                nonexistent="shift_forward")
    except Exception:
        ts = pd.DatetimeIndex([pd.Timestamp(dt_str, tz="UTC")])

    # ── 2. pvlib solar position ─────────────────────────────────────────────
    location = pvlib.location.Location(
        latitude=float(lat),
        longitude=float(lon),
        altitude=float(alt),
        tz=tz_str,
    )
    solar_pos = location.get_solarposition(ts)

    zenith    = float(solar_pos["apparent_zenith"].iloc[0])
    elevation = float(solar_pos["elevation"].iloc[0])
    is_daytime = elevation > 0.5           # small margin avoids twilight edge cases

    # ── 3. Airmass ─────────────────────────────────────────────────────────
    if is_daytime:
        am_df  = location.get_airmass(solar_position=solar_pos)
        airmass = float(am_df["airmass_relative"].fillna(37.9).iloc[0])
        airmass = min(max(airmass, 1.0), 37.9)
    else:
        airmass = 37.9

    # ── 4. GHI + clearness index + DNI ─────────────────────────────────────
    ghi = max(0.0, float(weather.get("GHI_RC_01", 0.0) or 0.0))

    doy       = pd.Timestamp(dt_str).dayofyear
    dni_extra = float(pvlib.irradiance.get_extra_radiation(doy))

    if is_daytime and ghi > 0:
        kt = float(pvlib.irradiance.clearness_index(
            ghi=ghi,
            solar_zenith=zenith,
            extra_radiation=dni_extra,
            min_cos_zenith=0.065,
            max_clearness_index=1.0,
        ))
        try:
            disc_out = pvlib.irradiance.disc(
                ghi=ghi, solar_zenith=zenith, datetime_or_doy=ts
            )
            dni = float(disc_out["dni"].fillna(0.0).iloc[0])
        except Exception:
            dni = 0.0
    else:
        kt  = 0.0
        dni = 0.0

    # ── 5. Raw weather scalars ──────────────────────────────────────────────
    temp_ws   = float(weather.get("Temp_WS",     15.0) or 15.0)
    rh_ws     = float(weather.get("RH_WS",       60.0) or 60.0)
    dwp_ws    = float(weather.get("DWP_WS",      10.0) or 10.0)
    ws_ws     = float(weather.get("WS_WS",        2.0) or  2.0)
    wd_ws     = float(weather.get("WD_WS",       180.0) or 180.0)
    prec_int  = max(0.0, float(weather.get("PREC_INT_WS", 0.0) or 0.0))

    # ── 6. Reference cell temperature (NOCT approximation) ─────────────────
    # Tc = Ta + (NOCT - 20) / 800 * GHI   (NOCT ≈ 45 °C for typical c-Si cell)
    temp_rc = temp_ws + (45.0 - 20.0) / 800.0 * ghi

    # ── 7. Cyclical wind-direction encoding ─────────────────────────────────
    wind_sin = float(np.sin(2 * np.pi * wd_ws / 360.0))
    wind_cos = float(np.cos(2 * np.pi * wd_ws / 360.0))

    # ── 8. Engineered features ──────────────────────────────────────────────
    is_raining       = 1 if prec_int > 0.1 else 0
    ghi_rolling_5min = ghi           # at hourly API resolution ≈ GHI
    temp_diff        = temp_ws - temp_rc
    dew_depression   = temp_ws - dwp_ws

    # ── 9. Assemble DataFrame ───────────────────────────────────────────────
    features = pd.DataFrame([{
        # Raw sensor / weather (8)
        "GHI_RC_01":        ghi,
        "Temp_WS":          temp_ws,
        "RH_WS":            rh_ws,
        "DWP_WS":           dwp_ws,
        "WS_WS":            ws_ws,
        "WD_WS":            wd_ws,
        "PREC_INT_WS":      prec_int,
        "PREC_DIFF_WS":     prec_int,   # same at hourly resolution
        "PREC_WS":          prec_int,
        "Temp_RC_merged":   temp_rc,
        "Temp_RC_01":       temp_rc,
        # pvlib solar (5)
        "zenith":           zenith,
        "elevation":        elevation,
        "airmass":          airmass,
        "clearness_kt":     kt,
        "dni":              dni,
        # Wind cyclical (2)
        "wind_sin":         wind_sin,
        "wind_cos":         wind_cos,
        # Engineered (4)
        "is_raining":       is_raining,
        "GHI_rolling_5min": ghi_rolling_5min,
        "temp_diff":        temp_diff,
        "dew_depression":   dew_depression,
    }])

    return features, is_daytime
