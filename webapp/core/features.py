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


def compute_features_batch(
    df: pd.DataFrame,
    tz_str: str = "UTC",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Vectorized batch version of compute_features, mirroring notebook 4_4_feature_engineering_All_Files.

    All pvlib solar calculations and feature derivations are computed in one pass
    over the full DataFrame instead of row by row — 100-1000x faster for large datasets.

    df must have columns: timestamp, lat, lon, alt,
                          GHI_RC_01, Temp_WS, RH_WS, DWP_WS, WS_WS, WD_WS, PREC_INT_WS

    Returns (features_df, is_daytime_array).
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame(), np.array([], dtype=bool)

    # Single pvlib Location using median coordinates (constant in webapp uploads)
    lat = float(df["lat"].median())
    lon = float(df["lon"].median())
    alt = float(df["alt"].median())

    # Build tz-aware DatetimeIndex for all rows at once
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz_str, ambiguous="NaT", nonexistent="shift_forward")
    else:
        ts = ts.dt.tz_convert(tz_str)
    idx = pd.DatetimeIndex(ts)

    location = pvlib.location.Location(latitude=lat, longitude=lon, altitude=alt, tz=tz_str)

    # One call for all timestamps
    solar_pos = location.get_solarposition(idx)
    zenith_arr    = solar_pos["apparent_zenith"].values.astype(float)
    elevation_arr = solar_pos["elevation"].values.astype(float)
    is_daytime    = elevation_arr > 0.5

    # Airmass vectorized
    am_df = location.get_airmass(solar_position=solar_pos)
    airmass_arr = am_df["airmass_relative"].fillna(37.9).values.astype(float)
    airmass_arr = np.clip(airmass_arr, 1.0, 37.9)
    airmass_arr = np.where(is_daytime, airmass_arr, 37.9)

    # GHI and extra radiation
    ghi_arr    = np.maximum(0.0, df["GHI_RC_01"].fillna(0.0).values.astype(float))
    doy_arr    = idx.dayofyear
    dni_extra  = pvlib.irradiance.get_extra_radiation(doy_arr).values.astype(float)

    # Clearness index — vectorized on daytime+positive-GHI rows
    kt_arr   = np.zeros(n, dtype=float)
    dni_arr  = np.zeros(n, dtype=float)
    day_mask = is_daytime & (ghi_arr > 0)

    if day_mask.any():
        kt_arr[day_mask] = pvlib.irradiance.clearness_index(
            ghi=ghi_arr[day_mask],
            solar_zenith=zenith_arr[day_mask],
            extra_radiation=dni_extra[day_mask],
            min_cos_zenith=0.065,
            max_clearness_index=1.0,
        )
        try:
            disc_out = pvlib.irradiance.disc(
                ghi=pd.Series(ghi_arr[day_mask]),
                solar_zenith=pd.Series(zenith_arr[day_mask]),
                datetime_or_doy=idx[day_mask],
            )
            dni_arr[day_mask] = disc_out["dni"].fillna(0.0).values
        except Exception:
            pass

    # Weather columns
    temp_ws  = df["Temp_WS"].fillna(15.0).values.astype(float)
    rh_ws    = df["RH_WS"].fillna(60.0).values.astype(float)
    dwp_ws   = df["DWP_WS"].fillna(10.0).values.astype(float)
    ws_ws    = df["WS_WS"].fillna(2.0).values.astype(float)
    wd_ws    = df["WD_WS"].fillna(180.0).values.astype(float)
    prec_int = np.maximum(0.0, df["PREC_INT_WS"].fillna(0.0).values.astype(float))

    # Derived features
    temp_rc        = temp_ws + (45.0 - 20.0) / 800.0 * ghi_arr
    wind_sin       = np.sin(2 * np.pi * wd_ws / 360.0)
    wind_cos       = np.cos(2 * np.pi * wd_ws / 360.0)
    is_raining     = (prec_int > 0.1).astype(float)
    temp_diff      = temp_ws - temp_rc
    dew_depression = temp_ws - dwp_ws

    features = pd.DataFrame({
        "GHI_RC_01":        ghi_arr,
        "Temp_WS":          temp_ws,
        "RH_WS":            rh_ws,
        "DWP_WS":           dwp_ws,
        "WS_WS":            ws_ws,
        "WD_WS":            wd_ws,
        "PREC_INT_WS":      prec_int,
        "PREC_DIFF_WS":     prec_int,
        "PREC_WS":          prec_int,
        "Temp_RC_merged":   temp_rc,
        "Temp_RC_01":       temp_rc,
        "zenith":           zenith_arr,
        "elevation":        elevation_arr,
        "airmass":          airmass_arr,
        "clearness_kt":     kt_arr,
        "dni":              dni_arr,
        "wind_sin":         wind_sin,
        "wind_cos":         wind_cos,
        "is_raining":       is_raining,
        "GHI_rolling_5min": ghi_arr,
        "temp_diff":        temp_diff,
        "dew_depression":   dew_depression,
    }, index=df.index)

    return features, is_daytime
