"""
core/features.py
────────────────
Feature engineering pipeline that mirrors the training notebooks
(4_feature_engineering_All_Files.ipynb).

Two entry points, one contract:

    compute_features(lat, lon, alt, dt_str, weather, tz_str)
        → (pd.DataFrame with one row of all 22 features, bool is_daytime)
        Used by Normal Mode and Expert Mode — one observation at a time.

    compute_features_batch(df, tz_str)
        → (pd.DataFrame with one row per input row, np.ndarray[bool] is_daytime)
        Used by Dataset Upload. A single vectorised pvlib pass over the whole
        table; tests/test_webapp_batch_features.py asserts it agrees with the
        single-row function to 1e-6 on every feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
from datetime import datetime

# ── Shared constants ─────────────────────────────────────────────────────────
#
# Defaults used only when a weather value is genuinely absent (None / NaN).
# They are the training-time conventions, not "typical" values for a location.
DEFAULTS = {
    "Temp_WS":     15.0,   # °C
    "RH_WS":       60.0,   # %
    "DWP_WS":      10.0,   # °C
    "WS_WS":        2.0,   # m/s
    "WD_WS":      180.0,   # °
    "PREC_INT_WS":  0.0,   # mm/h
}
AIRMASS_NIGHT        = 37.9   # relative airmass at the horizon — used when the sun is down
DAYTIME_ELEVATION    = 0.5    # degrees; small margin keeps twilight out of "daytime"
NOCT_SLOPE           = (45.0 - 20.0) / 800.0   # NOCT cell-temperature model, °C per W/m²

FEATURE_COLUMNS = [
    # Raw sensor / weather
    "GHI_RC_01", "Temp_WS", "RH_WS", "DWP_WS", "WS_WS", "WD_WS",
    "PREC_INT_WS", "PREC_DIFF_WS", "PREC_WS", "Temp_RC_merged", "Temp_RC_01",
    # pvlib solar
    "zenith", "elevation", "airmass", "clearness_kt", "dni",
    # Wind cyclical
    "wind_sin", "wind_cos",
    # Engineered
    "is_raining", "GHI_rolling_5min", "temp_diff", "dew_depression",
]


def _num(value, default: float) -> float:
    """Coerce `value` to float, falling back to `default` only when it is
    genuinely absent (None / NaN / unparseable).

    Deliberately NOT ``float(value or default)``: in Python ``0.0 or 15.0``
    evaluates to ``15.0``, which would silently turn a real 0 °C reading into
    15 °C, or a north wind (0°) into a south wind (180°).
    """
    if value is None:
        return float(default)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if not np.isfinite(v) else v


def _column(df: pd.DataFrame, name: str, default: float | np.ndarray) -> np.ndarray:
    """Vectorised twin of ``_num``: a float64 array with NaN replaced by `default`.

    `default` may itself be an array (used for PREC_DIFF_WS, whose fallback is
    the row's own PREC_INT_WS, and for the missing-column case)."""
    if name not in df.columns:
        return np.broadcast_to(np.asarray(default, dtype="float64"), (len(df),)).copy()
    v = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    return np.where(np.isfinite(v), v, default)


def localize_timestamps(values, tz_str: str) -> pd.DatetimeIndex:
    """Local wall time → tz-aware index, with the same DST policy as compute_features().

    Naive stamps are *localised* (they are read as wall time in `tz_str`);
    aware stamps are *converted*. The ambiguous hour on the autumn fall-back
    night becomes NaT (→ no solar geometry, → treated as night), exactly like
    the single-row path, so the two paths never disagree on a row.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    if idx.tz is None:
        return idx.tz_localize(tz_str, ambiguous="NaT", nonexistent="shift_forward")
    return idx.tz_convert(tz_str)


# ═════════════════════════════════════════════════════════════════════════════
#  Batch path — Dataset Upload
# ═════════════════════════════════════════════════════════════════════════════

def compute_features_batch(
    df: pd.DataFrame,
    tz_str: str = "UTC",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the feature table for every row of `df` in one vectorised pass.

    Expected columns: ``timestamp`` (naive local wall time or tz-aware),
    ``lat``, ``lon``, ``alt`` (per row — a moving sensor is fine), ``GHI_RC_01``
    and any of the weather columns in ``DEFAULTS`` / ``Temp_RC_01`` / ``Temp_RC``.

    Wind: if ``wind_sin`` / ``wind_cos`` columns are present they are used as
    they are — that is how the training data was built (sin/cos averaged over
    the minute → a resultant vector whose length is ≤ 1). Otherwise the bearing
    in ``WD_WS`` is encoded.
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS, dtype="float64"), np.zeros(0, dtype=bool)

    # ── 1. Time and place ───────────────────────────────────────────────────
    ts    = localize_timestamps(df["timestamp"], tz_str)
    valid = ~np.asarray(pd.isna(ts), dtype=bool)
    lat   = _column(df, "lat", 0.0)
    lon   = _column(df, "lon", 0.0)
    alt   = _column(df, "alt", 0.0)
    # day-of-year from the *wall-clock* stamp, as the single-row path does
    doy   = pd.DatetimeIndex(pd.to_datetime(df["timestamp"])).dayofyear.to_numpy()

    # ── 2. Solar position — pvlib accepts per-row coordinate arrays ─────────
    zenith    = np.full(n, np.nan)
    elevation = np.full(n, np.nan)
    if valid.any():
        solpos = pvlib.solarposition.get_solarposition(
            ts[valid], lat[valid], lon[valid], altitude=alt[valid],
        )
        zenith[valid]    = solpos["apparent_zenith"].to_numpy()
        elevation[valid] = solpos["elevation"].to_numpy()

    is_daytime = np.where(np.isfinite(elevation), elevation > DAYTIME_ELEVATION, False)

    # ── 3. Airmass (Kasten & Young, pvlib's default) ────────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        airmass = np.asarray(
            pvlib.atmosphere.get_relative_airmass(zenith, model="kastenyoung1989"),
            dtype="float64",
        )
    airmass = np.where(np.isfinite(airmass), airmass, AIRMASS_NIGHT)
    airmass = np.clip(airmass, 1.0, AIRMASS_NIGHT)
    airmass = np.where(is_daytime, airmass, AIRMASS_NIGHT)

    # ── 4. GHI, clearness index, DNI ────────────────────────────────────────
    ghi = np.maximum(0.0, _column(df, "GHI_RC_01", 0.0))
    dni_extra = np.asarray(pvlib.irradiance.get_extra_radiation(doy), dtype="float64")
    lit = is_daytime & (ghi > 0)

    kt  = np.zeros(n)
    dni = np.zeros(n)
    if lit.any():
        with np.errstate(invalid="ignore", divide="ignore"):
            kt_lit = pvlib.irradiance.clearness_index(
                ghi=ghi[lit], solar_zenith=zenith[lit], extra_radiation=dni_extra[lit],
                min_cos_zenith=0.065, max_clearness_index=1.0,
            )
            # The tz-aware index, not the wall-clock day-of-year: pvlib derives
            # disc's day-of-year in UTC from an index, and both the notebook and
            # compute_features() pass the index. An integer doy drifts dni by up
            # to 0.4 W/m2 on rows that straddle midnight UTC.
            dni_lit = pvlib.irradiance.disc(
                ghi=ghi[lit], solar_zenith=zenith[lit], datetime_or_doy=ts[lit],
            )["dni"]
        kt[lit]  = np.nan_to_num(np.asarray(kt_lit,  dtype="float64"), nan=0.0)
        dni[lit] = np.nan_to_num(np.asarray(dni_lit, dtype="float64"), nan=0.0)

    # ── 5. Raw weather ──────────────────────────────────────────────────────
    temp_ws   = _column(df, "Temp_WS", DEFAULTS["Temp_WS"])
    rh_ws     = _column(df, "RH_WS",   DEFAULTS["RH_WS"])
    dwp_ws    = _column(df, "DWP_WS",  DEFAULTS["DWP_WS"])
    ws_ws     = _column(df, "WS_WS",   DEFAULTS["WS_WS"])
    wd_ws     = _column(df, "WD_WS",   DEFAULTS["WD_WS"])
    prec_int  = np.maximum(0.0, _column(df, "PREC_INT_WS", DEFAULTS["PREC_INT_WS"]))
    prec_diff = np.maximum(0.0, _column(df, "PREC_DIFF_WS", prec_int))

    # ── 6. Reference-cell temperature: measured → merged sensor → NOCT model ─
    temp_rc = _column(df, "Temp_RC_01", np.nan)
    if "Temp_RC" in df.columns:                       # notebook 4: Temp_RC_01.fillna(Temp_RC)
        temp_rc = np.where(np.isfinite(temp_rc), temp_rc, _column(df, "Temp_RC", np.nan))
    temp_rc = np.where(np.isfinite(temp_rc), temp_rc, temp_ws + NOCT_SLOPE * ghi)

    # ── 7. Wind — resultant vector if supplied, otherwise encode the bearing ─
    if "wind_sin" in df.columns and "wind_cos" in df.columns:
        wind_sin = _column(df, "wind_sin", np.sin(np.deg2rad(wd_ws)))
        wind_cos = _column(df, "wind_cos", np.cos(np.deg2rad(wd_ws)))
    else:
        wind_sin = np.sin(2 * np.pi * wd_ws / 360.0)
        wind_cos = np.cos(2 * np.pi * wd_ws / 360.0)

    # ── 8. Engineered ───────────────────────────────────────────────────────
    is_raining       = (prec_int > 0.1).astype("int64")
    ghi_rolling_5min = _column(df, "GHI_rolling_5min", ghi)
    temp_diff        = temp_ws - temp_rc
    dew_depression   = temp_ws - dwp_ws

    features = pd.DataFrame({
        "GHI_RC_01":        ghi,
        "Temp_WS":          temp_ws,
        "RH_WS":            rh_ws,
        "DWP_WS":           dwp_ws,
        "WS_WS":            ws_ws,
        "WD_WS":            wd_ws,
        "PREC_INT_WS":      prec_int,
        "PREC_DIFF_WS":     prec_diff,
        "PREC_WS":          prec_int,
        "Temp_RC_merged":   temp_rc,
        "Temp_RC_01":       temp_rc,
        "zenith":           zenith,
        "elevation":        elevation,
        "airmass":          airmass,
        "clearness_kt":     kt,
        "dni":              dni,
        "wind_sin":         wind_sin,
        "wind_cos":         wind_cos,
        "is_raining":       is_raining,
        "GHI_rolling_5min": ghi_rolling_5min,
        "temp_diff":        temp_diff,
        "dew_depression":   dew_depression,
    }, index=df.index)

    return features[FEATURE_COLUMNS], np.asarray(is_daytime, dtype=bool)


# ═════════════════════════════════════════════════════════════════════════════
#  Single-observation path — Normal Mode and Expert Mode
# ═════════════════════════════════════════════════════════════════════════════

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
        is_daytime  : True when solar elevation > 0.5°.
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
    is_daytime = elevation > DAYTIME_ELEVATION

    # ── 3. Airmass ─────────────────────────────────────────────────────────
    if is_daytime:
        am_df  = location.get_airmass(solar_position=solar_pos)
        airmass = float(am_df["airmass_relative"].fillna(AIRMASS_NIGHT).iloc[0])
        airmass = min(max(airmass, 1.0), AIRMASS_NIGHT)
    else:
        airmass = AIRMASS_NIGHT

    # ── 4. GHI + clearness index + DNI ─────────────────────────────────────
    ghi = max(0.0, _num(weather.get("GHI_RC_01"), 0.0))

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
    temp_ws   = _num(weather.get("Temp_WS"),  DEFAULTS["Temp_WS"])
    rh_ws     = _num(weather.get("RH_WS"),    DEFAULTS["RH_WS"])
    dwp_ws    = _num(weather.get("DWP_WS"),   DEFAULTS["DWP_WS"])
    ws_ws     = _num(weather.get("WS_WS"),    DEFAULTS["WS_WS"])
    wd_ws     = _num(weather.get("WD_WS"),    DEFAULTS["WD_WS"])
    prec_int  = max(0.0, _num(weather.get("PREC_INT_WS"), DEFAULTS["PREC_INT_WS"]))
    prec_diff = max(0.0, _num(weather.get("PREC_DIFF_WS"), prec_int))

    # ── 6. Reference cell temperature (NOCT approximation) ─────────────────
    # Tc = Ta + (NOCT - 20) / 800 * GHI   (NOCT ≈ 45 °C for typical c-Si cell)
    measured_temp_rc = weather.get("Temp_RC_01")
    try:
        temp_rc = float(measured_temp_rc)
        if not np.isfinite(temp_rc):
            raise ValueError
    except (TypeError, ValueError):
        temp_rc = temp_ws + NOCT_SLOPE * ghi

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
        "PREC_DIFF_WS":     prec_diff,
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
