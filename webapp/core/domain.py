"""
core/domain.py
──────────────
Is this input something the model has seen before?

The model was trained on two stations in Brandenburg, Germany. Nothing stops
a visitor asking about Nairobi in January; XGBoost will answer just as
confidently, because trees extrapolate flat. These checks make that visible:

    check_location(lat, lon)          → LocationCheck   distance to the nearest training station
    check_features(features_df)       → [FeatureCheck]  one row: values outside the training range
    check_features_batch(features_df) → DataFrame       many rows: share outside the range, per feature
    describe(location, features)      → [str]           sentences for st.warning
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from .predict import training_bounds, NO_CLIP

# Station coordinates as used in notebook 4 (LOCATION_COORDS).
TRAINING_SITES = {
    "Laubsdorf": (51.68718711157154, 14.414256224166728),
    "Nebelin":   (53.11833921379164, 11.746088890223463),
}
# Inside this radius the climate and the annual sun path are comparable to the
# training stations; beyond it the prediction is an extrapolation.
NEAR_KM = 300.0

# What a reader sees instead of the column name.
FEATURE_LABELS = {
    "GHI_RC_01":      ("GHI",                    "W/m²"),
    "Temp_WS":        ("air temperature",        "°C"),
    "RH_WS":          ("relative humidity",      "%"),
    "DWP_WS":         ("dew point",              "°C"),
    "WS_WS":          ("wind speed",             "m/s"),
    "PREC_INT_WS":    ("rain intensity",         "mm/h"),
    "PREC_DIFF_WS":   ("rain event",             "mm"),
    "Temp_RC_merged": ("reference-cell temperature", "°C"),
    "zenith":         ("solar zenith",           "°"),
    "elevation":      ("solar elevation",        "°"),
    "airmass":        ("relative airmass",       ""),
    "clearness_kt":   ("clearness index",        ""),
    "dni":            ("direct normal irradiance", "W/m²"),
    "wind_sin":       ("wind direction (sin)",   ""),
    "wind_cos":       ("wind direction (cos)",   ""),
}
# Features that are never worth a warning. A warning should mean "this could
# move the answer"; these cannot, and crying wolf about them teaches visitors
# to ignore the ones that matter.
#
#   PREC_INT_WS   rain above the 99th percentile is ordinary weather, and
#                 0 -> 3 mm/h moves PAR by 5 umol/m2/s (0.4 %). Open-Meteo also
#                 reports gridded hourly precipitation, which ran ~2.6x the
#                 stations' tipping-bucket intensity (corr 0.35-0.43), so the
#                 comparison with the training range is not like for like.
#   PREC_DIFF_WS  constant 0 throughout training; importance 0.0000.
#   wind_sin/cos  bounded to [-1, 1] by construction.
#   WS_WS         importance 0.0004 (13th of 15). Measured against the model:
#                 1.85 -> 15 m/s moves PAR by 1.7 umol/m2/s (0.1 %). On top of
#                 that Open-Meteo reports wind at 10 m while the station
#                 anemometers sit lower, so its readings ran ~1.85x the
#                 training values at both sites - a height difference, not a
#                 unit error, and not something to alarm a visitor about.
_QUIET = {"PREC_INT_WS", "PREC_DIFF_WS", "wind_sin", "wind_cos", "WS_WS"}


class LocationCheck(NamedTuple):
    nearest: str
    distance_km: float
    in_domain: bool


class FeatureCheck(NamedTuple):
    feature: str
    value: float
    low: float
    high: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def check_location(lat: float, lon: float) -> LocationCheck:
    best_name, best_km = "", float("inf")
    for name, (slat, slon) in TRAINING_SITES.items():
        d = haversine_km(float(lat), float(lon), slat, slon)
        if d < best_km:
            best_name, best_km = name, d
    return LocationCheck(best_name, round(best_km, 1), best_km <= NEAR_KM)


def check_features(features_df: pd.DataFrame) -> list[FeatureCheck]:
    """Model features of a one-row table that fall outside the training range."""
    bounds = training_bounds()
    out: list[FeatureCheck] = []
    row = features_df.iloc[0]
    for name, (lo, hi) in bounds.items():
        if name in _QUIET or name not in row.index:
            continue
        v = float(row[name])
        if not np.isfinite(v):
            continue
        if v < lo or v > hi:
            out.append(FeatureCheck(name, v, lo, hi))
    return out


def check_features_batch(features_df: pd.DataFrame) -> pd.DataFrame:
    """Per feature: how many rows sit outside the training range.

    Columns: feature, label, unit, low, high, n_out, pct_out, min, max.
    Only features with at least one row outside are returned, worst first."""
    bounds = training_bounds()
    n = len(features_df)
    rows = []
    for name, (lo, hi) in bounds.items():
        if name in _QUIET or name not in features_df.columns or n == 0:
            continue
        v = pd.to_numeric(features_df[name], errors="coerce").to_numpy(dtype="float64")
        finite = np.isfinite(v)
        out = finite & ((v < lo) | (v > hi))
        n_out = int(out.sum())
        if n_out == 0:
            continue
        label, unit = FEATURE_LABELS.get(name, (name, ""))
        rows.append({
            "feature": name, "label": label, "unit": unit,
            "low": lo, "high": hi, "n_out": n_out,
            "pct_out": round(n_out / max(int(finite.sum()), 1) * 100.0, 2),
            "min": float(np.nanmin(v[finite])) if finite.any() else np.nan,
            "max": float(np.nanmax(v[finite])) if finite.any() else np.nan,
        })
    cols = ["feature", "label", "unit", "low", "high", "n_out", "pct_out", "min", "max"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values("pct_out", ascending=False).reset_index(drop=True)


def _fmt(v: float, unit: str) -> str:
    s = f"{v:.2f}" if abs(v) < 10 else (f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}")
    return f"{s}{(' ' + unit) if unit and unit != '°' else unit}" if unit else s


def describe(location: LocationCheck | None,
             features: list[FeatureCheck] | None) -> list[str]:
    """Plain sentences a visitor can act on. Empty list = nothing to flag."""
    notes: list[str] = []
    if location is not None and not location.in_domain:
        notes.append(
            f"This location is {location.distance_km:,.0f} km from the nearest training "
            f"station ({location.nearest}, Brandenburg, Germany). The model has only seen "
            f"German lowland conditions — treat the result as an extrapolation."
        )
    for fc in features or []:
        label, unit = FEATURE_LABELS.get(fc.feature, (fc.feature, ""))
        how = "above" if fc.value > fc.high else "below"
        treated = fc.high if fc.value > fc.high else fc.low
        tail = (f"; the model treats it as {_fmt(treated, unit)}"
                if fc.feature not in NO_CLIP else
                f"; the model extrapolates from its edge")
        notes.append(
            f"{label[0].upper()}{label[1:]} {_fmt(fc.value, unit)} is {how} the training range "
            f"{_fmt(fc.low, unit)} – {_fmt(fc.high, unit)}{tail}."
        )
    return notes


def error_note(mae: float, horizon_days: int | None = None) -> str:
    """How far the model's error figure actually reaches, in words.

    The MAE is what the model gets wrong *given the weather it is handed*. In
    Normal Mode that weather is Open-Meteo's rather than a station reading, and
    PAR follows GHI almost one for one, so the input's own error lands on top of
    the model's. Past a few days ahead the forecast term is the larger of the
    two — printing a bare "typical error ± 29" for a request 15 days out would
    claim an accuracy no part of the system has.

    ``horizon_days`` is 0 for today, ``None`` for a past date or for readings
    typed in by hand, and positive for a forecast.
    """
    note = f"model error ± {mae:.0f} µmol/m²/s for the weather shown"
    if horizon_days and horizon_days > 0:
        note += " — the forecast adds its own error on top"
    return note
