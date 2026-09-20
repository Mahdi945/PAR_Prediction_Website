"""
core/dataset.py
───────────────
Clean, resample, featurise and score an uploaded time series — the engine
behind the Dataset Upload page.

    detect_columns(columns)          → ColumnMap        guessed from the header names
    validate_timezone(name)          → canonical IANA name, or ValueError with suggestions
    parse_timestamps(series, tz)     → (naive local Series, n_unparsed, notes)
    prepare_dataset_for_prediction() → dict: results · cleaned · features · metrics · report · warnings · domain

The steps mirror the notebooks, so a file scored here is treated the way the
training data was:

    notebook 2   sentinels → missing · rows with ANY out-of-range value dropped ·
                 night rows (GHI ≤ 30 W/m²) dropped · duplicate stamps dropped ·
                 short sensor dropouts carried forward (ffill/bfill)
    notebook 4   wind encoded as sin/cos BEFORE the minute average (resultant
                 vector, |r| ≤ 1) · one vectorised pvlib pass ·
                 Temp_RC_01.fillna(Temp_RC) · rows with the sun below the horizon
    notebook 5   missing → training median · noisy channels clipped to their
                 training 1st/99th percentiles   (core.predict.prepare_matrix)

Performance: every step is a column operation. There is no per-row Python
loop anywhere between the upload and the model call.
"""

from __future__ import annotations

import math
import re
import sys
import warnings as _warnings
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence
from zoneinfo import ZoneInfo, available_timezones

import numpy as np
import pandas as pd

try:
    from .constants import MCCREE_FACTOR
    from .features import compute_features_batch, DEFAULTS as FEATURE_DEFAULTS
    from .predict import load_model, prepare_matrix, train_medians, model_card
    from .domain import check_location, check_features_batch, describe
except ImportError:  # pragma: no cover - script-mode fallback for Streamlit Cloud
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.constants import MCCREE_FACTOR
    from core.features import compute_features_batch, DEFAULTS as FEATURE_DEFAULTS
    from core.predict import load_model, prepare_matrix, train_medians, model_card
    from core.domain import check_location, check_features_batch, describe


# ═════════════════════════════════════════════════════════════════════════════
#  What an uploaded file can contain
# ═════════════════════════════════════════════════════════════════════════════

class SensorSpec(NamedTuple):
    label: str
    unit: str
    synonyms: tuple[str, ...]
    required: bool


# Canonical column → how to recognise it. Matching is exact, then
# case-insensitive, then with punctuation stripped ("Air Temp (C)" → "airtempc").
SENSOR_SPECS: dict[str, SensorSpec] = {
    "GHI_RC_01": SensorSpec("Global horizontal irradiance", "W/m²", (
        "GHI_RC_01", "ghi", "ghi_rc", "ghi_rc_01", "irradiance", "global_irradiance",
        "global_radiation", "shortwave_radiation", "sw_radiation", "solar_radiation",
        "solar_irradiance", "ghi_w_m2", "ghi_wm2", "rad", "pyranometer",
        "globalstrahlung", "einstrahlung", "strahlung",
    ), True),
    "Temp_WS": SensorSpec("Air temperature", "°C", (
        "Temp_WS", "temp", "temperature", "temp_ws", "air_temperature", "temp_air",
        "t_air", "tair", "temperature_2m", "air_temp", "t2m", "ta",
        "lufttemperatur", "temperatur", "air_temp_c",
    ), False),
    "RH_WS": SensorSpec("Relative humidity", "%", (
        "RH_WS", "rh", "humidity", "rh_ws", "relative_humidity", "humidity_rel",
        "relativehumidity_2m", "rel_hum", "hum", "luftfeuchte", "feuchte", "rel_feuchte",
    ), False),
    "DWP_WS": SensorSpec("Dew point", "°C", (
        "DWP_WS", "dwp", "dewpoint", "dwp_ws", "dew_point", "dewpoint_c", "td",
        "dewpoint_2m", "dew_point_temperature", "taupunkt",
    ), False),
    "WS_WS": SensorSpec("Wind speed", "m/s", (
        "WS_WS", "ws", "windspeed", "wind_speed", "ws_ws", "wind_speed_10m", "wind",
        "wspd", "ff", "windgeschwindigkeit",
    ), False),
    "WD_WS": SensorSpec("Wind direction", "°", (
        "WD_WS", "wd", "winddir", "wind_dir", "wind_direction", "wd_ws",
        "winddirection_10m", "wdir", "dd", "windrichtung",
    ), False),
    "PREC_INT_WS": SensorSpec("Rain intensity", "mm/h", (
        "PREC_INT_WS", "prec", "precip", "precipitation", "prec_int_ws", "rain",
        "rain_mm", "rain_rate", "precip_intensity", "rain_intensity", "rr", "niederschlag", "regen",
    ), False),
    "PREC_DIFF_WS": SensorSpec("Rain event", "mm", (
        "PREC_DIFF_WS", "prec_diff_ws", "prec_diff", "precipitation_event", "rain_event",
    ), False),
    "Temp_RC_01": SensorSpec("Reference-cell temperature", "°C", (
        "Temp_RC_01", "temp_rc_01", "temp_rc", "Temp_RC", "panel_temp", "cell_temp",
        "module_temp", "t_cell", "t_module", "tmod", "tcell", "modultemperatur", "zelltemperatur",
    ), False),
}

TIMESTAMP_SYNONYMS = (
    "TIMESTAMP", "timestamp", "datetime", "date_time", "time", "date", "created_at",
    "ts", "utc_time", "local_time", "time_stamp", "measurement_time", "zeit", "datum",
)
TARGET_SYNONYMS = (
    "PAR_PAR", "PAR", "par", "PAR_RC_01", "par_rc_01", "PAR_umol", "par_umol", "PPFD",
    "ppfd", "par_umol_m2_s", "par_measured", "par_obs",
)
# "par" or "ppfd" as a whole token — never the "par" inside "parameter" or "transparency".
_TARGET_RE = re.compile(r"(^|[^a-z])(par|ppfd)([^a-z]|$)")
_LAT_SYNONYMS = ("lat", "latitude", "breite", "y")
_LON_SYNONYMS = ("lon", "lng", "longitude", "laenge", "länge", "x")
_ALT_SYNONYMS = ("alt", "altitude", "elevation", "height", "hoehe", "höhe", "z")

# Logger sentinels and physical validity ranges — identical to notebooks 2 and 4.
SENTINEL_VALUES = (-99999, 99999, 3276.7, -3276.8)
VALIDITY_RANGES: dict[str, tuple[float, float]] = {
    "Temp_WS":      (-40, 60),
    "Temp_RC_01":   (-40, 80),
    "RH_WS":        (0, 100),
    "WD_WS":        (0, 360),
    "WS_WS":        (0, 60),
    "DWP_WS":       (-40, 40),
    "PREC_DIFF_WS": (0, 50),
    "PREC_INT_WS":  (0, 4),
    "GHI_RC_01":    (0, 1362),
    "target_par":   (0, 3000),
}
NIGHT_GHI = 30.0            # W/m² — notebook 2's daytime filter
DEFAULT_MAX_GAP = "3h"      # notebook 2 carried "a few minutes to a few hours" forward
MIN_ROWS_FOR_METRICS = 10

# Columns the cleaned export carries, in this order.
CLEANED_COLUMNS = ["timestamp", *SENSOR_SPECS, "lat", "lon", "alt", "target_par"]


class ColumnMap(NamedTuple):
    timestamp: str | None
    target: str | None
    sensors: dict[str, str | None]     # canonical → source column (None = not found)
    latitude: str | None = None
    longitude: str | None = None
    altitude: str | None = None


# ═════════════════════════════════════════════════════════════════════════════
#  Column detection and parsing helpers
# ═════════════════════════════════════════════════════════════════════════════

def _norm(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _pick(columns: Sequence[Any], candidates: Sequence[str]) -> str | None:
    cols = [str(c) for c in columns]
    for cand in candidates:                                # exact
        if cand in cols:
            return cand
    lowered = {c.lower(): c for c in cols}
    for cand in candidates:                                # case-insensitive
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    normed = {_norm(c): c for c in cols}
    for cand in candidates:                                # punctuation-insensitive
        if _norm(cand) in normed:
            return normed[_norm(cand)]
    return None


def detect_columns(columns: Sequence[Any]) -> ColumnMap:
    """Best guess of which column is which, from the header alone."""
    cols = [str(c) for c in columns]
    sensors = {canon: _pick(cols, spec.synonyms) for canon, spec in SENSOR_SPECS.items()}
    # a column may only play one role
    taken = {c for c in sensors.values() if c}
    target = _pick(cols, TARGET_SYNONYMS)
    if target is None:
        target = next((c for c in cols if c not in taken and _TARGET_RE.search(c.lower())), None)
    ts = _pick(cols, TIMESTAMP_SYNONYMS)
    if ts is None:
        ts = next((c for c in cols if "time" in c.lower() or "date" in c.lower()), None)
    return ColumnMap(
        timestamp=ts, target=target, sensors=sensors,
        latitude=_pick(cols, _LAT_SYNONYMS), longitude=_pick(cols, _LON_SYNONYMS),
        altitude=_pick(cols, _ALT_SYNONYMS),
    )


def validate_timezone(name: str) -> str:
    """Return the IANA name, or raise ValueError naming the closest real zones."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Timezone is empty — for example Europe/Berlin or UTC.")
    try:
        ZoneInfo(name)
        return name
    except Exception:
        zones = sorted(available_timezones())
        lowered = {z.lower(): z for z in zones}
        if name.lower() in lowered:
            return lowered[name.lower()]
        hints = get_close_matches(name, zones, n=3, cutoff=0.5)
        hint = f" Did you mean {', '.join(hints)}?" if hints else " Examples: Europe/Berlin, UTC, America/New_York."
        raise ValueError(f"'{name}' is not a valid IANA timezone.{hint}")


def parse_timestamps(series: pd.Series, tz_str: str) -> tuple[pd.Series, int, list[str]]:
    """Whatever the file holds → naive local wall time in `tz_str`.

    Handles ISO strings, day-first strings, mixed formats, tz-aware strings
    (converted), Unix epochs in seconds or milliseconds (read as UTC), and
    already-parsed datetime columns."""
    notes: list[str] = []
    s = series.reset_index(drop=True)

    if pd.api.types.is_datetime64_any_dtype(s):
        parsed = pd.Series(s)
    elif pd.api.types.is_numeric_dtype(s):
        nums = pd.to_numeric(s, errors="coerce")
        med = float(np.nanmedian(nums)) if nums.notna().any() else 0.0
        unit = "ms" if med > 1e11 else "s"
        parsed = pd.to_datetime(nums, unit=unit, errors="coerce", utc=True)
        notes.append(f"Numeric timestamps were read as Unix epoch {unit} (UTC) and converted to {tz_str}.")
    else:
        text = s.astype("string").str.strip()
        n_nonempty = int(text.notna().sum()) - int((text == "").sum())
        dayfirst = _looks_day_first(text)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")                     # pandas' format-inference chatter
            parsed = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)   # one format, from the first value
            n_bad = int(parsed.isna().sum()) - (len(text) - n_nonempty)
            if n_nonempty and n_bad > 0.01 * n_nonempty:         # one format did not fit: parse individually
                try:
                    alt = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=dayfirst)
                except (ValueError, TypeError):                  # mixed UTC offsets
                    alt = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=dayfirst, utc=True)
                if alt.notna().sum() > parsed.notna().sum():
                    parsed = alt
                    notes.append("Timestamps use more than one format; each value was parsed individually.")
        if dayfirst:
            notes.append("Timestamps were read as day-first (dd.mm.yyyy).")

    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(tz_str).dt.tz_localize(None)
        if not notes or "epoch" not in notes[-1]:
            notes.append(f"Timestamps carried a timezone offset and were converted to {tz_str} local time.")

    return parsed, int(parsed.isna().sum()), notes


_DMY_RE = re.compile(r"^\s*(\d{1,2})([./-])(\d{1,2})\2(\d{2,4})")


def _looks_day_first(text: pd.Series) -> bool:
    """Decide dd.mm.yyyy vs mm/dd/yyyy from a sample.

    A first field above 12 can only be a day; a second field above 12 can only
    be a month-first date. Undecidable dotted dates ('05.06.2025') are read as
    German day-first — that is the convention of the stations this app serves."""
    sample = text.dropna().head(2000)
    firsts, seconds, seps = [], [], set()
    for v in sample:
        m = _DMY_RE.match(str(v))
        if m:
            firsts.append(int(m.group(1))); seconds.append(int(m.group(3))); seps.add(m.group(2))
    if not firsts:
        return False
    if max(firsts) > 12:
        return True
    if max(seconds) > 12:
        return False
    return "." in seps


def _to_float(series: pd.Series) -> pd.Series:
    """Numbers as float64 — tolerant of decimal commas ('12,5') and thin spaces."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    num = pd.to_numeric(series, errors="coerce")
    n_src = int(series.notna().sum())
    if n_src and num.notna().sum() < 0.5 * n_src:
        alt = pd.to_numeric(
            series.astype("string").str.replace(" ", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if alt.notna().sum() > num.notna().sum():
            num = alt
    return num.astype("float64")


def native_resolution_seconds(timestamps: pd.Series) -> float | None:
    """Median spacing between consecutive stamps, ignoring duplicates."""
    d = pd.Series(timestamps).diff().dt.total_seconds()
    d = d[d > 0]
    return float(d.median()) if len(d) else None


def _emit(cb: Callable[[int, str], None] | None, pct: int, msg: str) -> None:
    if cb is not None:
        cb(int(max(0, min(100, pct))), msg)


# ═════════════════════════════════════════════════════════════════════════════
#  The pipeline
# ═════════════════════════════════════════════════════════════════════════════

def prepare_dataset_for_prediction(
    df: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
    target_column: str | None = None,          # None = detect; "" = the file has none
    sensor_mapping: dict[str, str | None] | None = None,
    timezone_str: str = "UTC",
    latitude: float | None = None,
    longitude: float | None = None,
    altitude: float | None = None,
    latitude_column: str | None = None,
    longitude_column: str | None = None,
    altitude_column: str | None = None,
    resample_period: str = "1min",
    max_gap: str = DEFAULT_MAX_GAP,
    progress_callback: Callable[[int, str], None] | None = None,
    # older keyword names — still accepted
    override_lat: float | None = None,
    override_lon: float | None = None,
    override_alt: float | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """Clean, aggregate, featurise and score `df`. See the module docstring."""

    prog = lambda pct, msg: _emit(progress_callback, pct, msg)   # noqa: E731
    report: list[dict[str, Any]] = []
    warnings: list[str] = []

    def step(name: str, before: int, after: int, note: str = "") -> None:
        report.append({"step": name, "rows_in": int(before), "rows_out": int(after),
                       "removed": int(before - after), "note": note})

    if df is None or df.empty:
        raise ValueError("The uploaded dataset is empty.")

    latitude  = latitude  if latitude  is not None else override_lat
    longitude = longitude if longitude is not None else override_lon
    altitude  = altitude  if altitude  is not None else override_alt
    tz = validate_timezone(timezone_str)

    # ── 0. Columns ───────────────────────────────────────────────────────────
    prog(2, "Detecting columns")
    guess = detect_columns(df.columns)
    ts_col = timestamp_column or guess.timestamp
    if not ts_col or ts_col not in df.columns:
        raise ValueError("No timestamp column was found — choose one in the column mapping.")

    sensors: dict[str, str | None] = dict(guess.sensors)
    if sensor_mapping:
        sensors.update({k: v for k, v in sensor_mapping.items() if k in SENSOR_SPECS})
    ghi_col = sensors.get("GHI_RC_01")
    if not ghi_col or ghi_col not in df.columns:
        raise ValueError("A GHI (global horizontal irradiance) column is required — "
                         "it is the model's main input. Map it in the column mapping.")

    if target_column is None:
        target_col = guess.target
    else:
        target_col = target_column or None
    if target_col and target_col not in df.columns:
        target_col = None

    # ── 1. Timestamps ────────────────────────────────────────────────────────
    prog(5, "Parsing timestamps")
    stamps, n_unparsed, ts_notes = parse_timestamps(df[ts_col], tz)
    warnings.extend(ts_notes)
    raw_rows = len(df)
    keep = stamps.notna().to_numpy()
    step("Unparseable timestamps dropped", raw_rows, int(keep.sum()),
         f"{n_unparsed:,} value(s) could not be read as a date/time" if n_unparsed else "")
    if not keep.any():
        raise ValueError(f"None of the values in '{ts_col}' could be read as a date/time.")

    # ── 2. Only the mapped columns, as float64 ───────────────────────────────
    prog(10, "Selecting the mapped columns")
    clean = pd.DataFrame({"timestamp": stamps[keep].to_numpy()})
    used: dict[str, str] = {"timestamp": ts_col}
    absent_model_inputs: list[str] = []
    for canon, spec in SENSOR_SPECS.items():
        src = sensors.get(canon)
        if src and src in df.columns:
            clean[canon] = _to_float(df[src]).to_numpy()[keep]
            used[canon] = src
        else:
            clean[canon] = np.nan
            if canon not in ("Temp_RC_01", "PREC_DIFF_WS"):     # both have physical fallbacks
                absent_model_inputs.append(canon)

    if latitude is not None and longitude is not None:
        clean["lat"] = float(latitude)
        clean["lon"] = float(longitude)
        clean["alt"] = float(altitude if altitude is not None else 0.0)
    else:
        lat_col = latitude_column  or guess.latitude
        lon_col = longitude_column or guess.longitude
        alt_col = altitude_column  or guess.altitude
        if not lat_col or not lon_col or lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError("Latitude and longitude are required — enter the station coordinates "
                             "or map latitude/longitude columns.")
        lat_v = _to_float(df[lat_col]).to_numpy()[keep]
        lon_v = _to_float(df[lon_col]).to_numpy()[keep]
        clean["lat"] = np.where(np.isfinite(lat_v), lat_v, np.nanmedian(lat_v))
        clean["lon"] = np.where(np.isfinite(lon_v), lon_v, np.nanmedian(lon_v))
        if alt_col and alt_col in df.columns:
            alt_v = _to_float(df[alt_col]).to_numpy()[keep]
            clean["alt"] = np.where(np.isfinite(alt_v), alt_v, 0.0)
        else:
            clean["alt"] = float(altitude if altitude is not None else 0.0)
        used.update({"lat": lat_col, "lon": lon_col, **({"alt": alt_col} if alt_col else {})})

    if target_col:
        clean["target_par"] = _to_float(df[target_col]).to_numpy()[keep]
        used["target_par"] = target_col
    else:
        clean["target_par"] = np.nan

    clean = clean.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    for canon in absent_model_inputs:
        spec = SENSOR_SPECS[canon]
        default = train_medians().get(canon, FEATURE_DEFAULTS.get(canon))
        if default is not None:
            warnings.append(f"{spec.label} is not in the file — the training median "
                            f"({default:.2f} {spec.unit}) is used for every row, which costs accuracy.")

    # ── 3. Sentinels ─────────────────────────────────────────────────────────
    prog(15, "Replacing logger sentinel values")
    value_cols = [*SENSOR_SPECS, "target_par"]
    sentinel_mask = clean[value_cols].isin(SENTINEL_VALUES)
    n_sentinel = int(sentinel_mask.to_numpy().sum())
    if n_sentinel:
        clean[value_cols] = clean[value_cols].mask(sentinel_mask)
    step("Sentinel values → missing", len(clean), len(clean),
         f"{n_sentinel:,} cell(s) held {', '.join(str(v) for v in SENTINEL_VALUES)}" if n_sentinel else "none found")

    # ── 4. Physical validity — drop the row, as notebook 2 does ──────────────
    prog(20, "Applying physical validity ranges")
    n0 = len(clean)
    bad = np.zeros(n0, dtype=bool)
    per_col: dict[str, int] = {}
    for col, (lo, hi) in VALIDITY_RANGES.items():
        if col not in clean.columns:
            continue
        v = clean[col].to_numpy(dtype="float64")
        m = np.isfinite(v) & ((v < lo) | (v > hi))
        if m.any():
            per_col[col] = int(m.sum())
            bad |= m
    clean = clean.loc[~bad]
    step("Out-of-range values (row dropped)", n0, len(clean),
         "; ".join(f"{c}: {n:,} outside {VALIDITY_RANGES[c]}" for c, n in per_col.items()) or "all values physical")

    # ── 5. Night filter ──────────────────────────────────────────────────────
    prog(25, "Removing night and twilight rows")
    n0 = len(clean)
    n_no_ghi = int(clean["GHI_RC_01"].isna().sum())
    clean = clean.loc[clean["GHI_RC_01"] > NIGHT_GHI]
    step(f"Night / twilight rows (GHI ≤ {NIGHT_GHI:.0f} W/m²)", n0, len(clean),
         f"{n_no_ghi:,} of them had no GHI reading" if n_no_ghi else "")

    # ── 6. Duplicates ────────────────────────────────────────────────────────
    prog(30, "Dropping duplicate timestamps")
    n0 = len(clean)
    clean = clean.drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    step("Duplicate timestamps (first kept)", n0, len(clean))

    if clean.empty:
        step("Result", 0, 0, "no daytime rows survived cleaning")
        return _empty_result(raw_rows, report, warnings, used, tz)

    # ── 7. Short dropouts carried forward (notebook 2: ffill → bfill) ───────
    prog(35, "Filling short sensor dropouts")
    res_s = native_resolution_seconds(clean["timestamp"])
    gap_limit = pd.Timedelta(max_gap)
    gap_rows = max(1, int(math.ceil(gap_limit.total_seconds() / max(res_s or 1.0, 1.0))))
    medians = train_medians()
    fill_notes: list[str] = []
    fill_summary: dict[str, dict[str, Any]] = {}
    for canon, spec in SENSOR_SPECS.items():
        s = clean[canon]
        n_missing = int(s.isna().sum())
        if n_missing == 0 or canon == "GHI_RC_01":
            continue
        if canon == "Temp_RC_01":
            # structural — notebook 2 leaves it; features.py falls back to the NOCT model
            fill_summary[canon] = {"missing": n_missing, "carried": 0, "median": 0, "fallback": "NOCT model"}
            continue
        carried = s.ffill(limit=gap_rows).bfill(limit=gap_rows)
        n_left = int(carried.isna().sum())
        default = float(medians.get(canon, FEATURE_DEFAULTS.get(canon, 0.0)))
        clean[canon] = carried.fillna(default)
        fill_summary[canon] = {"missing": n_missing, "carried": n_missing - n_left,
                               "median": n_left, "fallback": default}
        if n_left and canon in used:
            fill_notes.append(f"{spec.label}: {n_left:,} value(s) missing for longer than {max_gap} "
                              f"→ training median {default:.2f} {spec.unit}")
    if clean["target_par"].notna().any():
        clean["target_par"] = clean["target_par"].ffill(limit=gap_rows).bfill(limit=gap_rows)
    step("Short dropouts carried forward", len(clean), len(clean),
         f"gaps ≤ {max_gap} filled ({gap_rows} sample(s) at {res_s:.0f} s resolution)" if res_s else "")
    warnings.extend(fill_notes)

    # ── 8. Wind as a vector, then the minute average ─────────────────────────
    prog(45, f"Aggregating to {resample_period} (wind as sin/cos first)")
    wd_rad = np.deg2rad(clean["WD_WS"].to_numpy(dtype="float64"))
    clean["wind_sin"] = np.sin(wd_rad)
    clean["wind_cos"] = np.cos(wd_rad)

    period = pd.Timedelta(resample_period)
    if res_s is not None and res_s >= period.total_seconds():
        resampled = clean.copy()
        resampled["n_samples"] = 1
        agg_note = f"native resolution {res_s:.0f} s ≥ {resample_period}: rows kept as they are"
    else:
        key = clean["timestamp"].dt.floor(resample_period)
        num_cols = [c for c in clean.columns if c != "timestamp"]
        grouped = clean.groupby(key, sort=True)
        resampled = grouped[num_cols].mean()
        resampled["n_samples"] = grouped.size()
        resampled = resampled.reset_index()
        agg_note = f"{len(clean):,} rows → {len(resampled):,} bins (means; wind as the resultant vector)"
    resampled["WD_WS"] = np.degrees(np.arctan2(resampled["wind_sin"], resampled["wind_cos"])) % 360.0
    resampled["GHI_rolling_5min"] = resampled["GHI_RC_01"].rolling(5, min_periods=1, center=True).mean()
    step(f"Aggregation to {resample_period}", len(clean), len(resampled), agg_note)

    # ── 9. Features — one vectorised pvlib pass ──────────────────────────────
    prog(55, "Computing solar geometry and features")
    features, is_day = compute_features_batch(resampled, tz_str=tz)
    n_night = int((~is_day).sum())
    if n_night:
        step("Sun below the horizon at the aggregated stamp", len(resampled), len(resampled),
             f"{n_night:,} bin(s) get PAR = 0 (elevation ≤ 0.5°)")

    # ── 10. Model — one call ─────────────────────────────────────────────────
    prog(75, "Running the model")
    model, _ = load_model()
    X, prep_info = prepare_matrix(features, return_info=True)
    preds = np.maximum(0.0, np.asarray(model.predict(X), dtype="float64"))
    preds = np.where(is_day, preds, 0.0)
    ghi = resampled["GHI_RC_01"].to_numpy(dtype="float64")
    baseline = np.maximum(0.0, ghi * MCCREE_FACTOR)

    # ── 11. Results ──────────────────────────────────────────────────────────
    prog(88, "Assembling results")
    target = resampled["target_par"].to_numpy(dtype="float64")
    results = pd.DataFrame({
        "timestamp":           resampled["timestamp"].to_numpy(),
        "lat":                 resampled["lat"].to_numpy(dtype="float64"),
        "lon":                 resampled["lon"].to_numpy(dtype="float64"),
        "alt":                 resampled["alt"].to_numpy(dtype="float64"),
        "GHI":                 ghi,
        "model_prediction":    preds,
        "baseline_prediction": baseline,
        "difference":          preds - baseline,
        "is_day":              is_day,
        "target_par":          target,
        "Temp_WS":             resampled["Temp_WS"].to_numpy(dtype="float64"),
        "RH_WS":               resampled["RH_WS"].to_numpy(dtype="float64"),
        "DWP_WS":              resampled["DWP_WS"].to_numpy(dtype="float64"),
        "WS_WS":               resampled["WS_WS"].to_numpy(dtype="float64"),
        "WD_WS":               resampled["WD_WS"].to_numpy(dtype="float64"),
        "PREC_INT_WS":         resampled["PREC_INT_WS"].to_numpy(dtype="float64"),
        "Temp_RC_merged":      features["Temp_RC_merged"].to_numpy(dtype="float64"),
        "zenith":              features["zenith"].to_numpy(dtype="float64"),
        "elevation":           features["elevation"].to_numpy(dtype="float64"),
        "clearness_kt":        features["clearness_kt"].to_numpy(dtype="float64"),
        "n_samples":           resampled["n_samples"].to_numpy(),
    })
    features_out = pd.concat(
        [resampled[["timestamp", "lat", "lon", "alt"]].reset_index(drop=True),
         features.reset_index(drop=True)], axis=1,
    )
    cleaned_out = clean[CLEANED_COLUMNS].copy()
    for c in [*SENSOR_SPECS, "target_par"]:                 # half the memory for the export copy
        cleaned_out[c] = cleaned_out[c].astype("float32")

    metrics = _metrics(results) if np.isfinite(target).any() else {}

    # ── 12. Is this inside what the model has seen? ──────────────────────────
    prog(94, "Checking the training domain")
    loc = check_location(float(np.nanmedian(results["lat"])), float(np.nanmedian(results["lon"])))
    feat_dom = check_features_batch(features.loc[is_day]) if is_day.any() else check_features_batch(features.iloc[0:0])
    # Deliberately NOT added to `warnings`: the page has a dedicated
    # "Training-domain check" expander and the JSON report carries the same
    # facts. A banner repeated on every file outside Germany only teaches
    # people to skip banners, including the ones about their own data.
    domain_notes = describe(loc, None)

    prog(100, "Completed")
    return {
        "raw_rows":        int(raw_rows),
        "parsed_rows":     int(keep.sum()),
        "clean_rows":      int(len(clean)),
        "resampled_rows":  int(len(results)),
        "daytime_rows":    int(is_day.sum()),
        "results":         results,
        "cleaned":         cleaned_out,
        "features":        features_out,
        "metrics":         metrics,
        "report":          report,
        "warnings":        warnings,
        "mapping":         used,
        "fill_summary":    fill_summary,
        "prep_info":       prep_info,
        "domain":          {"location": loc._asdict(), "features": feat_dom,
                            "notes": domain_notes},
        "resolution_seconds": res_s,
        "resample_period": resample_period,
        "timezone":        tz,
        "coordinates":     {"lat": float(np.nanmedian(results["lat"])),
                            "lon": float(np.nanmedian(results["lon"])),
                            "alt": float(np.nanmedian(results["alt"]))},
        "model":           model_card(),
        "period":          {"start": results["timestamp"].min(), "end": results["timestamp"].max()},
        # kept for older callers
        "selected_columns": {"timestamp": ts_col, "lat": used.get("lat"), "lon": used.get("lon"), "alt": used.get("alt")},
        "sensor_mapping":  {k: v for k, v in used.items() if k in SENSOR_SPECS},
    }


def _empty_result(raw_rows: int, report: list, warnings: list, used: dict, tz: str) -> dict[str, Any]:
    empty = pd.DataFrame()
    return {
        "raw_rows": int(raw_rows), "parsed_rows": 0, "clean_rows": 0, "resampled_rows": 0, "daytime_rows": 0,
        "results": empty, "cleaned": empty, "features": empty, "metrics": {}, "report": report,
        "warnings": warnings + ["No daytime rows survived cleaning — check the GHI column and its unit (W/m²)."],
        "mapping": used, "fill_summary": {}, "prep_info": {}, "domain": {"location": None, "features": empty},
        "resolution_seconds": None, "resample_period": None, "timezone": tz, "coordinates": {},
        "model": model_card(), "period": {},
        "selected_columns": {}, "sensor_mapping": {},
    }


def _metrics(results: pd.DataFrame) -> dict[str, float]:
    """Model vs McCree on daytime bins with an observed PAR > 0 — the same
    conditions notebook 6 evaluates on."""
    mask = (
        results["is_day"].to_numpy(dtype=bool)
        & np.isfinite(results["target_par"].to_numpy(dtype="float64"))
        & (results["target_par"].to_numpy(dtype="float64") > 0)
    )
    ev = results.loc[mask]
    if len(ev) < MIN_ROWS_FOR_METRICS:
        return {}

    y  = ev["target_par"].to_numpy(dtype="float64")
    ym = ev["model_prediction"].to_numpy(dtype="float64")
    yb = ev["baseline_prediction"].to_numpy(dtype="float64")
    mean_y = float(np.mean(y))

    def mae(p):  return float(np.mean(np.abs(y - p)))
    def rmse(p): return float(np.sqrt(np.mean((y - p) ** 2)))
    def mbe(p):  return float(np.mean(p - y))
    def r2(p):
        ss_res = float(np.sum((y - p) ** 2)); ss_tot = float(np.sum((y - mean_y) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else (1.0 if ss_res <= 1e-12 else 0.0)
    def pct(v):  return v / mean_y * 100.0 if mean_y > 1e-6 else float("nan")

    m_mae, b_mae, m_rmse, b_rmse = mae(ym), mae(yb), rmse(ym), rmse(yb)
    return {
        "daytime_rows_evaluated": int(len(ev)),
        "model_mae": m_mae,   "model_rmse": m_rmse,   "model_nrmse": pct(m_rmse),
        "model_mbe": mbe(ym), "model_nmbe": pct(mbe(ym)), "model_r2": r2(ym),
        "baseline_mae": b_mae, "baseline_rmse": b_rmse, "baseline_nrmse": pct(b_rmse),
        "baseline_mbe": mbe(yb), "baseline_nmbe": pct(mbe(yb)), "baseline_r2": r2(yb),
        "mae_improvement_pct":  (b_mae - m_mae) / b_mae * 100.0 if b_mae > 1e-6 else 0.0,
        "rmse_improvement_pct": (b_rmse - m_rmse) / b_rmse * 100.0 if b_rmse > 1e-6 else 0.0,
    }
