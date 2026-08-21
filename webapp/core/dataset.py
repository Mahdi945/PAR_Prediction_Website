"""
core/dataset.py
---------------
Utilities for preparing uploaded time-series datasets for PAR prediction.
"""

from __future__ import annotations

from typing import Any, Callable
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pvlib

try:
    from .features import compute_features, compute_features_batch
    from .predict import is_model_available, load_model, mccree_estimate, predict_par
    from .weather import fetch_weather, geocode_city
except ImportError:  # pragma: no cover - script-mode fallback for Streamlit Cloud
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.features import compute_features, compute_features_batch
    from core.predict import is_model_available, load_model, mccree_estimate, predict_par
    from core.weather import fetch_weather, geocode_city


def _pick_column(df: pd.DataFrame, candidates: list[str], default: str | None = None) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return default


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _emit_progress(progress_callback: Callable[[int, str], None] | None, pct: int, message: str) -> None:
    if progress_callback is None:
        return
    progress_callback(int(max(0, min(100, pct))), message)


def _infer_city_from_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    base = Path(filename).stem
    if not base:
        return None
    city = base.split("_")[0].strip()
    return city if city else None


def _infer_location_from_filename(filename: str | None, sample_dt: pd.Timestamp | None = None) -> dict[str, Any] | None:
    city = _infer_city_from_filename(filename)
    if not city:
        return None

    candidates = geocode_city(city)
    if not candidates:
        return None

    location = candidates[0].copy()
    if not location.get("timezone") and sample_dt is not None:
        try:
            weather = fetch_weather(location["latitude"], location["longitude"], sample_dt)
            location["timezone"] = weather.get("_timezone", "UTC")
        except Exception:
            location["timezone"] = "UTC"

    return location


def prepare_dataset_for_prediction(
    df: pd.DataFrame,
    timestamp_column: str | None = None,
    latitude_column: str | None = None,
    longitude_column: str | None = None,
    altitude_column: str | None = None,
    timezone_str: str = "UTC",
    filename: str | None = None,
    override_lat: float | None = None,
    override_lon: float | None = None,
    override_alt: float | None = None,
    sensor_mapping: dict[str, str] | None = None,
    target_column: str | None = None,
    resample_period: str = "1min",
    predict_fn: Any | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """
    Clean, resample and score a user-uploaded dataset.

    The function expects at least a timestamp and geolocation information.
    Weather sensors can be mapped through sensor_mapping; if omitted, common
    column names are guessed automatically.
    """

    _emit_progress(progress_callback, 2, "Validating uploaded dataset")

    if df is None or df.empty:
        raise ValueError("The uploaded dataset is empty.")

    work_df = df.copy()
    timestamp_col = timestamp_column or _pick_column(
        work_df,
        ["timestamp", "datetime", "date_time", "time", "date", "created_at", "ts"],
    )
    if not timestamp_col:
        raise ValueError("No timestamp column was found in the uploaded dataset.")

    _emit_progress(progress_callback, 6, "Detecting required columns")

    latitude_col  = latitude_column  or _pick_column(work_df, ["lat", "latitude", "y"])
    longitude_col = longitude_column or _pick_column(work_df, ["lon", "longitude", "x"])
    altitude_col  = altitude_column  or _pick_column(work_df, ["alt", "altitude", "elevation", "height"])

    if override_lat is not None:
        work_df["lat"] = override_lat;  latitude_col = "lat"
    if override_lon is not None:
        work_df["lon"] = override_lon;  longitude_col = "lon"
    if override_alt is not None:
        work_df["alt"] = override_alt;  altitude_col = "alt"

    if filename is not None:
        inferred_location = _infer_location_from_filename(filename, None)
        if inferred_location:
            if not latitude_col:
                work_df["lat"] = inferred_location["latitude"];  latitude_col = "lat"
            if not longitude_col:
                work_df["lon"] = inferred_location["longitude"]; longitude_col = "lon"
            if not altitude_col and override_alt is None:
                work_df["alt"] = inferred_location.get("elevation", 0.0); altitude_col = "alt"
            if timezone_str == "UTC":
                timezone_str = inferred_location.get("timezone", "UTC")

    if not latitude_col or not longitude_col:
        raise ValueError("Latitude and longitude columns are required.")

    # ── Parse timestamps ──────────────────────────────────────────────────────
    work_df[timestamp_col] = pd.to_datetime(work_df[timestamp_col], errors="coerce")
    work_df = work_df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)

    clean_df = work_df.copy()
    clean_df[latitude_col]  = _safe_numeric(clean_df[latitude_col])
    clean_df[longitude_col] = _safe_numeric(clean_df[longitude_col])
    if altitude_col:
        clean_df[altitude_col] = _safe_numeric(clean_df[altitude_col])

    clean_df["lat"] = clean_df[latitude_col].fillna(clean_df[latitude_col].median())
    clean_df["lon"] = clean_df[longitude_col].fillna(clean_df[longitude_col].median())
    clean_df["alt"] = clean_df[altitude_col].fillna(clean_df[altitude_col].median()) if altitude_col else 0.0

    # ── Map sensor columns with generalized names used across stations ─────
    # Many uploaded datasets use variants like "ghi_rc" instead of "GHI_RC_01".
    # We accept case-insensitive matches and common synonyms to avoid a hard stop.
    default_mapping = {
        "GHI_RC_01": _pick_column(
            clean_df,
            [
                "ghi", "GHI", "GHI_RC_01", "GHI_RC", "ghi_rc_01", "ghi_rc",
                "irradiance", "shortwave_radiation", "sw_radiation"
            ],
            default="GHI",
        ),
        "Temp_WS": _pick_column(
            clean_df,
            ["temp", "temperature", "Temp_WS", "temp_ws", "air_temperature", "temp_air"],
            default="Temp",
        ),
        "RH_WS": _pick_column(
            clean_df,
            ["rh", "humidity", "RH_WS", "rh_ws", "relative_humidity", "humidity_rel"],
            default="RH",
        ),
        "DWP_WS": _pick_column(
            clean_df,
            ["dwp", "dewpoint", "DWP_WS", "dwp_ws", "dew_point", "dewpoint_c"],
            default="DWP",
        ),
        "WS_WS": _pick_column(
            clean_df,
            ["ws", "windspeed", "wind_speed", "WS_WS", "ws_ws", "wind_speed_10m"],
            default="WS",
        ),
        "WD_WS": _pick_column(
            clean_df,
            ["wd", "winddir", "wind_dir", "wind_direction", "WD_WS", "wd_ws"],
            default="WD",
        ),
        "PREC_INT_WS": _pick_column(
            clean_df,
            ["prec", "precip", "precipitation", "PREC_INT_WS", "prec_int_ws", "rain", "rain_mm"],
            default="PREC",
        ),
        "PREC_WS": _pick_column(
            clean_df,
            ["prec_ws", "PREC_WS", "precip_total", "rain_total", "precipitation_total"],
            default="PREC_WS",
        ),
        "Temp_RC_01": _pick_column(
            clean_df,
            ["temp_rc_01", "Temp_RC_01", "temp_rc", "panel_temp", "cell_temp", "module_temp"],
            default="Temp_RC_01",
        ),
    }
    if sensor_mapping:
        default_mapping.update(sensor_mapping)

    for dst_col, src_col in default_mapping.items():
        clean_df[dst_col] = _safe_numeric(clean_df[src_col]) if (src_col and src_col in clean_df.columns) else np.nan

    # ── Resolve observed PAR column (flexible name matching) ─────────────────
    resolved_target: str | None = None
    if target_column:
        resolved_target = target_column if target_column in clean_df.columns else next(
            (c for c in clean_df.columns if c.lower() == target_column.lower()), None
        )
    if resolved_target is None:
        resolved_target = next((c for c in clean_df.columns if "par" in c.lower()), None)

    if resolved_target:
        clean_df["target_par"] = _safe_numeric(clean_df[resolved_target])
    else:
        clean_df["target_par"] = np.nan

    raw_rows = len(clean_df)
    # ── Sentinel replacement + physical validity ranges (align with notebooks)
    _emit_progress(progress_callback, 12, "Replacing sentinels and removing out-of-range values")

    # Sentinels used in notebooks
    SENTINEL_VALUES = [-99999, 99999, 3276.7, -3276.8]
    for sv in SENTINEL_VALUES:
        clean_df.replace(sv, np.nan, inplace=True)

    # Validity ranges from the notebooks
    VALIDITY_RANGES = {
        'Temp_WS': (-40, 60),
        'Temp_RC_01': (-40, 80),
        'Temp_RC': (-40, 80),
        'RH_WS': (0, 100),
        'WD_WS': (0, 360),
        'WS_WS': (0, 60),
        'DWP_WS': (-40, 40),
        'PREC_WS': (0, 200),
        'PREC_DIFF_WS': (0, 50),
        'PREC_INT_WS': (0, 4),
        'PAR_PAR': (0, 3000),
        'GHI_RC_01': (0, 1362),
    }

    # Drop rows violating any validity range (if column exists)
    for col, (lo, hi) in VALIDITY_RANGES.items():
        if col not in clean_df.columns:
            continue
        before = len(clean_df)
        mask = clean_df[col].notna() & ((clean_df[col] < lo) | (clean_df[col] > hi))
        if mask.any():
            clean_df = clean_df[~mask].copy()

    # If target present, additionally constrain target PAR
    if resolved_target:
        par_ok = ~(clean_df["target_par"] < 0) & ~(clean_df["target_par"] > 3000)
        clean_df = clean_df[par_ok].copy()

    # ── Night-time / twilight filter — match notebook: remove GHI <= 30 W/m²
    _emit_progress(progress_callback, 18, "Filtering twilight/night rows (GHI <= 30 W/m²)")
    if "GHI_RC_01" in clean_df.columns:
        before = len(clean_df)
        clean_df = clean_df[clean_df["GHI_RC_01"] > 30].copy()
        _emit_progress(progress_callback, 20, f"GHI filter kept {len(clean_df):,} rows (removed {before - len(clean_df):,})")

    # ── Remove duplicate timestamps (notebooks drop duplicates on TIMESTAMP)
    if timestamp_col in clean_df.columns:
        before = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=[timestamp_col], keep='first').copy()
        if len(clean_df) != before:
            _emit_progress(progress_callback, 22, f"Dropped {before - len(clean_df):,} duplicate timestamps")

    clean_df = clean_df.sort_values(by=timestamp_col).reset_index(drop=True)

    # ── Time-interpolation on cleaned daytime data ────────────────────────────
    _emit_progress(progress_callback, 24, "Interpolating missing sensor values")
    clean_df = clean_df.set_index(timestamp_col)
    sensor_cols = ["GHI_RC_01", "Temp_WS", "RH_WS", "DWP_WS", "WS_WS", "WD_WS", "PREC_INT_WS", "lat", "lon", "alt"]
    for col in sensor_cols:
        clean_df[col] = clean_df[col].astype(float).interpolate(method="time", limit_direction="both").fillna(clean_df[col].median())

    if resolved_target:
        clean_df["target_par"] = (
            clean_df["target_par"].astype(float)
            .interpolate(method="time", limit_direction="both")
            .fillna(clean_df["target_par"].median())
        )

    clean_df = clean_df.reset_index()
    clean_rows = len(clean_df)

    _emit_progress(progress_callback, 30, "Resampling to target period")

    resampled = (
        clean_df.set_index(timestamp_col)
        .resample(resample_period)
        .agg({
            "lat":         "mean",
            "lon":         "mean",
            "alt":         "mean",
            "GHI_RC_01":   "mean",
            "Temp_WS":     "mean",
            "RH_WS":       "mean",
            "DWP_WS":      "mean",
            "WS_WS":       "mean",
            "WD_WS":       "mean",
            "PREC_INT_WS": "sum",
            "target_par":  "mean",
        })
        .reset_index()
        .rename(columns={timestamp_col: "timestamp"})
        .dropna(subset=["lat", "lon"])
        .reset_index(drop=True)
    )

    # Compute GHI_rolling_5min as proper 5-minute rolling mean (mirrors notebook 4)
    resampled["GHI_rolling_5min"] = (
        resampled["GHI_RC_01"]
        .rolling(window=5, min_periods=1, center=True)
        .mean()
    )

    _emit_progress(progress_callback, 36, "Computing solar geometry for all rows")

    # Drop rows with invalid timestamps before vectorized processing
    resampled["timestamp"] = pd.to_datetime(resampled["timestamp"], errors="coerce")
    resampled = resampled.dropna(subset=["timestamp"]).reset_index(drop=True)

    _emit_progress(progress_callback, 42, "Building feature matrix (vectorized)")

    # One pvlib pass + one model.predict() call — mirrors notebook 4_4_feature_engineering_All_Files
    feat_df, is_daytime_arr = compute_features_batch(resampled, tz_str=timezone_str)

    _emit_progress(progress_callback, 72, "Running batch model inference")

    ghi_vals: np.ndarray       = np.asarray(resampled["GHI_RC_01"].fillna(0.0), dtype=float)
    baseline_preds: np.ndarray = np.maximum(0.0, ghi_vals * 2.06)

    if predict_fn is not None:
        all_preds = np.array([float(predict_fn(feat_df.iloc[[i]])) for i in range(len(feat_df))])
    elif is_model_available():
        model, feature_names = load_model()
        feature_names = feature_names or []
        X = pd.DataFrame(index=feat_df.index)
        for col in feature_names:
            X[col] = feat_df[col].values if col in feat_df.columns else 0.0
        all_preds = model.predict(X).astype(float)
    else:
        all_preds = np.full(len(resampled), float("nan"))

    # Night-time rows → 0 prediction (mirrors training filter)
    all_preds = np.where(is_daytime_arr, np.maximum(0.0, all_preds), 0.0)

    _emit_progress(progress_callback, 88, "Assembling results")

    target_vals = resampled["target_par"].values.astype(float)

    results_df = pd.DataFrame({
        "timestamp":           resampled["timestamp"].values,
        "lat":                 resampled["lat"].values.astype(float),
        "lon":                 resampled["lon"].values.astype(float),
        "alt":                 resampled["alt"].values.astype(float),
        "GHI":                 ghi_vals,
        "model_prediction":    all_preds,
        "baseline_prediction": baseline_preds,
        "difference":          all_preds - baseline_preds,
        "is_day":              (resampled["GHI_RC_01"].values > 30),
        "target_par":          target_vals,
    })
    if results_df.empty:
        _emit_progress(progress_callback, 100, "Completed (no rows after processing)")
        return {
            "raw_rows": len(work_df),
            "clean_rows": len(clean_df),
            "resampled_rows": 0,
            "results": results_df,
            "metrics": {},
            "selected_columns": {"timestamp": timestamp_col, "lat": latitude_col, "lon": longitude_col, "alt": altitude_col},
            "sensor_mapping": default_mapping,
        }

    metrics = {}
    if pd.notna(results_df["target_par"]).any():
        _emit_progress(progress_callback, 93, "Computing metrics")

        # Mirror notebook 5_5_stratification: evaluate on DAYTIME rows with positive PAR only
        day_mask = (
            results_df["is_day"].astype(bool)
            & np.isfinite(results_df["target_par"].astype(float))
            & (results_df["target_par"].astype(float) > 0)
            & np.isfinite(results_df["model_prediction"].astype(float))
            & np.isfinite(results_df["baseline_prediction"].astype(float))
        )
        ev = results_df[day_mask]

        if len(ev) >= 10:
            y_true     = np.asarray(ev["target_par"],           dtype=float)
            y_model    = np.asarray(ev["model_prediction"],     dtype=float)
            y_baseline = np.asarray(ev["baseline_prediction"],  dtype=float)
            mean_true  = float(np.mean(y_true))

            def _mae(yt: np.ndarray, yp: np.ndarray) -> float:
                return float(np.mean(np.abs(yt - yp)))

            def _rmse(yt: np.ndarray, yp: np.ndarray) -> float:
                return float(np.sqrt(np.mean(np.square(yt - yp))))

            def _nrmse(yt: np.ndarray, yp: np.ndarray) -> float:
                return _rmse(yt, yp) / mean_true * 100.0 if mean_true > 1e-6 else float("nan")

            def _mbe(yt: np.ndarray, yp: np.ndarray) -> float:
                return float(np.mean(yp - yt))

            def _nmbe(yt: np.ndarray, yp: np.ndarray) -> float:
                return _mbe(yt, yp) / mean_true * 100.0 if mean_true > 1e-6 else float("nan")

            def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
                ss_res = float(np.sum(np.square(yt - yp)))
                ss_tot = float(np.sum(np.square(yt - np.mean(yt))))
                if ss_tot <= 1e-12:
                    return 1.0 if ss_res <= 1e-12 else 0.0
                return 1.0 - ss_res / ss_tot

            m_mae  = _mae(y_true, y_model);    b_mae  = _mae(y_true, y_baseline)
            m_rmse = _rmse(y_true, y_model);   b_rmse = _rmse(y_true, y_baseline)
            metrics = {
                "daytime_rows_evaluated": int(len(ev)),
                "model_mae":    m_mae,
                "model_rmse":   m_rmse,
                "model_nrmse":  _nrmse(y_true, y_model),
                "model_mbe":    _mbe(y_true, y_model),
                "model_nmbe":   _nmbe(y_true, y_model),
                "model_r2":     _r2(y_true, y_model),
                "baseline_mae":   b_mae,
                "baseline_rmse":  b_rmse,
                "baseline_nrmse": _nrmse(y_true, y_baseline),
                "baseline_mbe":   _mbe(y_true, y_baseline),
                "baseline_nmbe":  _nmbe(y_true, y_baseline),
                "baseline_r2":    _r2(y_true, y_baseline),
                "mae_improvement_pct":  (b_mae  - m_mae)  / b_mae  * 100.0 if b_mae  > 1e-6 else 0.0,
                "rmse_improvement_pct": (b_rmse - m_rmse) / b_rmse * 100.0 if b_rmse > 1e-6 else 0.0,
            }

    return {
        "raw_rows": len(work_df),
        "clean_rows": len(clean_df),
        "resampled_rows": len(results_df),
        "results": results_df,
        "metrics": metrics,
        "selected_columns": {"timestamp": timestamp_col, "lat": latitude_col, "lon": longitude_col, "alt": altitude_col},
        "sensor_mapping": default_mapping,
    }