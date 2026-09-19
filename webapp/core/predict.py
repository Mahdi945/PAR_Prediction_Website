"""
core/predict.py
───────────────
Model loading and PAR inference for PAR Predictor.

Public API:
    model_status()         → ModelStatus(ok, state, detail)   precise reason when not ok
    is_model_available()   → bool                             cheap file check
    load_model()           → (model, feature_names)
    predict_par(df)        → float          [µmol/m²/s]  one row
    predict_par_batch(df)  → np.ndarray     [µmol/m²/s]  many rows, one model call
    prepare_matrix(df)     → np.ndarray     exactly what the model saw in training
    training_bounds()      → {feature: (low, high)}  fitted on the training split
    train_medians()        → {feature: median}        imputation values
    model_card()           → dict           held-out metrics, sizes, stations
    mccree_estimate(ghi)   → float          [µmol/m²/s]  physics baseline
"""

from __future__ import annotations

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import NamedTuple

from .constants import MCCREE_FACTOR

# ════════════════════════════════════════════════════════════════════════════════
#  XGBEnsemble stub — must be defined BEFORE joblib.load()
#  The training notebook defined this class in its __main__ scope.
#  joblib pickles the class reference as "__main__.XGBEnsemble", so we must
#  inject the same class into __main__ before unpickling.
# ════════════════════════════════════════════════════════════════════════════════
class XGBEnsemble:
    """
    Ensemble of XGBoost models trained with different random seeds.
    Predictions are averaged across all seed models.
    """
    def __init__(self, models=None, **kwargs):
        self.models = models if models is not None else []
        # Accept any extra attributes that were saved (clip bounds, etc.)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def predict(self, X):
        if not self.models:
            return np.zeros(len(X))
        preds = np.mean([m.predict(X) for m in self.models], axis=0)
        # Apply clip bounds if stored
        clip_min = getattr(self, "clip_min", None)
        clip_max = getattr(self, "clip_max", None)
        if clip_min is not None or clip_max is not None:
            preds = np.clip(preds, clip_min, clip_max)
        return preds

    @property
    def feature_importances_(self):
        if not self.models:
            return None
        try:
            imps = np.array([m.feature_importances_ for m in self.models])
            return imps.mean(axis=0)
        except Exception:
            return None

# Inject into __main__ so joblib can resolve the class during unpickling
import __main__ as _main_module
if not hasattr(_main_module, "XGBEnsemble"):
    setattr(_main_module, "XGBEnsemble", XGBEnsemble)

# ── Paths (resolved relative to this file → project root) ───────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent   # project root

MODEL_PATH = _ROOT / "data" / "results" / "xgboost_model_all_locations.pkl"
METRICS_PATH = _ROOT / "data" / "results" / "xgboost_metrics_all_locations.pkl"
_PKL_DIR = _ROOT / "data" / "processed" / "pkl_features_GradientBoosting"
FEATURE_NAMES_PATH  = _PKL_DIR / "feature_names_all_locations.pkl"
CLIP_BOUNDS_PATH    = _PKL_DIR / "clip_bounds_all_locations.pkl"
TRAIN_MEDIANS_PATH  = _PKL_DIR / "train_medians_all_locations.pkl"

# ── Fallback feature list — the 15 features the model was trained on, in order.
#    Must match feature_names_all_locations.pkl (written by 5_stratification_All_Files.ipynb).
_FALLBACK_FEATURES = [
    "GHI_RC_01", "Temp_WS", "RH_WS", "DWP_WS", "WS_WS",
    "PREC_INT_WS", "PREC_DIFF_WS", "Temp_RC_merged",
    "zenith", "elevation", "airmass", "clearness_kt", "dni",
    "wind_sin", "wind_cos",
]

# Notebook 5 fitted these on the training split only, then applied them to the
# validation and test splits before scoring. The app applies them at inference
# so the model is fed exactly what it was evaluated on. The fallbacks are the
# values in the pkl files as of the 2026-09-16 training run; the files win
# whenever they are present.
#
# Radiation and geometry were deliberately NOT clipped (their extremes are
# physically meaningful); for those the bounds are the plain training min/max
# and are used only by core.domain to flag out-of-range inputs.
NO_CLIP = {
    "GHI_RC_01", "zenith", "elevation", "airmass", "clearness_kt", "dni",
    "wind_sin", "wind_cos",
}
_FALLBACK_CLIP_BOUNDS = {
    "GHI_RC_01":      (30.100000381469727, 1314.7729789115288),
    "Temp_WS":        (-2.299999952316284, 30.5),
    "RH_WS":          (27.109999865531922, 100.0),
    "DWP_WS":         (-6.947169861703549, 18.646666813532512),
    "WS_WS":          (0.0, 5.328333275318146),
    "PREC_INT_WS":    (0.0, 0.800000011920929),
    "PREC_DIFF_WS":   (0.0, 0.0),
    "Temp_RC_merged": (-0.5533333460489909, 49.14230793219346),
    "zenith":         (28.24161002588489, 87.99137979676603),
    "elevation":      (1.7074383149232557, 61.74934880798415),
    "airmass":        (1.1344695451167888, 19.387471965437523),
    "clearness_kt":   (0.0265734169374535, 1.0),
    "dni":            (0.0, 978.8169917691856),
    "wind_sin":       (-0.9999619230641712, 0.9999619230641712),
    "wind_cos":       (-0.9994832608221976, 1.0),
}
_FALLBACK_TRAIN_MEDIANS = {
    "GHI_RC_01": 183.22845167, "Temp_WS": 14.69999981, "RH_WS": 69.56000005,
    "DWP_WS": 8.08269262, "WS_WS": 1.79230766, "PREC_INT_WS": 0.0,
    "PREC_DIFF_WS": 0.0, "Temp_RC_merged": 19.65576966, "zenith": 64.26673346,
    "elevation": 25.69859518, "airmass": 2.29389216, "clearness_kt": 0.33620073,
    "dni": 59.92854889, "wind_sin": -0.50072304, "wind_cos": -0.20882766,
}

# Held-out results of the deployed model (xgboost_metrics_all_locations.pkl,
# 2026-09-16). Used when the metrics file is not shipped with the app.
_FALLBACK_MODEL_CARD = {
    "model":        "XGBoost (Optuna + 3-seed ensemble)",
    "split":        "day-grouped, stratified by station x month (70/15/15)",
    "test_r2":      0.9896320806082648,
    "baseline_r2":  0.9850754985800821,
    "test_mae":     29.43461816840608,
    "baseline_mae": 36.13830358712322,
    "test_rmse":    50.255379895220074,
    "baseline_rmse": 60.295745632049766,
    "test_nrmse":   8.57195048064841,
    "test_mbe":     2.0001707142826874,
    "mae_gain_pct": 18.550083300273656,
    "rmse_gain_pct": 16.65186429254937,
    "n_train":      189660,
    "n_val":        41700,
    "n_test":       42031,
    "n_features":   15,
}
TRAINING_STATIONS = ("Laubsdorf", "Nebelin")   # both in Brandenburg, Germany

# ── Module-level cache ───────────────────────────────────────────────────────
# Modules stay in sys.modules across Streamlit reruns, so these globals act as
# a process-wide cache — the same lifetime st.cache_resource would give, but
# usable from plain Python (tests, scripts) as well.
_model         = None
_feature_names = None
_clip_bounds:   dict | None = None
_medians:       dict | None = None
_model_card:    dict | None = None


class ModelStatus(NamedTuple):
    ok: bool
    state: str      # "ready" | "missing" | "lfs_pointer" | "load_error"
    detail: str


# ── Public helpers ────────────────────────────────────────────────────────────

def _is_lfs_pointer(path: Path) -> bool:
    """An un-pulled Git LFS file is a ~130-byte text stub, not the model.

    Loading it gives an opaque joblib/pickle error; detecting it first lets the
    pages say exactly what went wrong."""
    try:
        with path.open("rb") as fh:
            return fh.read(40).startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def _load_joblib_robust(path):
    """
    Load a joblib file saved from a Jupyter notebook, auto-patching any
    missing __main__ symbols (custom classes, metric functions, etc.).

    Notebooks define helpers in __main__ scope; joblib pickles them as
    "__main__.symbol_name". We inject stubs iteratively until the load
    succeeds — each attempt reveals the NEXT missing symbol.
    """
    import re, __main__ as _main

    for _attempt in range(30):
        try:
            return joblib.load(path)
        except AttributeError as exc:
            msg = str(exc)
            if "__main__" not in msg or "has no attribute" not in msg:
                raise       # different error — re-raise
            m = re.search(r"attribute '([^']+)'", msg)
            if not m:
                raise
            missing = m.group(1)

            # ── Known symbols with proper implementations ──────────────────
            if missing == "XGBEnsemble":
                setattr(_main, missing, XGBEnsemble)

            elif missing in ("xgb_r2", "r2_score_xgb", "custom_r2",
                             "xgb_rmse", "custom_rmse", "xgb_mae"):
                # XGBoost custom eval metric — only used during training,
                # never called during inference.  Return a valid (name, float).
                _name = missing
                setattr(_main, missing,
                        lambda y_pred, dtrain, _n=_name: (_n, 0.0))

            else:
                # Unknown symbol: fail loudly. A generic stub whose predict()
                # returns zeros would let the app display PAR = 0 with no error.
                raise RuntimeError(
                    f"Model file references an unknown notebook symbol '{missing}'. "
                    "Retrain with 6_model_training_All_Files.ipynb (or "
                    "src/run_pipeline_xgboost.py --run --only training) so the pickle "
                    "only depends on XGBEnsemble."
                ) from exc
    raise RuntimeError(
        f"Could not load model after 30 attempts. "
        f"Too many missing __main__ symbols in {path}"
    )


def is_model_available() -> bool:
    """True when a real model file (not an LFS stub) exists on disk."""
    return MODEL_PATH.exists() and not _is_lfs_pointer(MODEL_PATH)


def model_status() -> ModelStatus:
    """Why the model can or cannot be used — for the pages' status banners."""
    if not MODEL_PATH.exists():
        return ModelStatus(False, "missing",
                           f"No model file at {MODEL_PATH}. Locally: run `git lfs pull` "
                           "from the project root. On a deployment: the file "
                           "data/results/xgboost_model_all_locations.pkl must be in the repository.")
    if _is_lfs_pointer(MODEL_PATH):
        return ModelStatus(False, "lfs_pointer",
                           f"{MODEL_PATH.name} is a Git LFS pointer ({MODEL_PATH.stat().st_size} bytes), "
                           "not the model itself. Locally: `git lfs pull`. On a host that does "
                           "not resolve LFS, commit the real file instead of the pointer.")
    try:
        load_model()
    except Exception as exc:                          # pragma: no cover - environment specific
        return ModelStatus(False, "load_error", f"{type(exc).__name__}: {exc}")
    return ModelStatus(True, "ready", "")


def load_model():
    """
    Load (and cache) the XGBoost model + feature name list.
    Raises FileNotFoundError if the model pkl is missing.
    """
    global _model, _feature_names

    if _model is not None:
        return _model, _feature_names

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at:\n  {MODEL_PATH}\n\n"
            "Run  git lfs pull  to download model files from Git LFS."
        )
    if _is_lfs_pointer(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} is a Git LFS pointer stub, not the model. "
            "Run  git lfs pull  (or commit the real file to the deployment repository)."
        )

    _model = _load_joblib_robust(MODEL_PATH)

    if FEATURE_NAMES_PATH.exists() and not _is_lfs_pointer(FEATURE_NAMES_PATH):
        _feature_names = list(joblib.load(FEATURE_NAMES_PATH))
    else:
        _feature_names = list(_FALLBACK_FEATURES)

    return _model, _feature_names


def _feature_list() -> list[str]:
    if _feature_names is not None:
        return list(_feature_names)
    if FEATURE_NAMES_PATH.exists() and not _is_lfs_pointer(FEATURE_NAMES_PATH):
        return list(joblib.load(FEATURE_NAMES_PATH))
    return list(_FALLBACK_FEATURES)


def training_bounds() -> dict[str, tuple[float, float]]:
    """(low, high) per model feature, fitted on the training split.

    Clipped channels: 1st/99th percentile. Unclipped channels (see NO_CLIP):
    training min/max. Keyed by feature name."""
    global _clip_bounds
    if _clip_bounds is None:
        names = _feature_list()
        if CLIP_BOUNDS_PATH.exists() and not _is_lfs_pointer(CLIP_BOUNDS_PATH):
            raw = joblib.load(CLIP_BOUNDS_PATH)          # {position: (lo, hi)}
            _clip_bounds = {names[i]: (float(lo), float(hi)) for i, (lo, hi) in raw.items()
                            if i < len(names)}
        else:
            _clip_bounds = dict(_FALLBACK_CLIP_BOUNDS)
    return dict(_clip_bounds)


def train_medians() -> dict[str, float]:
    """Training-split median per model feature — what NaN was imputed with."""
    global _medians
    if _medians is None:
        names = _feature_list()
        if TRAIN_MEDIANS_PATH.exists() and not _is_lfs_pointer(TRAIN_MEDIANS_PATH):
            raw = np.asarray(joblib.load(TRAIN_MEDIANS_PATH), dtype="float64")
            _medians = {names[i]: float(raw[i]) for i in range(min(len(names), len(raw)))}
        else:
            _medians = dict(_FALLBACK_TRAIN_MEDIANS)
    return dict(_medians)


def prepare_matrix(features_df: pd.DataFrame, *, return_info: bool = False):
    """
    Turn a feature table into the float64 matrix the model expects.

    1. Columns in training order. A missing column raises instead of being
       silently filled with 0 — a zero-filled feature would produce a
       confident but wrong prediction.
    2. NaN → training median (notebook 5, cell 28).
    3. Noisy meteorological channels clipped to their training 1st/99th
       percentiles (notebook 5, cell 26). Radiation and geometry untouched.

    With ``return_info=True`` also returns ``{"imputed": {feature: n}, "clipped": {feature: n}}``.
    """
    _, names = load_model()
    missing = [c for c in names if c not in features_df.columns]
    if missing:
        raise KeyError(
            f"Feature(s) required by the model are missing: {missing}. "
            "core.features.compute_features() and the training notebooks are out of sync."
        )

    X = features_df.loc[:, list(names)].to_numpy(dtype="float64", copy=True)
    medians = train_medians()
    bounds  = training_bounds()
    info = {"imputed": {}, "clipped": {}}

    for i, name in enumerate(names):
        col = X[:, i]
        nan = ~np.isfinite(col)
        if nan.any():
            col[nan] = medians.get(name, 0.0)
            info["imputed"][name] = int(nan.sum())
        if name not in NO_CLIP and name in bounds:
            lo, hi = bounds[name]
            out = (col < lo) | (col > hi)
            if out.any():
                np.clip(col, lo, hi, out=col)
                info["clipped"][name] = int(out.sum())

    return (X, info) if return_info else X


def predict_par(features_df: pd.DataFrame) -> float:
    """
    Predict PAR [µmol/m²/s] for a one-row feature DataFrame
    (see core.features.compute_features()). Extra columns are ignored.
    """
    model, _ = load_model()
    pred = model.predict(prepare_matrix(features_df))[0]
    return float(max(0.0, pred))


def predict_par_batch(features_df: pd.DataFrame) -> np.ndarray:
    """PAR [µmol/m²/s] for every row — one model call, negative values clipped to 0."""
    if len(features_df) == 0:
        return np.zeros(0, dtype="float64")
    model, _ = load_model()
    pred = np.asarray(model.predict(prepare_matrix(features_df)), dtype="float64")
    return np.maximum(0.0, pred)


def get_feature_importance() -> pd.Series | None:
    """
    Return a pd.Series of feature importances (index = feature names)
    if the model exposes `feature_importances_`, else None.
    """
    try:
        model, feature_names = load_model()
        if hasattr(model, "feature_importances_"):
            return pd.Series(
                model.feature_importances_, index=feature_names
            ).sort_values(ascending=False)
    except Exception:
        pass
    return None


def model_card() -> dict:
    """Held-out performance and training-set facts for the deployed model.

    Read from xgboost_metrics_all_locations.pkl when it ships with the app,
    otherwise from the recorded fallback. ``source`` says which."""
    global _model_card
    if _model_card is None:
        card = dict(_FALLBACK_MODEL_CARD)
        source = "fallback"
        if METRICS_PATH.exists() and not _is_lfs_pointer(METRICS_PATH):
            try:
                saved = joblib.load(METRICS_PATH)
                for k in _FALLBACK_MODEL_CARD:
                    if k in saved:
                        v = saved[k]
                        card[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) \
                            and not isinstance(v, bool) else v
                source = METRICS_PATH.name
            except Exception:                         # pragma: no cover - corrupt file
                pass
        for k in ("n_train", "n_val", "n_test", "n_features"):
            card[k] = int(card[k])
        card["stations"] = list(TRAINING_STATIONS)
        card["source"] = source
        _model_card = card
    return dict(_model_card)


def mccree_estimate(ghi: float) -> float:
    """
    Classic McCree baseline  [µmol/m²/s].

    PAR (µmol/m²/s) = GHI (W/m²) × PAR_ENERGY_FRACTION  (0.45, dimensionless)
                                  × PAR_QUANTUM_EFFICACY (4.57 µmol/J)
                    = GHI × MCCREE_FACTOR

    The factor lives in core.constants so the app, the notebooks and the
    pipeline cannot drift apart — see tests/test_mccree_factor.py.
    """
    return max(0.0, ghi * MCCREE_FACTOR)
