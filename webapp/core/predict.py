"""
core/predict.py
───────────────
Model loading and PAR inference for PAR Predictor.

Public API:
    is_model_available() → bool
    load_model()         → (model, feature_names)
    predict_par(df)      → float   [µmol/m²/s]
    mccree_estimate(ghi) → float   [µmol/m²/s]  simple baseline
"""

from __future__ import annotations

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

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
FEATURE_NAMES_PATH = (
    _ROOT / "data" / "processed"
    / "pkl_features_GradientBoosting"
    / "feature_names_all_locations.pkl"
)

# ── Fallback feature list (mirrors training notebooks) ──────────────────────
_FALLBACK_FEATURES = [
    "GHI_RC_01", "Temp_WS", "RH_WS", "WS_WS", "DWP_WS",
    "PREC_WS", "PREC_INT_WS", "Temp_RC_merged",
    "zenith", "airmass", "clearness_kt",
    "wind_sin", "wind_cos",
    "is_raining", "GHI_rolling_5min", "temp_diff", "dew_depression",
]

# ── Module-level cache ───────────────────────────────────────────────────────
_model         = None
_feature_names = None


# ── Public helpers ────────────────────────────────────────────────────────────

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
                # Generic fallback: callable that returns None
                setattr(_main, missing,
                        type(missing, (), {
                            "__init__": lambda s, *a, **k: None,
                            "__call__": lambda s, *a, **k: None,
                            "predict":  lambda s, X: np.zeros(len(X)),
                        })())
    raise RuntimeError(
        f"Could not load model after 30 attempts. "
        f"Too many missing __main__ symbols in {path}"
    )


def is_model_available() -> bool:
    """Return True if the model .pkl file exists on disk."""
    return MODEL_PATH.exists()


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

    _model = _load_joblib_robust(MODEL_PATH)

    if FEATURE_NAMES_PATH.exists():
        _feature_names = joblib.load(FEATURE_NAMES_PATH)
    else:
        _feature_names = _FALLBACK_FEATURES

    return _model, _feature_names


def predict_par(features_df: pd.DataFrame) -> float:
    """
    Predict PAR [µmol/m²/s] from a feature DataFrame.

    The DataFrame must contain at least the columns returned by
    core.features.compute_features().  Extra columns are ignored;
    missing columns are filled with 0.
    """
    model, feature_names = load_model()

    # Build ordered feature matrix
    X = pd.DataFrame(index=features_df.index)
    for col in feature_names:
        X[col] = features_df[col] if col in features_df.columns else 0.0

    pred = model.predict(X)[0]
    return float(max(0.0, pred))


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


def mccree_estimate(ghi: float) -> float:
    """
    Classic McCree baseline  [µmol/m²/s].

    PAR (µmol/m²/s) = GHI (W/m²) × 0.45 (PAR energy fraction)
                                  × 4.57 (µmol/J conversion for solar spectrum)
                    ≈ GHI × 2.06
    """
    return max(0.0, ghi * 2.06)
