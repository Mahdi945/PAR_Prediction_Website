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
FEATURE_NAMES_PATH = (
    _ROOT / "data" / "processed"
    / "pkl_features_GradientBoosting"
    / "feature_names_all_locations.pkl"
)

# ── Fallback feature list — the 15 features the model was trained on, in order.
#    Must match feature_names_all_locations.pkl (written by 5_stratification_All_Files.ipynb).
_FALLBACK_FEATURES = [
    "GHI_RC_01", "Temp_WS", "RH_WS", "DWP_WS", "WS_WS",
    "PREC_INT_WS", "PREC_DIFF_WS", "Temp_RC_merged",
    "zenith", "elevation", "airmass", "clearness_kt", "dni",
    "wind_sin", "wind_cos",
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

    The DataFrame must contain every column the model was trained on
    (see core.features.compute_features()).  Extra columns are ignored.
    A missing column raises instead of being silently filled with 0 —
    a zero-filled feature would produce a confident but wrong prediction.
    """
    model, feature_names = load_model()

    missing = [c for c in feature_names if c not in features_df.columns]
    if missing:
        raise KeyError(
            f"Feature(s) required by the model are missing: {missing}. "
            "core.features.compute_features() and the training notebooks are out of sync."
        )

    X = features_df.loc[:, list(feature_names)].to_numpy(dtype="float64")
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

    PAR (µmol/m²/s) = GHI (W/m²) × PAR_ENERGY_FRACTION  (0.45, dimensionless)
                                  × PAR_QUANTUM_EFFICACY (4.57 µmol/J)
                    = GHI × MCCREE_FACTOR

    The factor lives in core.constants so the app, the notebooks and the
    pipeline cannot drift apart — see tests/test_mccree_factor.py.
    """
    return max(0.0, ghi * MCCREE_FACTOR)
