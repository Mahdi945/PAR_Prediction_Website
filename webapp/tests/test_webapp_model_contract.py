"""The model is fed exactly what it was trained on — and says clearly when it cannot run.

Covers core.predict: LFS-pointer detection, the status the pages show, the
training-faithful matrix preparation (median imputation, 1st/99th-percentile
clipping of the noisy channels only), batch = row-by-row, and the model card
the pages quote their error bars from.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from _common import MODEL_FEATURES
from core import predict as P
from core import features as F

needs_model = pytest.mark.skipif(not P.is_model_available(), reason="model file not available (git lfs pull)")

LFS_STUB = b"version https://git-lfs.github.com/spec/v1\noid sha256:ae1dc98c\nsize 14623511\n"


# ── Status ────────────────────────────────────────────────────────────────────

def test_lfs_pointer_stub_is_not_a_usable_model(tmp_path, monkeypatch):
    """Streamlit Cloud does not resolve Git LFS. A 133-byte pointer where the
    model should be must be named as such, not surface as a pickle error."""
    stub = tmp_path / "xgboost_model_all_locations.pkl"
    stub.write_bytes(LFS_STUB)
    monkeypatch.setattr(P, "MODEL_PATH", stub)
    assert not P.is_model_available()
    status = P.model_status()
    assert not status.ok
    assert status.state == "lfs_pointer"
    assert "pointer" in status.detail and "git lfs pull" in status.detail


def test_missing_model_file_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "MODEL_PATH", tmp_path / "nothing_here.pkl")
    status = P.model_status()
    assert status.state == "missing" and not status.ok
    assert "xgboost_model_all_locations.pkl" in status.detail


@needs_model
def test_real_model_reports_ready():
    status = P.model_status()
    assert status.ok and status.state == "ready" and status.detail == ""


# ── Training-faithful preprocessing ───────────────────────────────────────────

def test_training_bounds_and_medians_are_keyed_by_feature_name():
    bounds = P.training_bounds()
    medians = P.train_medians()
    assert set(bounds) == set(MODEL_FEATURES)
    assert set(medians) == set(MODEL_FEATURES)
    for name, (lo, hi) in bounds.items():
        assert lo <= hi, name
        assert lo <= medians[name] <= hi, name


def test_no_clip_set_matches_notebook_5():
    assert P.NO_CLIP == {"GHI_RC_01", "zenith", "elevation", "airmass", "clearness_kt", "dni",
                        "wind_sin", "wind_cos"}


def _median_row(**overrides) -> pd.DataFrame:
    row = dict(P.train_medians())
    row.update(overrides)
    return pd.DataFrame([row])


@needs_model
def test_nan_is_imputed_with_the_training_median():
    X, info = P.prepare_matrix(_median_row(Temp_WS=np.nan), return_info=True)
    i = MODEL_FEATURES.index("Temp_WS")
    assert X[0, i] == pytest.approx(P.train_medians()["Temp_WS"])
    assert info["imputed"] == {"Temp_WS": 1}
    assert np.isfinite(X).all()


@needs_model
def test_noisy_channels_are_clipped_but_radiation_is_not():
    hi_temp = P.training_bounds()["Temp_WS"][1]
    X, info = P.prepare_matrix(_median_row(Temp_WS=45.0, GHI_RC_01=1400.0), return_info=True)
    assert X[0, MODEL_FEATURES.index("Temp_WS")] == pytest.approx(hi_temp)
    assert X[0, MODEL_FEATURES.index("GHI_RC_01")] == 1400.0          # NO_CLIP
    assert info["clipped"] == {"Temp_WS": 1}


@needs_model
def test_missing_feature_column_still_raises():
    with pytest.raises(KeyError, match="dni"):
        P.prepare_matrix(_median_row().drop(columns=["dni"]))


@needs_model
def test_batch_prediction_equals_row_by_row():
    rng = np.random.default_rng(3)
    n = 12
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-06-15 08:00", periods=n, freq="45min"),
        "lat": 51.687, "lon": 14.414, "alt": 0.0,
        "GHI_RC_01": rng.uniform(100, 900, n), "Temp_WS": rng.uniform(10, 28, n),
        "RH_WS": rng.uniform(40, 90, n), "DWP_WS": rng.uniform(5, 15, n),
        "WS_WS": rng.uniform(0, 5, n), "WD_WS": rng.uniform(0, 360, n),
    })
    feats, _ = F.compute_features_batch(df, "Europe/Berlin")
    batch = P.predict_par_batch(feats)
    rows = np.array([P.predict_par(feats.iloc[[i]]) for i in range(n)])
    np.testing.assert_allclose(batch, rows, atol=1e-6)
    assert (batch >= 0).all()


@needs_model
def test_predict_par_batch_on_empty_frame():
    assert P.predict_par_batch(pd.DataFrame(columns=MODEL_FEATURES)).shape == (0,)


# ── Model card ────────────────────────────────────────────────────────────────

def test_model_card_carries_what_the_pages_show():
    card = P.model_card()
    for key in ("test_mae", "baseline_mae", "test_rmse", "baseline_rmse", "test_r2", "baseline_r2",
                "n_train", "n_val", "n_test", "n_features", "stations", "source"):
        assert key in card, key
    assert card["test_mae"] < card["baseline_mae"]
    assert card["test_r2"] > card["baseline_r2"]
    assert card["n_features"] == 15
    assert card["n_test"] > 10_000 and card["n_train"] > card["n_test"]
    assert card["stations"] == ["Laubsdorf", "Nebelin"]
    assert isinstance(card["n_train"], int)


def test_model_card_fallback_when_metrics_file_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "METRICS_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr(P, "_model_card", None)
    card = P.model_card()
    assert card["source"] == "fallback"
    assert card["test_mae"] == pytest.approx(29.43, abs=0.01)
    monkeypatch.setattr(P, "_model_card", None)      # do not leak the fallback into other tests


# ── Ensemble spread → PAR band ────────────────────────────────────────────────
# Each member is a different guess at the irradiance, so each has to go through
# feature engineering in full: clearness_kt and dni are DERIVED from GHI, and
# swapping GHI alone would leave them describing a sky no member predicted.

_WHEN = datetime(2025, 6, 15, 12, 0)
_WEATHER = {"GHI_RC_01": 600.0, "Temp_WS": 22.0, "RH_WS": 55.0, "DWP_WS": 12.0,
            "WS_WS": 2.0, "PREC_INT_WS": 0.0, "PREC_DIFF_WS": 0.0, "WD_WS": 180.0}


def _spread(members):
    return P.par_spread(members, lat=51.6872, lon=14.4143, alt=84.0,
                        when=_WHEN, weather=_WEATHER, tz_str="Europe/Berlin")


@needs_model
def test_par_spread_widens_when_the_members_disagree():
    tight = _spread([595.0, 598.0, 600.0, 602.0, 605.0])
    wide = _spread([300.0, 450.0, 600.0, 750.0, 900.0])
    assert tight and wide
    assert wide["sd"] > tight["sd"] * 3
    assert wide["high"] - wide["low"] > tight["high"] - tight["low"]


@needs_model
def test_par_spread_recomputes_the_ghi_derived_features():
    """A member at half the irradiance must move PAR, not just clearness_kt."""
    lo = _spread([300.0] * 5)
    hi = _spread([600.0] * 5)
    assert lo and hi
    assert hi["median"] > lo["median"] * 1.5


@needs_model
def test_par_spread_needs_at_least_two_members():
    assert _spread([600.0]) is None
    assert _spread([]) is None
    assert _spread([None, float("nan")]) is None
