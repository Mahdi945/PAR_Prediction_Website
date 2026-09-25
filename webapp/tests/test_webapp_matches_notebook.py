"""Integration check: the web app must reproduce the notebook's own predictions.

Replays rows from data/results/test_predictions_all_locations.csv (written by
6_model_training_All_Files.ipynb) through core.features + core.predict and
compares against the prediction the notebook recorded for the same row.

Skips when the pipeline artefacts are not available (e.g. CI without Git LFS).
"""
import numpy as np
import pandas as pd
import pytest

from _common import ROOT

PREDICTIONS = ROOT / "data" / "results" / "test_predictions_all_locations.csv"
FEATURES = ROOT / "data" / "processed" / "features_GradientBoosting" / "all_locations_all_months_features.csv"
MODEL = ROOT / "data" / "results" / "xgboost_model_all_locations.pkl"

COORDS = {
    "Laubsdorf": (51.68718711157154, 14.414256224166728),
    "Nebelin": (53.11833921379164, 11.746088890223463),
}
N_ROWS = 60
SOLAR = ("zenith", "elevation", "airmass", "clearness_kt", "dni")


def _available(path) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as fh:                       # un-pulled LFS files are text stubs
        return not fh.read(40).startswith(b"version https://git-lfs.github.com/spec")


pytestmark = pytest.mark.skipif(
    not all(_available(p) for p in (PREDICTIONS, FEATURES, MODEL)),
    reason="pipeline artefacts not available (run `git lfs pull`)",
)


@pytest.fixture(scope="module")
def replayed() -> pd.DataFrame:
    from core import features as F, predict as P

    pred = pd.read_csv(PREDICTIONS, parse_dates=["TIMESTAMP"])
    feat = pd.read_csv(FEATURES, parse_dates=["TIMESTAMP"])
    sample = (pred.merge(feat, on=["TIMESTAMP", "location"], how="left", suffixes=("", "_f"))
                  .sample(N_ROWS, random_state=0).reset_index(drop=True))

    rows = []
    for _, r in sample.iterrows():
        lat, lon = COORDS[r["location"]]
        bearing = float(np.degrees(np.arctan2(r["wind_sin"], r["wind_cos"])) % 360)
        weather = dict(GHI_RC_01=r["GHI_RC_01"], Temp_WS=r["Temp_WS"], RH_WS=r["RH_WS"],
                       DWP_WS=r["DWP_WS"], WS_WS=r["WS_WS"], WD_WS=bearing,
                       PREC_INT_WS=r["PREC_INT_WS"], PREC_DIFF_WS=r["PREC_DIFF_WS"],
                       Temp_RC_01=r["Temp_RC_merged"])
        df, is_day = F.compute_features(lat, lon, 0.0, r["TIMESTAMP"].to_pydatetime(),
                                        weather, "Europe/Berlin")
        if not is_day:
            continue
        rows.append({
            "app": P.predict_par(df),
            "notebook": r["y_pred"],
            "actual": r["PAR_PAR"],
            "baseline": r["y_base"],
            **{f"app_{c}": df[c].iloc[0] for c in SOLAR},
            **{f"nb_{c}": r[c] for c in SOLAR},
        })

    assert rows, "no daytime rows were replayed"
    return pd.DataFrame(rows)


@pytest.mark.parametrize("feature", SOLAR)
def test_solar_geometry_matches_the_notebook_exactly(replayed, feature):
    """Same timezone handling and the same pvlib calls => agreement to machine precision.

    This is what catches a regression of the naive-timestamp bug, where pvlib
    silently treated local German timestamps as UTC.
    """
    drift = (replayed[f"app_{feature}"] - replayed[f"nb_{feature}"]).abs().max()
    assert drift < 1e-6, f"{feature} drifted by up to {drift:.6g}"


def test_predictions_track_the_notebook(replayed):
    """Residual differences come only from the wind encoding: the notebook stores a
    resultant (vector) mean with radius <= 1, the app re-encodes a single bearing."""
    diff = (replayed["app"] - replayed["notebook"]).abs()
    assert diff.mean() < 5.0, f"mean divergence {diff.mean():.2f} umol/m2/s"
    assert replayed["app"].corr(replayed["notebook"]) > 0.999


def test_app_is_as_accurate_as_the_notebook(replayed):
    app_mae = (replayed["app"] - replayed["actual"]).abs().mean()
    nb_mae = (replayed["notebook"] - replayed["actual"]).abs().mean()
    assert app_mae < nb_mae * 1.15, f"app MAE {app_mae:.2f} vs notebook {nb_mae:.2f}"


# Note: "the model beats the physics baseline" is deliberately NOT asserted here.
# The margin (+18.6 % MAE) is a property of the full 42,031-row test set and is
# concentrated in clear-sky conditions; on a 60-row replay sample it sits inside
# sampling noise and would flap. That claim is tested at full scale against the
# saved metrics in test_model_artifacts.py::test_model_beats_physics_baseline_on_test_days.


def test_noct_fallback_is_close_to_the_measured_sensor(replayed):
    """Pages 1 and 2 have no reference-cell sensor, so Temp_RC is estimated from
    air temperature and GHI. That approximation must not move PAR much."""
    from core import features as F, predict as P

    pred = pd.read_csv(PREDICTIONS, parse_dates=["TIMESTAMP"])
    feat = pd.read_csv(FEATURES, parse_dates=["TIMESTAMP"])
    sample = (pred.merge(feat, on=["TIMESTAMP", "location"], how="left", suffixes=("", "_f"))
                  .sample(25, random_state=7).reset_index(drop=True))

    shifts = []
    for _, r in sample.iterrows():
        lat, lon = COORDS[r["location"]]
        bearing = float(np.degrees(np.arctan2(r["wind_sin"], r["wind_cos"])) % 360)
        base = dict(GHI_RC_01=r["GHI_RC_01"], Temp_WS=r["Temp_WS"], RH_WS=r["RH_WS"],
                    DWP_WS=r["DWP_WS"], WS_WS=r["WS_WS"], WD_WS=bearing,
                    PREC_INT_WS=r["PREC_INT_WS"], PREC_DIFF_WS=r["PREC_DIFF_WS"])
        ts = r["TIMESTAMP"].to_pydatetime()
        with_sensor, day = F.compute_features(lat, lon, 0.0, ts,
                                              {**base, "Temp_RC_01": r["Temp_RC_merged"]},
                                              "Europe/Berlin")
        with_noct, _ = F.compute_features(lat, lon, 0.0, ts,
                                          {**base, "Temp_RC_01": None}, "Europe/Berlin")
        if day:
            shifts.append(abs(P.predict_par(with_sensor) - P.predict_par(with_noct)))

    assert shifts, "no daytime rows"
    assert np.mean(shifts) < 15.0, f"NOCT fallback shifts PAR by {np.mean(shifts):.1f} on average"
