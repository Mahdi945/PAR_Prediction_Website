"""The Dataset Upload engine: recognise real-world files, clean them like the
training data, and never fabricate.

Synthetic one-second files with German headers, decimal commas, logger
sentinels, out-of-range spikes, a north wind straddling 0°/360° and a
multi-hour outage — everything a station export tends to throw at a parser.
"""
import numpy as np
import pandas as pd
import pytest

from core import dataset as D
from core import predict as P

needs_model = pytest.mark.skipif(not P.is_model_available(), reason="model file not available (git lfs pull)")


# ── Column detection ──────────────────────────────────────────────────────────

def test_detects_german_and_english_headers():
    g = D.detect_columns(["Zeit", "Globalstrahlung", "Lufttemperatur", "rel. Feuchte",
                          "Windrichtung", "wind_speed", "PPFD", "parameter_x"])
    assert g.timestamp == "Zeit"
    assert g.sensors["GHI_RC_01"] == "Globalstrahlung"
    assert g.sensors["Temp_WS"] == "Lufttemperatur"
    assert g.sensors["RH_WS"] == "rel. Feuchte"
    assert g.sensors["WD_WS"] == "Windrichtung"
    assert g.sensors["WS_WS"] == "wind_speed"
    assert g.target == "PPFD"


def test_station_export_headers_are_recognised():
    g = D.detect_columns(["TIMESTAMP", "GHI_RC_01", "Temp_WS", "RH_WS", "DWP_WS", "WS_WS", "WD_WS",
                          "PREC_INT_WS", "PREC_DIFF_WS", "Temp_RC_01", "PAR_PAR"])
    assert g.timestamp == "TIMESTAMP" and g.target == "PAR_PAR"
    assert all(g.sensors[c] == c for c in D.SENSOR_SPECS)


def test_the_word_parameter_is_not_taken_for_par():
    g = D.detect_columns(["time", "ghi", "parameter", "transparency", "spare"])
    assert g.target is None
    assert D.detect_columns(["time", "ghi", "par_umol"]).target == "par_umol"


# ── Timezone ──────────────────────────────────────────────────────────────────

def test_timezone_is_validated_case_insensitively_with_suggestions():
    assert D.validate_timezone("europe/berlin") == "Europe/Berlin"
    assert D.validate_timezone("UTC") == "UTC"
    with pytest.raises(ValueError, match="Europe/Berlin"):
        D.validate_timezone("Berlin")
    with pytest.raises(ValueError, match="empty"):
        D.validate_timezone("")


# ── Timestamps ────────────────────────────────────────────────────────────────

def test_iso_strings_parse_without_notes():
    parsed, bad, notes = D.parse_timestamps(pd.Series(["2025-06-15 12:00:00", "2025-06-15 12:00:01"]), "Europe/Berlin")
    assert bad == 0 and notes == []
    assert parsed.iloc[0] == pd.Timestamp("2025-06-15 12:00:00")


def test_german_dotted_dates_are_read_day_first():
    parsed, _, notes = D.parse_timestamps(pd.Series(["05.06.2025 12:00", "06.06.2025 12:00"]), "Europe/Berlin")
    assert list(parsed.dt.month) == [6, 6]
    assert list(parsed.dt.day) == [5, 6]
    assert any("day-first" in n for n in notes)


def test_slash_dates_with_a_day_above_twelve_are_day_first():
    parsed, _, _ = D.parse_timestamps(pd.Series(["03/04/2025 10:00", "25/04/2025 10:00"]), "UTC")
    assert list(parsed.dt.month) == [4, 4]


def test_epoch_seconds_are_utc_and_converted_to_local():
    parsed, bad, notes = D.parse_timestamps(pd.Series([1_750_000_000]), "Europe/Berlin")
    expected = pd.Timestamp(1_750_000_000, unit="s", tz="UTC").tz_convert("Europe/Berlin").tz_localize(None)
    assert bad == 0 and parsed.iloc[0] == expected
    assert any("epoch" in n for n in notes)


def test_offset_aware_strings_are_converted_to_the_station_zone():
    parsed, _, notes = D.parse_timestamps(pd.Series(["2025-06-15T10:00:00Z", "2025-06-15T11:00:00Z"]), "Europe/Berlin")
    assert parsed.iloc[0] == pd.Timestamp("2025-06-15 12:00:00")          # CEST = UTC+2
    assert parsed.dt.tz is None
    assert any("converted" in n for n in notes)


def test_unparseable_values_are_counted_not_guessed():
    parsed, bad, _ = D.parse_timestamps(pd.Series(["2025-06-15 12:00", "junk", None]), "UTC")
    assert bad == 2 and parsed.notna().sum() == 1


def test_decimal_commas_become_numbers():
    s = pd.Series(["12,5", "13,0", "1 234,5", None])
    out = D._to_float(s)
    assert out.tolist()[:3] == [12.5, 13.0, 1234.5] and np.isnan(out.iloc[3])


# ── The whole pipeline ────────────────────────────────────────────────────────

def _station_file(hours: float = 6.0, freq: str = "1s", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-06-15 08:00", periods=int(hours * 3600 / pd.Timedelta(freq).total_seconds()), freq=freq)
    n = len(t)
    hour = t.hour + t.minute / 60
    ghi = np.clip(900 * np.sin(np.pi * (hour - 4) / 16), 0, None) + rng.normal(0, 8, n)
    return pd.DataFrame({
        "Zeit":            t.strftime("%d.%m.%Y %H:%M:%S"),
        "Globalstrahlung": ghi.round(1),
        "Lufttemperatur":  pd.Series((18 + 6 * np.sin(np.pi * (hour - 8) / 14)).round(2)).map(lambda v: str(v).replace(".", ",")),
        "rel. Feuchte":    65.0,
        "Windrichtung":    np.where(rng.random(n) < 0.5, 355.0, 5.0),      # north wind straddling 0/360
        "wind_speed":      2.5,
        "PPFD":            (ghi * 2.1 + rng.normal(0, 15, n)).round(1),
        "parameter_x":     42.0,
    })


COMMON = dict(timestamp_column="Zeit", target_column="PPFD", timezone_str="Europe/Berlin",
              latitude=51.687, longitude=14.414, altitude=84.0)


@needs_model
def test_end_to_end_on_a_messy_one_second_file():
    df = _station_file()
    df.loc[1000:1100, "Lufttemperatur"] = "-99999"        # logger sentinel → missing
    df.loc[5000:5050, "Globalstrahlung"] = 1500.0         # impossible GHI → rows dropped
    res = D.prepare_dataset_for_prediction(df, **COMMON)

    steps = {r["step"]: r for r in res["report"]}
    assert steps["Sentinel values → missing"]["note"].startswith("101 cell")
    assert steps["Out-of-range values (row dropped)"]["removed"] == 51
    assert res["clean_rows"] < res["raw_rows"]
    # 6 h of daytime seconds → about 360 one-minute bins
    assert 355 <= res["resampled_rows"] <= 361

    r = res["results"]
    assert r["timestamp"].iloc[0].month == 6 and r["timestamp"].iloc[0].day == 15    # day-first parse
    assert 17 < r["Temp_WS"].mean() < 25                                              # decimal commas parsed
    bearing = r["WD_WS"]
    assert ((bearing < 20) | (bearing > 340)).mean() > 0.95      # resultant vector → north, never 180°
    assert (r["model_prediction"] >= 0).all() and r["model_prediction"].max() > 500
    assert set(res["cleaned"].columns) == set(D.CLEANED_COLUMNS)
    assert res["features"].shape[1] == 4 + 22
    assert res["metrics"]["daytime_rows_evaluated"] > 300
    assert np.isfinite(res["metrics"]["model_mae"]) and np.isfinite(res["metrics"]["baseline_mae"])
    assert res["domain"]["location"]["nearest"] == "Laubsdorf"
    assert res["model"]["n_features"] == 15
    assert res["mapping"]["GHI_RC_01"] == "Globalstrahlung"


@needs_model
def test_absent_optional_sensor_is_named_and_the_training_median_is_used():
    df = _station_file(hours=1.0).drop(columns=["Lufttemperatur"])
    res = D.prepare_dataset_for_prediction(df, sensor_mapping={"Temp_WS": None}, **COMMON)
    assert any(w.startswith("Air temperature is not in the file") for w in res["warnings"])
    assert res["results"]["Temp_WS"].iloc[0] == pytest.approx(P.train_medians()["Temp_WS"])


def test_ghi_is_required():
    df = _station_file(hours=0.5).drop(columns=["Globalstrahlung"])
    with pytest.raises(ValueError, match="GHI"):
        D.prepare_dataset_for_prediction(df, **COMMON)


def test_timestamp_column_is_required():
    df = _station_file(hours=0.5).drop(columns=["Zeit"])
    with pytest.raises(ValueError, match="timestamp"):
        D.prepare_dataset_for_prediction(df, **{**COMMON, "timestamp_column": None})


@needs_model
def test_a_night_only_file_gives_an_honest_empty_result():
    df = _station_file(hours=1.0)
    df["Globalstrahlung"] = 10.0
    res = D.prepare_dataset_for_prediction(df, **COMMON)
    assert res["resampled_rows"] == 0 and res["results"].empty
    assert any("No daytime rows" in w for w in res["warnings"])


@needs_model
def test_hourly_data_is_scored_at_its_own_resolution():
    df = _station_file(hours=8.0, freq="1h")
    res = D.prepare_dataset_for_prediction(df, **COMMON)
    agg = next(r for r in res["report"] if r["step"].startswith("Aggregation"))
    assert "native resolution" in agg["note"]
    assert res["resampled_rows"] == res["clean_rows"]
    assert res["resolution_seconds"] == 3600.0


@needs_model
def test_long_outage_gets_the_training_median_and_is_reported():
    df = _station_file(hours=6.0, freq="1min")
    df.loc[60:299, "Lufttemperatur"] = None                # 4 h with no temperature, mid-file
    res = D.prepare_dataset_for_prediction(df, max_gap="60min", **COMMON)
    fill = res["fill_summary"]["Temp_WS"]
    # 60 min carried forward from before the gap, 60 min carried back from after it
    assert fill["missing"] == 240 and fill["carried"] == 120 and fill["median"] == 120
    assert any("training median" in w for w in res["warnings"])


@needs_model
def test_progress_callback_runs_from_start_to_completion():
    seen = []
    D.prepare_dataset_for_prediction(_station_file(hours=0.5), progress_callback=lambda p, m: seen.append(p), **COMMON)
    assert seen[0] <= 5 and seen[-1] == 100 and seen == sorted(seen)
