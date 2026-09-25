"""Exports must round-trip, respect format limits and never choke on numpy/pandas types."""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from core import export as E


@pytest.fixture
def table() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-06-15 08:00", periods=6, freq="h", tz="Europe/Berlin"),
        "GHI": [120.5, 340.0, np.nan, 810.25, 600.0, 55.0],
        "model_prediction": np.linspace(200, 1500, 6),
        "is_day": [True, True, True, True, True, False],
        "n_samples": np.arange(6, dtype="int64"),
    })


def test_csv_roundtrip_flattens_the_timezone(table):
    back = pd.read_csv(io.BytesIO(E.to_csv_bytes(table)))
    assert list(back.columns)[0] == "timestamp_local"
    assert len(back) == 6
    assert back["timestamp_local"].iloc[0] == "2025-06-15 08:00:00"      # local wall time, no offset
    assert np.isnan(back["GHI"].iloc[2])


def test_json_records_write_nan_as_null(table):
    records = json.loads(E.to_json_bytes(table))
    assert len(records) == 6
    assert records[2]["GHI"] is None
    assert records[0]["is_day"] is True


def test_json_handles_numpy_pandas_and_nan_in_structures():
    payload = {"a": np.float64(1.5), "b": np.int64(2), "t": pd.Timestamp("2025-01-01 10:00"),
               "arr": np.array([1, 2]), "nan": float("nan"), "flag": np.bool_(True),
               "frame": pd.DataFrame({"x": [1, 2]})}
    out = json.loads(E.to_json_bytes(payload))
    assert out["a"] == 1.5 and out["b"] == 2 and out["arr"] == [1, 2]
    assert out["nan"] is None and out["flag"] is True
    assert out["t"].startswith("2025-01-01T10:00")
    assert out["frame"] == [{"x": 1}, {"x": 2}]


@pytest.mark.skipif(not E.parquet_available(), reason="pyarrow not installed")
def test_parquet_roundtrip_keeps_dtypes(table):
    back = pd.read_parquet(io.BytesIO(E.to_parquet_bytes(table)))
    assert len(back) == 6
    assert str(back["timestamp"].dtype).startswith("datetime64")
    assert back["n_samples"].dtype == "int64"


@pytest.mark.skipif(not E.excel_available(), reason="openpyxl not installed")
def test_excel_roundtrip(table):
    back = pd.read_excel(io.BytesIO(E.to_excel_bytes(table)))
    assert len(back) == 6 and "timestamp_local" in back.columns
    assert back["model_prediction"].iloc[-1] == pytest.approx(1500.0)


@pytest.mark.skipif(not E.excel_available(), reason="openpyxl not installed")
def test_excel_refuses_more_rows_than_a_sheet_holds(table, monkeypatch):
    monkeypatch.setattr(E, "EXCEL_MAX_ROWS", 5)
    with pytest.raises(E.ExportError, match="Excel"):
        E.to_excel_bytes(table)
    assert "xlsx" not in E.available_formats(6)
    assert "xlsx" in E.available_formats(5)
    assert "rows" in (E.unavailable_reason("xlsx", 6) or "")


def test_available_formats_always_offers_csv_and_json():
    fmts = E.available_formats(10)
    assert fmts[0] == "csv" and "json" in fmts


def test_convert_dispatches_and_rejects_unknown(table):
    assert E.convert(table, "csv") == E.to_csv_bytes(table)
    with pytest.raises(E.ExportError):
        E.convert(table, "docx")


def test_zip_bundles_files_and_calls_callables_lazily(table):
    calls = []

    def make():
        calls.append(1)
        return E.to_csv_bytes(table)

    files = {"predictions.csv": make, "README.txt": b"hello"}
    assert calls == []                                  # nothing rendered yet
    data = E.to_zip_bytes(files)
    assert calls == [1]
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert sorted(zf.namelist()) == ["README.txt", "predictions.csv"]
        assert zf.read("README.txt") == b"hello"


def test_file_name_is_filesystem_safe_and_stamped():
    name = E.file_name("my results/2025 (v2)", "csv")
    assert name.startswith("parpredict_my_results_2025_v2_") and name.endswith(".csv")
    assert "/" not in name and " " not in name
    assert E.file_name("x", "parquet").endswith(".parquet")
    assert E.mime("json") == "application/json"
