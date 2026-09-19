"""
core/export.py
──────────────
Turn results into files a visitor can take away.

Formats: CSV (always), Excel (openpyxl, ≤ 1,048,575 rows), JSON, Parquet
(pyarrow — Streamlit itself depends on it, so it is always present in the app).

    convert(df, fmt)                 → bytes
    to_json_bytes(obj)               → bytes    DataFrame, dict or list; numpy/pandas types handled
    to_zip_bytes({name: bytes|fn})   → bytes    several files in one download
    available_formats(n_rows)        → [fmt]    what makes sense for a table this size
    file_name(base, fmt)             → "parpredict_<base>_<timestamp>.<ext>"
    download_bar(df, base, key=...)  → Streamlit UI: format picker + lazy download button
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, date
from typing import Callable, Mapping

import numpy as np
import pandas as pd


class ExportError(RuntimeError):
    """A format cannot be produced for this table (size, missing library)."""


EXCEL_MAX_ROWS = 1_048_576 - 1            # one row is the header
_TIMESTAMP_FMT = "%Y%m%d-%H%M"

FORMATS: dict[str, tuple[str, str, str]] = {
    # fmt: (label, extension, MIME type)
    "csv":     ("CSV",     ".csv",     "text/csv"),
    "xlsx":    ("Excel",   ".xlsx",    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "json":    ("JSON",    ".json",    "application/json"),
    "parquet": ("Parquet", ".parquet", "application/vnd.apache.parquet"),
}


def excel_available() -> bool:
    try:
        import openpyxl  # noqa: F401
        return True
    except ImportError:
        return False


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def available_formats(n_rows: int) -> list[str]:
    fmts = ["csv"]
    if excel_available() and n_rows <= EXCEL_MAX_ROWS:
        fmts.append("xlsx")
    fmts.append("json")
    if parquet_available():
        fmts.append("parquet")
    return fmts


def unavailable_reason(fmt: str, n_rows: int) -> str | None:
    if fmt == "xlsx":
        if not excel_available():
            return "Excel export needs the openpyxl package."
        if n_rows > EXCEL_MAX_ROWS:
            return f"Excel holds at most {EXCEL_MAX_ROWS:,} rows; this table has {n_rows:,}. Use CSV or Parquet."
    if fmt == "parquet" and not parquet_available():
        return "Parquet export needs the pyarrow package."
    if fmt not in FORMATS:
        return f"Unknown format '{fmt}'."
    return None


def file_name(base: str, fmt: str, when: datetime | None = None) -> str:
    stamp = (when or datetime.now()).strftime(_TIMESTAMP_FMT)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in base)
    safe = re.sub(r"_+", "_", safe).strip("_") or "export"
    return f"parpredict_{safe}_{stamp}{FORMATS[fmt][1]}"


def mime(fmt: str) -> str:
    return FORMATS[fmt][2]


# ── Table preparation ────────────────────────────────────────────────────────

def _flatten_time(df: pd.DataFrame) -> pd.DataFrame:
    """Excel and JSON cannot carry a timezone: write local wall time, and say so
    in the column name once (``timestamp`` → ``timestamp_local``)."""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            out[col] = s.dt.tz_localize(None)
            if col == "timestamp":
                out = out.rename(columns={col: "timestamp_local"})
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return _flatten_time(df).to_csv(index=False, lineterminator="\n").encode("utf-8")


def to_excel_bytes(sheets: Mapping[str, pd.DataFrame] | pd.DataFrame) -> bytes:
    if not excel_available():
        raise ExportError("Excel export needs the openpyxl package.")
    if isinstance(sheets, pd.DataFrame):
        sheets = {"data": sheets}
    for name, df in sheets.items():
        if len(df) > EXCEL_MAX_ROWS:
            raise ExportError(f"Sheet '{name}' has {len(df):,} rows; Excel allows {EXCEL_MAX_ROWS:,}.")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            _flatten_time(df).to_excel(xw, sheet_name=str(name)[:31] or "data", index=False)
    return buf.getvalue()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if isinstance(o, pd.Series):
        return o.to_dict()
    if isinstance(o, pd.DataFrame):
        return json.loads(o.to_json(orient="records", date_format="iso"))
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "_asdict"):
        return o._asdict()
    return str(o)


def _sanitize(obj):
    """Walk a structure and turn NaN / inf floats into None *before* json.dumps
    sees them — the encoder rejects them without ever calling ``default``."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def to_json_bytes(obj) -> bytes:
    """Records for a table; a structure otherwise. NaN becomes null."""
    if isinstance(obj, pd.DataFrame):
        return _flatten_time(obj).to_json(orient="records", date_format="iso", indent=2).encode("utf-8")
    text = json.dumps(_sanitize(obj), indent=2, ensure_ascii=False, default=_json_default, allow_nan=False)
    return text.encode("utf-8")


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    if not parquet_available():
        raise ExportError("Parquet export needs the pyarrow package.")
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    return buf.getvalue()


def convert(df: pd.DataFrame, fmt: str) -> bytes:
    reason = unavailable_reason(fmt, len(df))
    if reason:
        raise ExportError(reason)
    if fmt == "csv":
        return to_csv_bytes(df)
    if fmt == "xlsx":
        return to_excel_bytes(df)
    if fmt == "json":
        return to_json_bytes(df)
    if fmt == "parquet":
        return to_parquet_bytes(df)
    raise ExportError(f"Unknown format '{fmt}'.")    # pragma: no cover - guarded above


def to_zip_bytes(files: Mapping[str, bytes | Callable[[], bytes]]) -> bytes:
    """Bundle several files. Values may be bytes or zero-argument callables,
    so nothing is rendered until the archive is actually requested."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            data = content() if callable(content) else content
            zf.writestr(name, data)
    return buf.getvalue()


# ── Streamlit UI helper ──────────────────────────────────────────────────────

def download_bar(
    df: pd.DataFrame,
    base_name: str,
    *,
    key: str,
    label: str = "Download",
    help: str | None = None,
    formats: list[str] | None = None,
    default: str = "csv",
) -> None:
    """A format picker next to one download button.

    The bytes are produced by a callable, so Streamlit generates the file when
    the visitor clicks — not on every rerun, and never for formats nobody asked
    for. That keeps a 2-million-row result from being serialised four times
    each time a widget changes.
    """
    import streamlit as st

    n = len(df)
    fmts = formats or available_formats(n)
    if default not in fmts:
        default = fmts[0]
    c_fmt, c_btn = st.columns([1, 2.4], gap="small")
    with c_fmt:
        fmt = st.selectbox(
            "Format", fmts, index=fmts.index(default),
            format_func=lambda f: FORMATS[f][0], key=f"{key}_fmt",
            label_visibility="collapsed",
        )
    with c_btn:
        st.download_button(
            f"⬇ {label} · {n:,} rows · {FORMATS[fmt][0]}",
            data=lambda df=df, fmt=fmt: convert(df, fmt),
            file_name=file_name(base_name, fmt),
            mime=mime(fmt),
            key=f"{key}_{fmt}",
            help=help,
            use_container_width=True,
        )
    hidden = [f for f in ("xlsx", "parquet") if f not in fmts]
    for f in hidden:
        reason = unavailable_reason(f, n)
        if reason:
            st.caption(f"ℹ️ {reason}")
