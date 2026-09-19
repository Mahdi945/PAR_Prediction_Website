"""
pages/3_Dataset_Upload.py  –  ParPredict · Dataset Upload Mode
────────────────────────────────────────────────────────
Upload a sensor file, map its columns, clean it the way the training data was
cleaned, score every minute with the model, compare against the McCree
baseline — and take everything home: predictions, the cleaned file, the
feature matrix and a JSON report, in CSV / Excel / JSON / Parquet.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as _st_components
import plotly.graph_objects as go

from core.dataset import (
    prepare_dataset_for_prediction, detect_columns, validate_timezone,
    SENSOR_SPECS, DEFAULT_MAX_GAP,
)
from core.predict import get_feature_importance, model_status, model_card
from core.cache import geocode_city
from core.export import download_bar, to_json_bytes, to_csv_bytes, to_zip_bytes, file_name

# ── Limits — keep MAX_UPLOAD_MB equal to server.maxUploadSize in .streamlit/config.toml
MAX_UPLOAD_MB = 200
MAX_ROWS      = 3_000_000       # ≈ 35 days of one-second data; a month per file is comfortable
PREVIEW_ROWS  = 2_000
NONE_LABEL    = "— not in my file —"
CHOOSE_LABEL  = "— choose a column —"


# ═════════════════════════════════════════════════════════════════════════════
#  File helpers
# ═════════════════════════════════════════════════════════════════════════════

def _file_key(f) -> str:
    return str(getattr(f, "file_id", None) or f"{f.name}-{f.size}")


def _is_excel(name: str) -> bool:
    return name.lower().endswith((".xlsx", ".xls"))


def _sniff_csv(f) -> dict:
    """Delimiter and encoding from the first 64 KB — German exports use ';'."""
    f.seek(0)
    head = f.read(65536)
    f.seek(0)
    encoding = "utf-8-sig"
    try:
        sample = head.decode("utf-8-sig")
    except UnicodeDecodeError:
        encoding = "latin-1"
        sample = head.decode("latin-1", errors="replace")
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        sep = max(",;\t|", key=first.count) if first else ","
    return {"sep": sep, "encoding": encoding}


def _read_preview(f, info: dict) -> pd.DataFrame:
    f.seek(0)
    if _is_excel(f.name):
        return pd.read_excel(f, nrows=PREVIEW_ROWS)
    return pd.read_csv(f, sep=info["sep"], encoding=info["encoding"], nrows=PREVIEW_ROWS)


def _read_full(f, info: dict, usecols: list[str]) -> pd.DataFrame:
    """Only the mapped columns are read — a 15-column file with 4 mapped columns
    needs about a quarter of the memory."""
    f.seek(0)
    if _is_excel(f.name):
        return pd.read_excel(f, usecols=usecols)
    return pd.read_csv(f, sep=info["sep"], encoding=info["encoding"], usecols=usecols, low_memory=False)


def _infer_city(filename: str) -> str | None:
    base = Path(filename).stem
    city = base.split("_")[0].strip() if base else ""
    return city if city and not city[0].isdigit() else None


def _run_with_progress(call_args: dict) -> dict:
    progress_text = st.empty()
    progress_bar = st.progress(0)

    def _on_progress(pct: int, message: str) -> None:
        progress_bar.progress(max(0, min(100, int(pct))))
        progress_text.caption(f"{message} — {pct}%")

    _on_progress(1, "Starting")
    result = prepare_dataset_for_prediction(**call_args, progress_callback=_on_progress)
    progress_bar.empty()
    progress_text.empty()
    return result


def _fmt_seconds(s: float | None) -> str:
    if s is None:
        return "—"
    if s < 60:
        return f"{s:.0f} s"
    if s < 3600:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


# ═════════════════════════════════════════════════════════════════════════════
#  Page
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Dataset Upload · ParPredict",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
.panel-title {
    font-size: .78rem; font-weight: 700; color: #2ecc71;
    text-transform: uppercase; letter-spacing: 1.5px;
    border-left: 3px solid #2ecc71; padding-left: .5rem;
    margin: .9rem 0 .7rem 0;
}
.welcome-card {
    background:#1a1d2e; border:1px dashed #2a2d3e; border-radius:16px;
    padding:4rem 2rem; text-align:center; color:#8892b0;
}
.pill {
    display:inline-block; background:#1a1d2e; border:1px solid #2a2d3e; border-radius:8px;
    padding:.15rem .55rem; font-size:.78rem; color:#e8eaf6; margin:.15rem .2rem .15rem 0;
}
.pill b { color:#2ecc71; }
@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .welcome-card { padding: 2.5rem 1.2rem; }
    div[data-testid="column"] > div:first-child { min-width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

for _k, _v in {"dataset_result": None, "dataset_token": None, "dataset_file_key": None}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.markdown("# 📊 Dataset Upload Mode")
st.caption("Upload a sensor file, clean it the way the training data was cleaned, "
           "score every minute with the model, and download the results in the format you need.")
st.divider()

_status = model_status()
if not _status.ok:
    st.error(f"⚠️ The prediction model is not usable here. {_status.detail}")
    st.stop()
_card = model_card()

# ─────────────────────────────────────────────────────────────────────────────
#  1 · Upload
# ─────────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="panel-title">📁 1 · Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a CSV, TXT or Excel file",
        type=["csv", "txt", "xlsx", "xls"],
        help="One row per measurement. Needs a timestamp and a GHI column; other weather "
             "sensors improve the prediction, and a measured PAR column enables the accuracy report.",
        accept_multiple_files=False,
    )
    st.caption(f"Up to {MAX_UPLOAD_MB} MB and {MAX_ROWS:,} rows per file · any delimiter · "
               f"decimal commas are fine · timestamps in local time, UTC, or Unix epoch.")

    if uploaded_file is None:
        st.markdown("""
        <div class="welcome-card">
            <div style="font-size:3rem;margin-bottom:1rem">📊</div>
            <div style="font-size:1.15rem;font-weight:700;color:#fff;margin-bottom:.8rem">Ready to score your own data</div>
            <div style="font-size:.9rem;line-height:1.75">
                Upload a file with <strong style="color:#2ecc71">timestamps</strong> and
                <strong style="color:#2ecc71">GHI</strong> (plus any weather sensors you have).<br>
                The page cleans it like the training data, resamples to one-minute values, computes the
                22 features, runs the model and compares it with the McCree baseline.<br><br>
                Everything can be downloaded afterwards — predictions, the cleaned file, the feature matrix
                and a JSON report — as <strong style="color:#2ecc71">CSV, Excel, JSON or Parquet</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # a new file invalidates the previous result
    fkey = _file_key(uploaded_file)
    if st.session_state.dataset_file_key != fkey:
        st.session_state.dataset_file_key = fkey
        st.session_state.dataset_result = None
        st.session_state.dataset_token = None

    size_mb = uploaded_file.size / 1_048_576
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"This file is {size_mb:,.0f} MB; the limit is {MAX_UPLOAD_MB} MB. "
                 "Split it (for example one month per file) and upload the parts one by one.")
        st.stop()

    try:
        read_info = {} if _is_excel(uploaded_file.name) else _sniff_csv(uploaded_file)
        preview = _read_preview(uploaded_file, read_info)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()
    if preview.empty or len(preview.columns) < 2:
        st.error("The file has fewer than two columns — it needs at least a timestamp and a GHI column.")
        st.stop()

    sep_label = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}.get(read_info.get("sep", ","), "?")
    st.success(f"**{uploaded_file.name}** · {size_mb:,.1f} MB · {len(preview.columns)} columns"
               + ("" if _is_excel(uploaded_file.name) else f" · {sep_label}-separated"))
    with st.expander(f"Preview — first {min(8, len(preview))} rows"):
        st.dataframe(preview.head(8), use_container_width=True, hide_index=True)

    columns = [str(c) for c in preview.columns]
    guess = detect_columns(columns)

# ─────────────────────────────────────────────────────────────────────────────
#  2 · Column mapping
# ─────────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="panel-title">🧭 2 · Which column is what</div>', unsafe_allow_html=True)
    st.caption("Guessed from the header names — correct anything that is wrong. "
               "Only the mapped columns are read, so unrelated columns cost nothing.")

    c1, c2, c3 = st.columns(3)
    with c1:
        ts_idx = columns.index(guess.timestamp) if guess.timestamp in columns else 0
        timestamp_col = st.selectbox("Timestamp *", columns, index=ts_idx, key=f"ts_{fkey}",
                                     help="Local wall time at the station, an ISO string with offset, or a Unix epoch.")
    with c2:
        ghi_opts = [CHOOSE_LABEL] + columns
        ghi_guess = guess.sensors.get("GHI_RC_01")
        ghi_col = st.selectbox("GHI — global horizontal irradiance, W/m² *", ghi_opts, key=f"ghi_{fkey}",
                               index=ghi_opts.index(ghi_guess) if ghi_guess in ghi_opts else 0,
                               help="The model's main input. Required.")
    with c3:
        par_opts = [NONE_LABEL] + columns
        par_guess = guess.target
        target_col = st.selectbox("Measured PAR, µmol/m²/s (optional)", par_opts, key=f"par_{fkey}",
                                  index=par_opts.index(par_guess) if par_guess in par_opts else 0,
                                  help="Enables MAE / RMSE / R² of the model and the baseline on your data.")

    optional = [c for c in SENSOR_SPECS if c != "GHI_RC_01"]
    n_found = sum(1 for c in optional if guess.sensors.get(c) in columns)
    with st.expander(f"Other weather sensors (optional) — {n_found} of {len(optional)} recognised"):
        st.caption("A sensor that is not in your file is replaced by the training median for every row. "
                   "The prediction still works, with somewhat lower accuracy.")
        sensor_mapping: dict[str, str | None] = {}
        cols = st.columns(2)
        for i, canon in enumerate(optional):
            spec = SENSOR_SPECS[canon]
            opts = [NONE_LABEL] + columns
            g = guess.sensors.get(canon)
            with cols[i % 2]:
                choice = st.selectbox(f"{spec.label} ({spec.unit})", opts,
                                      index=opts.index(g) if g in opts else 0, key=f"map_{canon}_{fkey}")
            sensor_mapping[canon] = None if choice == NONE_LABEL else choice
    sensor_mapping["GHI_RC_01"] = None if ghi_col == CHOOSE_LABEL else ghi_col

    mapped = [timestamp_col] + [c for c in sensor_mapping.values() if c]
    if target_col != NONE_LABEL:
        mapped.append(target_col)
    dupes = sorted({c for c in mapped if mapped.count(c) > 1})
    mapping_ok = ghi_col != CHOOSE_LABEL and not dupes
    if ghi_col == CHOOSE_LABEL:
        st.warning("Choose the GHI column — without it the model has nothing to work with.")
    if dupes:
        st.warning(f"The same column is mapped twice: {', '.join(dupes)}.")

# ─────────────────────────────────────────────────────────────────────────────
#  3 · Where and when
# ─────────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="panel-title">📍 3 · Station location and timezone</div>', unsafe_allow_html=True)

    inferred = None
    city = _infer_city(uploaded_file.name)
    if city:
        try:
            candidates = geocode_city(city)          # cached; returns [] on failure
        except Exception:
            candidates = []
        if candidates:
            inferred = candidates[0]
            st.caption(f"📌 Guessed from the file name: **{inferred['display']}** — edit below if wrong.")
        else:
            st.caption(f"Could not resolve “{city}” from the file name — enter the coordinates below.")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.3])
    with c1:
        lat = st.number_input("Latitude (°N)", -90.0, 90.0,
                              value=float(inferred["latitude"]) if inferred else 51.6872,
                              format="%.5f", key=f"lat_{fkey}")
    with c2:
        lon = st.number_input("Longitude (°E)", -180.0, 180.0,
                              value=float(inferred["longitude"]) if inferred else 14.4143,
                              format="%.5f", key=f"lon_{fkey}")
    with c3:
        alt = st.number_input("Altitude (m)", -430.0, 8848.0,
                              value=float(inferred.get("elevation") or 0.0) if inferred else 0.0,
                              format="%.0f", key=f"alt_{fkey}")
    with c4:
        tz_default = (inferred or {}).get("timezone") or "Europe/Berlin"
        tz_input = st.text_input("Timezone of the timestamps (IANA)", value=tz_default, key=f"tz_{fkey}",
                                 help="Europe/Berlin, UTC, America/New_York … Timestamps that carry "
                                      "their own offset are converted to this zone.")
    tz_ok = True
    try:
        tz_str = validate_timezone(tz_input)
    except ValueError as exc:
        tz_ok = False
        tz_str = tz_input
        st.error(str(exc))

    coords_ok = True
    if abs(lat) < 1e-9 and abs(lon) < 1e-9:
        coords_ok = st.checkbox("0° N, 0° E is in the Atlantic Ocean — yes, the station really is there.",
                                key=f"zero_ok_{fkey}")
        if not coords_ok:
            st.warning("Enter the station coordinates — solar geometry depends on them.")

    st.caption(f"Data is cleaned as in the training pipeline, aggregated to **1-minute means** "
               f"(the model's native resolution); gaps up to **{DEFAULT_MAX_GAP}** are carried forward, "
               f"longer gaps get the training median.")

    ready = mapping_ok and tz_ok and coords_ok
    run = st.button("🚀 Clean, engineer features and predict", type="primary",
                    use_container_width=True, disabled=not ready)

# ─────────────────────────────────────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────────────────────────────────────
if run and ready:
    try:
        usecols = list(dict.fromkeys(mapped))            # unique, order kept
        with st.spinner(f"Reading {len(usecols)} of {len(columns)} columns…"):
            full = _read_full(uploaded_file, read_info, usecols)
        if len(full) > MAX_ROWS:
            st.error(f"{len(full):,} rows exceed the limit of {MAX_ROWS:,}. Split the file "
                     "(for example one month per file) and upload the parts separately.")
            st.stop()
        result = _run_with_progress({
            "df": full,
            "timestamp_column": timestamp_col,
            "target_column": "" if target_col == NONE_LABEL else target_col,
            "sensor_mapping": sensor_mapping,
            "timezone_str": tz_str,
            "latitude": float(lat), "longitude": float(lon), "altitude": float(alt),
            "resample_period": "1min",
        })
        del full
        result["file_name"] = uploaded_file.name
        result["file_size_mb"] = round(size_mb, 2)
        result["generated_at"] = datetime.now().isoformat(timespec="seconds")
        st.session_state.dataset_result = result
        st.session_state.dataset_token = uuid.uuid4().hex[:8]
        st.session_state.dataset_scroll = True
    except ValueError as exc:
        st.error(f"❌ {exc}")
        st.stop()
    except MemoryError:
        st.error("❌ The server ran out of memory while processing this file. "
                 "Upload a smaller file (one month at a time works well).")
        st.stop()
    except Exception as exc:
        st.error(f"❌ Processing failed: {type(exc).__name__}: {exc}")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  Results
# ─────────────────────────────────────────────────────────────────────────────
result = st.session_state.dataset_result
if result is None:
    st.stop()

tok = st.session_state.dataset_token or "r"
if st.session_state.get("dataset_scroll"):
    _st_components.html(
        "<script>setTimeout(function(){var e=window.parent.document.getElementById('results-anchor');"
        "if(e) e.scrollIntoView({behavior:'smooth'});},300);</script>", height=0)
    st.session_state.dataset_scroll = False

st.divider()
st.markdown('<a id="results-anchor"></a>', unsafe_allow_html=True)
st.markdown("## 📈 Results")

res_df: pd.DataFrame = result.get("results", pd.DataFrame())
metrics = result.get("metrics", {})
if res_df is None or res_df.empty:
    st.error("No rows were left to score. " + " ".join(result.get("warnings", [])))
    with st.expander("Cleaning report"):
        st.dataframe(pd.DataFrame(result.get("report", [])), use_container_width=True, hide_index=True)
    st.stop()

# ── Counts ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows uploaded", f"{result['raw_rows']:,}")
c2.metric("Rows after cleaning", f"{result['clean_rows']:,}",
          delta=f"-{result['raw_rows'] - result['clean_rows']:,}", delta_color="off")
c3.metric("1-minute bins scored", f"{result['resampled_rows']:,}")
c4.metric("Native resolution", _fmt_seconds(result.get("resolution_seconds")))
period = result.get("period", {})
if period:
    c5.metric("Period", f"{pd.Timestamp(period['start']):%Y-%m-%d} → {pd.Timestamp(period['end']):%Y-%m-%d}")

coords = result.get("coordinates", {})
st.markdown(
    f'<span class="pill">📍 <b>{coords.get("lat", 0):.4f}°, {coords.get("lon", 0):.4f}°</b> · {coords.get("alt", 0):.0f} m</span>'
    f'<span class="pill">🕒 <b>{result.get("timezone")}</b></span>'
    f'<span class="pill">🗂 <b>{result.get("file_name", "")}</b></span>'
    + "".join(f'<span class="pill">{SENSOR_SPECS[k].label} ← <b>{v}</b></span>'
              for k, v in result.get("mapping", {}).items() if k in SENSOR_SPECS),
    unsafe_allow_html=True,
)

for note in result.get("warnings", []):
    st.warning(f"⚠️ {note}")

# ── Cleaning report ───────────────────────────────────────────────────────────
with st.expander("🧹 Cleaning report — what was removed, and why"):
    rep = pd.DataFrame(result.get("report", []))
    if not rep.empty:
        rep = rep.rename(columns={"step": "Step", "rows_in": "Rows in", "rows_out": "Rows out",
                                  "removed": "Removed", "note": "Detail"})
        st.dataframe(rep, use_container_width=True, hide_index=True)
    fill = result.get("fill_summary", {})
    if fill:
        fill_df = pd.DataFrame([
            {"Sensor": SENSOR_SPECS[k].label, "Missing": v["missing"], "Carried forward": v["carried"],
             "Training median": v["median"],
             "Fallback": v["fallback"] if isinstance(v["fallback"], str)
             else f"{v['fallback']:.2f} {SENSOR_SPECS[k].unit}"}
            for k, v in fill.items()
        ])
        st.caption("Missing sensor values and how they were filled")
        st.dataframe(fill_df, use_container_width=True, hide_index=True)
    prep = result.get("prep_info", {})
    if prep.get("clipped"):
        st.caption("Values clipped to the training 1st–99th percentile before the model saw them "
                   "(the model was trained on clipped values): "
                   + ", ".join(f"{k} ×{n:,}" for k, n in prep["clipped"].items()))

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown('<div class="panel-title">📊 Baseline vs Model — full metrics comparison</div>', unsafe_allow_html=True)
if metrics:
    n_eval = metrics.get("daytime_rows_evaluated", 0)
    st.caption(f"Evaluated on {n_eval:,} daytime bins with measured PAR > 0 — the conditions the model "
               f"was evaluated on. For reference, on its own held-out test days the model reached "
               f"MAE {_card['test_mae']:.1f} vs {_card['baseline_mae']:.1f} µmol/m²/s for the baseline.")

    def _fmt(v) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float)) and np.isfinite(v) else "—"

    # (label, baseline, model, kind) — kind: "lower", "higher", "bias", "gain"
    rows = [
        ("R²",                   metrics.get("baseline_r2"),    metrics.get("model_r2"),    "higher"),
        ("nRMSE (%)",            metrics.get("baseline_nrmse"), metrics.get("model_nrmse"), "lower"),
        ("RMSE (µmol/m²/s)",     metrics.get("baseline_rmse"),  metrics.get("model_rmse"),  "lower"),
        ("MAE (µmol/m²/s)",      metrics.get("baseline_mae"),   metrics.get("model_mae"),   "lower"),
        ("nMBE (%)",             metrics.get("baseline_nmbe"),  metrics.get("model_nmbe"),  "bias"),
        ("MBE (µmol/m²/s)",      metrics.get("baseline_mbe"),   metrics.get("model_mbe"),   "bias"),
        ("MAE improvement (%)",  None, metrics.get("mae_improvement_pct"),  "gain"),
        ("RMSE improvement (%)", None, metrics.get("rmse_improvement_pct"), "gain"),
    ]
    html = """
    <table style="width:100%;border-collapse:collapse;font-size:.92rem;margin-bottom:1rem">
    <thead><tr>
      <th style="text-align:left;padding:.55rem .8rem;color:#8892b0;border-bottom:1px solid #2a2d3e">Metric</th>
      <th style="text-align:right;padding:.55rem .8rem;color:#f39c12;border-bottom:1px solid #2a2d3e">Baseline (McCree)</th>
      <th style="text-align:right;padding:.55rem .8rem;color:#2ecc71;border-bottom:1px solid #2a2d3e">Model (XGBoost)</th>
      <th style="text-align:right;padding:.55rem .8rem;color:#8892b0;border-bottom:1px solid #2a2d3e">Winner</th>
    </tr></thead><tbody>"""
    for label, bv, mv, kind in rows:
        b_ok = isinstance(bv, (int, float)) and np.isfinite(bv)
        m_ok = isinstance(mv, (int, float)) and np.isfinite(mv)
        if kind == "lower" and b_ok and m_ok:
            winner = "✅ Model" if mv < bv else ("⚠️ Baseline" if bv < mv else "—")
        elif kind == "higher" and b_ok and m_ok:
            winner = "✅ Model" if mv > bv else ("⚠️ Baseline" if bv > mv else "—")
        elif kind == "bias" and b_ok and m_ok:
            winner = "✅ Model" if abs(mv) < abs(bv) else ("⚠️ Baseline" if abs(bv) < abs(mv) else "—")
        elif kind == "gain" and m_ok:
            winner = "✅ Yes" if mv > 0 else "⚠️ No"
        else:
            winner = ""
        html += (f"<tr><td style='padding:.45rem .8rem;border-bottom:1px solid #1a1d2e;color:#e8eaf6'>{label}</td>"
                 f"<td style='padding:.45rem .8rem;border-bottom:1px solid #1a1d2e;text-align:right;color:#f39c12'>{_fmt(bv) if bv is not None else ''}</td>"
                 f"<td style='padding:.45rem .8rem;border-bottom:1px solid #1a1d2e;text-align:right;color:#2ecc71'>{_fmt(mv)}</td>"
                 f"<td style='padding:.45rem .8rem;border-bottom:1px solid #1a1d2e;text-align:right;color:#8892b0'>{winner}</td></tr>")
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)
elif result.get("mapping", {}).get("target_par"):
    st.info("ℹ️ Fewer than 10 daytime bins had a measured PAR > 0, so no accuracy metrics were computed.")
else:
    st.info("ℹ️ Accuracy metrics need a **measured PAR column** — map one in step 2 and run again.")

# ── Chart ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="panel-title">📉 Model vs baseline over time</div>', unsafe_allow_html=True)
chart_df = res_df.dropna(subset=["model_prediction", "baseline_prediction"])
if len(chart_df) > 5000:                      # keep the browser responsive
    every = max(1, len(chart_df) // 5000)
    chart_df = chart_df.iloc[::every]
    st.caption(f"Showing every {every}th bin of {len(res_df):,} — the downloads contain all of them.")
if not chart_df.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["baseline_prediction"], mode="lines",
                             name="Baseline (McCree)", line=dict(color="#f39c12", width=1.6)))
    fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["model_prediction"], mode="lines",
                             name="Model (XGBoost)", line=dict(color="#2ecc71", width=1.6)))
    if chart_df["target_par"].notna().any():
        fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["target_par"], mode="lines",
                                 name="Measured PAR", line=dict(color="#ffffff", width=1.2, dash="dot")))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,46,0.7)",
                      font=dict(color="#e8eaf6"), height=380, margin=dict(l=10, r=10, t=10, b=0),
                      legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.01),
                      xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                      yaxis=dict(title="PAR (µmol/m²/s)", gridcolor="rgba(255,255,255,0.06)"),
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ── Domain check ──────────────────────────────────────────────────────────────
dom = result.get("domain", {})
feat_dom = dom.get("features")
loc = dom.get("location") or {}
with st.expander("🧭 Training-domain check — is this data like what the model learned from?"):
    if loc:
        st.markdown(f"Nearest training station: **{loc.get('nearest')}**, "
                    f"**{loc.get('distance_km', 0):,.0f} km** away — "
                    + ("inside the region the model knows." if loc.get("in_domain")
                       else "outside it; treat the results as an extrapolation."))
    if isinstance(feat_dom, pd.DataFrame) and not feat_dom.empty:
        show = feat_dom.rename(columns={"label": "Input", "unit": "Unit", "low": "Train min", "high": "Train max",
                                        "n_out": "Bins outside", "pct_out": "% of daytime bins",
                                        "min": "Your min", "max": "Your max"})
        st.dataframe(show[["Input", "Unit", "Train min", "Train max", "Your min", "Your max",
                           "Bins outside", "% of daytime bins"]],
                     use_container_width=True, hide_index=True)
        st.caption("Trees do not extrapolate: outside its training range the model answers as it would at the edge.")
    else:
        st.success("Every model input stays within the training range in every daytime bin.")

# ── Feature importance ────────────────────────────────────────────────────────
imp = get_feature_importance()
if imp is not None:
    with st.expander("✨ Feature importance (gain, averaged over the 3 seeds)"):
        fi_df = imp.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        st.dataframe(fi_df, use_container_width=True, hide_index=True, height=320)

# ── Exports ───────────────────────────────────────────────────────────────────
st.markdown('<div class="panel-title">📦 Download</div>', unsafe_allow_html=True)
cleaned_df: pd.DataFrame = result.get("cleaned", pd.DataFrame())
features_df: pd.DataFrame = result.get("features", pd.DataFrame())


def _report_dict() -> dict:
    return {
        "generated_at": result.get("generated_at"),
        "file": {"name": result.get("file_name"), "size_mb": result.get("file_size_mb"),
                 "rows_uploaded": result["raw_rows"]},
        "settings": {"timezone": result.get("timezone"), "coordinates": result.get("coordinates"),
                     "resample_period": result.get("resample_period"), "max_gap": DEFAULT_MAX_GAP,
                     "native_resolution_seconds": result.get("resolution_seconds")},
        "column_mapping": result.get("mapping"),
        "rows": {"uploaded": result["raw_rows"], "parsed": result.get("parsed_rows"),
                 "after_cleaning": result["clean_rows"], "bins_scored": result["resampled_rows"],
                 "daytime_bins": result.get("daytime_rows")},
        "cleaning_report": result.get("report"),
        "missing_values": result.get("fill_summary"),
        "clipped_or_imputed_before_model": result.get("prep_info"),
        "warnings": result.get("warnings"),
        "metrics": metrics,
        "domain_check": {"location": loc,
                         "features": feat_dom.to_dict("records") if isinstance(feat_dom, pd.DataFrame) else []},
        "model": result.get("model"),
        "columns": {
            "predictions": "timestamp (local), lat, lon, alt, GHI [W/m²], model_prediction / baseline_prediction "
                           "/ difference [µmol/m²/s], is_day, target_par (measured, if given), weather sensors, "
                           "Temp_RC_merged, zenith, elevation, clearness_kt, n_samples (raw rows in the bin)",
            "cleaned": "native-resolution rows after sentinel removal, validity ranges, night filter, "
                       "duplicate removal and short-gap filling",
            "features": "1-minute bins with all 22 computed features (15 of them are model inputs)",
        },
    }


t_pred, t_clean, t_feat, t_rep, t_zip = st.tabs([
    f"Predictions ({len(res_df):,})", f"Cleaned data ({len(cleaned_df):,})",
    f"Feature matrix ({len(features_df):,})", "Report (JSON)", "Everything (ZIP)",
])
with t_pred:
    st.caption("One row per scored 1-minute bin: model and baseline PAR, the weather that went in, "
               "solar geometry and — if you provided it — the measured PAR.")
    download_bar(res_df, "predictions", key=f"dl_pred_{tok}", label="Predictions")
with t_clean:
    st.caption("Your data after the cleaning steps, at its native resolution, before aggregation. "
               "Filled values are included; the cleaning report says how many.")
    download_bar(cleaned_df, "cleaned", key=f"dl_clean_{tok}", label="Cleaned data")
with t_feat:
    st.caption("Exactly what the model saw — the 22 computed features per 1-minute bin, "
               "so results can be reproduced outside this app.")
    download_bar(features_df, "features", key=f"dl_feat_{tok}", label="Feature matrix")
with t_rep:
    st.caption("Settings, column mapping, every cleaning step with counts, metrics, the domain check "
               "and the model card — everything needed to cite or audit this run.")
    st.download_button("⬇ Report · JSON", data=lambda: to_json_bytes(_report_dict()),
                       file_name=file_name("report", "json"), mime="application/json",
                       key=f"dl_rep_{tok}", use_container_width=True)
with t_zip:
    st.caption("Predictions, cleaned data and feature matrix as CSV plus the JSON report, in one archive.")
    st.download_button(
        "⬇ Everything · ZIP",
        data=lambda: to_zip_bytes({
            "predictions.csv": lambda: to_csv_bytes(res_df),
            "cleaned_data.csv": lambda: to_csv_bytes(cleaned_df),
            "feature_matrix.csv": lambda: to_csv_bytes(features_df),
            "report.json": lambda: to_json_bytes(_report_dict()),
            "README.txt": (
                "ParPredict — Dataset Upload export\n\n"
                f"Generated: {result.get('generated_at')}\nSource file: {result.get('file_name')}\n\n"
                "predictions.csv     one row per 1-minute bin (model vs McCree baseline)\n"
                "cleaned_data.csv    your data after cleaning, native resolution\n"
                "feature_matrix.csv  the 22 computed features per bin (model inputs)\n"
                "report.json         settings, cleaning steps, metrics, domain check, model card\n"
            ).encode("utf-8"),
        }),
        file_name=file_name("everything", "csv")[:-4] + ".zip",
        mime="application/zip", key=f"dl_zip_{tok}", use_container_width=True,
    )

# ── Preview ───────────────────────────────────────────────────────────────────
with st.expander("📋 Preview of the scored data (first 20 bins)"):
    preview_cols = ["timestamp", "GHI", "model_prediction", "baseline_prediction", "difference",
                    "target_par", "Temp_WS", "RH_WS", "clearness_kt", "elevation", "n_samples"]
    st.dataframe(
        res_df[preview_cols].head(20).rename(columns={
            "timestamp": "Timestamp (local)", "GHI": "GHI W/m²", "model_prediction": "Model PAR",
            "baseline_prediction": "Baseline PAR", "difference": "Δ Model−Baseline", "target_par": "Measured PAR",
            "Temp_WS": "Temp °C", "RH_WS": "RH %", "clearness_kt": "kt", "elevation": "Sun elev. °",
            "n_samples": "Rows/bin"}),
        use_container_width=True, hide_index=True,
    )
