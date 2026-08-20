"""
pages/3_Dataset_Upload.py  –  PAR Predictor · Dataset Upload Mode
────────────────────────────────────────────────────────
Upload a full dataset with sensor measurements and geolocation data,
apply cleaning + feature engineering + resampling, and compare the model
against a McCree baseline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inspect
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as _st_components
import plotly.graph_objects as go


def infer_city_from_filename(filename: str) -> str | None:
    base = Path(filename).stem
    if not base:
        return None
    city = base.split("_")[0].strip()
    return city if city else None


from core.dataset import prepare_dataset_for_prediction
from core.predict import get_feature_importance, is_model_available
from core.weather import geocode_city


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _build_processing_args(
    df: pd.DataFrame,
    timestamp_col: str,
    timezone_str: str,
    filename: str,
    override_lat: float,
    override_lon: float,
    override_alt: float,
    resample_period: str,
    target_col: str | None = None,
) -> dict:
    """
    Build call arguments compatible with multiple prepare_dataset_for_prediction versions.

    Some versions support override_lat/lon/alt directly, while older ones only
    support latitude/longitude/altitude column names.
    """
    sig_params = inspect.signature(prepare_dataset_for_prediction).parameters
    call_args = {
        "df": df,
        "timestamp_column": timestamp_col,
        "resample_period": resample_period,
    }

    if "timezone_str" in sig_params:
        call_args["timezone_str"] = timezone_str
    if "filename" in sig_params:
        call_args["filename"] = filename
    if "target_column" in sig_params and target_col:
        call_args["target_column"] = target_col

    has_override = all(name in sig_params for name in ["override_lat", "override_lon", "override_alt"])
    if has_override:
        call_args["override_lat"] = float(override_lat)
        call_args["override_lon"] = float(override_lon)
        call_args["override_alt"] = float(override_alt)
        if "latitude_column" in sig_params:
            call_args["latitude_column"] = None
        if "longitude_column" in sig_params:
            call_args["longitude_column"] = None
        if "altitude_column" in sig_params:
            call_args["altitude_column"] = None
        return call_args

    # Fallback for older signatures: inject deterministic coordinate columns.
    fallback_df = df.copy()
    fallback_df["_upload_lat"] = float(override_lat)
    fallback_df["_upload_lon"] = float(override_lon)
    fallback_df["_upload_alt"] = float(override_alt)
    call_args["df"] = fallback_df

    if "latitude_column" in sig_params:
        call_args["latitude_column"] = "_upload_lat"
    if "longitude_column" in sig_params:
        call_args["longitude_column"] = "_upload_lon"
    if "altitude_column" in sig_params:
        call_args["altitude_column"] = "_upload_alt"

    return call_args


def _run_processing_with_progress(call_args: dict) -> dict:
    """
    Run dataset processing with real progress events from the core pipeline.
    """
    progress_text = st.empty()
    progress_bar = st.progress(0)

    def _on_progress(pct: int, message: str) -> None:
        safe_pct = max(0, min(100, int(pct)))
        progress_bar.progress(safe_pct)
        progress_text.caption(f"{message} - {safe_pct}%")

    _on_progress(1, "Starting")

    run_args = dict(call_args)
    if "progress_callback" in inspect.signature(prepare_dataset_for_prediction).parameters:
        run_args["progress_callback"] = _on_progress

    result = prepare_dataset_for_prediction(**run_args)
    _on_progress(100, "Completed")
    return result


st.set_page_config(
    page_title="Dataset Upload · PAR Predictor",
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
    margin-bottom: .7rem;
}
.result-card {
    background: linear-gradient(135deg,#0d2b1a 0%,#0f1117 100%);
    border: 2px solid #2ecc71; border-radius: 18px; padding: 1.4rem; text-align: center;
}
.metric-card {
    background: #1a1d2e; border: 1px solid #2a2d3e; border-radius: 14px; padding: 1rem;
}
.welcome-card {
    background:#1a1d2e; border:1px dashed #2a2d3e; border-radius:16px;
    padding:4rem 2rem; text-align:center; color:#8892b0;
}
</style>
""", unsafe_allow_html=True)

if "dataset_result" not in st.session_state:
    st.session_state.dataset_result = None

st.markdown("# 📊 Dataset Upload Mode")
st.caption("Upload a full sensor dataset, clean it, resample to one-minute values, and benchmark the PAR model against a baseline.")
st.divider()

if not is_model_available():
    st.error("⚠️ Model not found. Run `git lfs pull` from the project root before running this page.")
    st.stop()

with st.container(border=True):
    st.markdown('<div class="panel-title">📁 Upload Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="The dataset should include timestamp, latitude, longitude and weather/sensor columns.",
        accept_multiple_files=False,
    )
    st.caption("Max file size: 10 GB")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as exc:
            st.error(f"Could not load the uploaded file: {exc}")
            st.stop()

        st.success(f"Loaded {len(df)} rows from {uploaded_file.name}.")

        st.markdown('<div class="panel-title">🧭 Column Mapping</div>', unsafe_allow_html=True)
        timestamp_candidates = ["timestamp", "datetime", "date_time", "time", "date", "created_at", "ts"]
        default_timestamp = _pick_column(df, timestamp_candidates) or df.columns[0]
        column_options = list(df.columns)
        default_timestamp_idx = column_options.index(default_timestamp) if default_timestamp in column_options else 0
        timestamp_col = st.selectbox("Timestamp column", options=column_options, index=default_timestamp_idx)

        st.caption("Latitude, longitude, and altitude are entered below. The app will infer values from the filename if available.")

        # Optional: let the user identify the observed PAR column for metric computation
        # Accept any column whose name contains "par" (case-insensitive)
        par_candidates = ["PAR_PAR", "PAR", "par", "PAR_RC_01", "par_rc_01", "PAR_umol", "par_umol"]
        par_fuzzy = [c for c in df.columns if "par" in c.lower() and c not in par_candidates]
        par_candidates = par_candidates + par_fuzzy
        default_par = _pick_column(df, par_candidates)
        par_col_options = ["— none —"] + list(df.columns)
        default_par_idx = par_col_options.index(default_par) if default_par in par_col_options else 0
        par_col_label = st.selectbox(
            "Observed PAR column (optional — needed for error metrics)",
            options=par_col_options,
            index=default_par_idx,
            help="Select the column containing measured PAR [µmol/m²/s]. Required to compute MAE, R², etc.",
        )
        target_col = par_col_label if par_col_label != "— none —" else None

        st.markdown('<div class="panel-title">⚙️ Processing Settings</div>', unsafe_allow_html=True)
        filename = uploaded_file.name
        inferred_city = infer_city_from_filename(filename)
        inferred_location = None
        inferred_timezone = "UTC"
        inferred_lat = None
        inferred_lon = None
        inferred_alt = None
        if inferred_city:
            st.caption(f"Inferred city from filename: {inferred_city}")
            candidates = geocode_city(inferred_city)
            if candidates:
                inferred_location = candidates[0]
                inferred_lat = inferred_location.get("latitude")
                inferred_lon = inferred_location.get("longitude")
                inferred_timezone = inferred_location.get("timezone", "UTC") if isinstance(inferred_location, dict) else "UTC"
                inferred_alt = inferred_location.get("elevation")
                st.caption(f"Using inferred location: {inferred_location['display']}")

        timezone_str = st.text_input("Timezone", value=inferred_timezone)

        st.markdown('<div class="panel-title">📍 Coordinates (editable)</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            override_lat = st.number_input(
                "Latitude",
                value=float(inferred_lat) if inferred_lat is not None else 0.0,
                format="%.6f",
                help="Adjust the latitude used for prediction.",
            )
        with c2:
            override_lon = st.number_input(
                "Longitude",
                value=float(inferred_lon) if inferred_lon is not None else 0.0,
                format="%.6f",
                help="Adjust the longitude used for prediction.",
            )
        with c3:
            override_alt = st.number_input(
                "Altitude",
                value=float(inferred_alt) if inferred_alt is not None else 0.0,
                format="%.1f",
                help="Adjust the altitude used for prediction.",
            )

        resample_period = st.selectbox("Resample to", options=["1min", "5min", "10min", "30min", "1H"], index=0)

        if st.button("🚀 Run cleaning, feature engineering and prediction", type="primary", use_container_width=True):
            with st.spinner("Processing the uploaded dataset... This may take a few moments."):
                try:
                    call_args = _build_processing_args(
                        df=df,
                        timestamp_col=timestamp_col,
                        timezone_str=timezone_str,
                        filename=filename,
                        override_lat=override_lat,
                        override_lon=override_lon,
                        override_alt=override_alt,
                        resample_period=resample_period,
                        target_col=target_col,
                    )
                    result = _run_processing_with_progress(call_args)
                    st.session_state.dataset_result = result
                    st.session_state.dataset_scroll = True
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")
                    st.stop()

        if st.session_state.dataset_result is not None and st.button("🔄 Re-run processing", use_container_width=True):
            with st.spinner("Re-processing dataset with current settings..."):
                try:
                    call_args = _build_processing_args(
                        df=df,
                        timestamp_col=timestamp_col,
                        timezone_str=timezone_str,
                        filename=filename,
                        override_lat=override_lat,
                        override_lon=override_lon,
                        override_alt=override_alt,
                        resample_period=resample_period,
                        target_col=target_col,
                    )
                    result = _run_processing_with_progress(call_args)
                    st.session_state.dataset_result = result
                    st.session_state.dataset_scroll = True
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")
                    st.stop()

        if st.session_state.dataset_result is not None:
            # Auto-scroll to results anchor on first render after processing
            if st.session_state.get("dataset_scroll"):
                _st_components.html(
                    "<script>window.parent.document.getElementById('results-anchor')"
                    ".scrollIntoView({behavior:'smooth'});</script>",
                    height=0,
                )
                st.session_state.dataset_scroll = False

            st.divider()
            st.markdown('<a id="results-anchor"></a>', unsafe_allow_html=True)
            st.markdown("## 📈 Results")
            result = st.session_state.dataset_result
            metrics = result.get("metrics", {})
            res_df = result.get("results")

            if res_df is not None and not res_df.empty:
                # ── Row counts ────────────────────────────────────────────────
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Rows uploaded", f"{result['raw_rows']:,}")
                with c2:
                    st.metric("Rows after cleaning", f"{result['clean_rows']:,}")
                with c3:
                    st.metric("Rows after resampling", f"{result['resampled_rows']:,}")
                with c4:
                    st.metric("Timezone", timezone_str)

                # ── Comparison table: Baseline (left) vs Model (right) ────────
                st.markdown('<div class="panel-title">📊 Baseline vs Model — Full Metrics Comparison</div>', unsafe_allow_html=True)
                has_metrics = bool(metrics)
                if has_metrics:
                    n_eval = metrics.get("daytime_rows_evaluated", "?")
                    st.caption(f"Evaluated on {n_eval:,} daytime rows with PAR > 0 — matches training conditions." if isinstance(n_eval, int) else "")

                    def _fmt(v: float, suffix: str = "") -> str:
                        return f"{v:.4f}{suffix}" if np.isfinite(v) else "—"

                    # Rows: (label, baseline_val, model_val, lower_is_better)
                    metric_rows = [
                        ("R²",                  metrics.get("baseline_r2",    float("nan")), metrics.get("model_r2",    float("nan")), False),
                        ("nRMSE (%)",            metrics.get("baseline_nrmse", float("nan")), metrics.get("model_nrmse", float("nan")), True),
                        ("RMSE (µmol/m²/s)",    metrics.get("baseline_rmse",  float("nan")), metrics.get("model_rmse",  float("nan")), True),
                        ("MAE  (µmol/m²/s)",    metrics.get("baseline_mae",   float("nan")), metrics.get("model_mae",   float("nan")), True),
                        ("nMBE (%)",             metrics.get("baseline_nmbe",  float("nan")), metrics.get("model_nmbe",  float("nan")), None),
                        ("MBE  (µmol/m²/s)",    metrics.get("baseline_mbe",   float("nan")), metrics.get("model_mbe",   float("nan")), None),
                        ("MAE improvement (%)",  float("nan"),                                metrics.get("mae_improvement_pct",  float("nan")), False),
                        ("RMSE improvement (%)", float("nan"),                                metrics.get("rmse_improvement_pct", float("nan")), False),
                    ]
                    table_html = """
                    <table style="width:100%;border-collapse:collapse;font-size:.92rem;margin-bottom:1rem">
                    <thead>
                        <tr>
                            <th style="text-align:left;padding:.55rem .8rem;color:#8892b0;border-bottom:1px solid #2a2d3e">Metric</th>
                            <th style="text-align:right;padding:.55rem .8rem;color:#f39c12;border-bottom:1px solid #2a2d3e">Baseline (McCree)</th>
                            <th style="text-align:right;padding:.55rem .8rem;color:#2ecc71;border-bottom:1px solid #2a2d3e">Model (XGBoost)</th>
                            <th style="text-align:right;padding:.55rem .8rem;color:#8892b0;border-bottom:1px solid #2a2d3e">Winner</th>
                        </tr>
                    </thead><tbody>"""
                    for label, bv, mv, lower_better in metric_rows:
                        if lower_better is True and np.isfinite(bv) and np.isfinite(mv):
                            winner = "✅ Model" if mv < bv else ("⚠️ Baseline" if bv < mv else "—")
                        elif lower_better is False and np.isfinite(bv) and np.isfinite(mv):
                            # for improvement cols: positive = model wins
                            if "improvement" in label:
                                winner = "✅ Yes" if mv > 0 else "⚠️ No"
                            else:
                                winner = "✅ Model" if mv > bv else ("⚠️ Baseline" if bv > mv else "—")
                        elif lower_better is None and np.isfinite(mv):
                            winner = "✅ ~0" if abs(mv) < 5 else "⚠️ Bias"
                        else:
                            winner = ""
                        table_html += (
                            f"<tr>"
                            f"<td style='padding:.45rem .8rem;border-bottom:1px solid #1a1d2e;color:#e8eaf6'>{label}</td>"
                            f"<td style='padding:.45rem .8rem;border-bottom:1px solid #1a1d2e;text-align:right;color:#f39c12'>{_fmt(bv)}</td>"
                            f"<td style='padding:.45rem .8rem;border-bottom:1px solid #1a1d2e;text-align:right;color:#2ecc71'>{_fmt(mv)}</td>"
                            f"<td style='padding:.45rem .8rem;border-bottom:1px solid #1a1d2e;text-align:right;color:#8892b0'>{winner}</td>"
                            f"</tr>"
                        )
                    table_html += "</tbody></table>"
                    st.markdown(table_html, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ Accuracy metrics require an **Observed PAR column** — select one above and re-run to compute MAE, RMSE, MAPE and R².")

                # ── Performance Comparison chart (model + baseline always; observed if available) ──
                st.markdown('<div class="panel-title">📉 Performance Comparison</div>', unsafe_allow_html=True)
                chart_df = res_df.dropna(subset=["model_prediction", "baseline_prediction"]).copy()
                # Only plot a sample for large datasets to keep the chart responsive
                if len(chart_df) > 5000:
                    chart_df = chart_df.iloc[::max(1, len(chart_df) // 5000)].copy()
                if not chart_df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=chart_df["timestamp"], y=chart_df["baseline_prediction"],
                        mode="lines", name="Baseline (McCree)",
                        line=dict(color="#f39c12", width=2),
                    ))
                    fig.add_trace(go.Scatter(
                        x=chart_df["timestamp"], y=chart_df["model_prediction"],
                        mode="lines", name="Model (XGBoost)",
                        line=dict(color="#2ecc71", width=2),
                    ))
                    has_obs = chart_df["target_par"].notna().any()
                    if has_obs:
                        fig.add_trace(go.Scatter(
                            x=chart_df["timestamp"], y=chart_df["target_par"],
                            mode="lines", name="Observed PAR",
                            line=dict(color="#ffffff", width=1.5, dash="dot"),
                        ))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(26,29,46,0.7)",
                        font=dict(color="#e8eaf6"),
                        height=380,
                        margin=dict(l=10, r=10, t=10, b=0),
                        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.01),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                        yaxis=dict(title="PAR (µmol/m²/s)", gridcolor="rgba(255,255,255,0.06)"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ── Feature importance ────────────────────────────────────────
                feature_importance = get_feature_importance()
                if feature_importance is not None:
                    st.markdown('<div class="panel-title">✨ Feature Importance</div>', unsafe_allow_html=True)
                    fi_df = feature_importance.reset_index()
                    fi_df.columns = ["Feature", "Importance"]
                    st.dataframe(fi_df.head(20), use_container_width=True, height=320)

                # ── Preview ───────────────────────────────────────────────────
                st.markdown('<div class="panel-title">📋 Preview of Processed Data</div>', unsafe_allow_html=True)
                preview = res_df.head(20).copy()
                preview = preview.rename(columns={
                    "timestamp": "Timestamp",
                    "model_prediction": "Model PAR",
                    "baseline_prediction": "Baseline PAR",
                    "difference": "Δ Model−Baseline",
                    "target_par": "Observed PAR",
                })
                st.dataframe(preview, use_container_width=True, height=320)
            else:
                st.info("No processed rows were generated. Please verify your columns and timestamp values.")
    else:
        st.markdown("""
        <div class="welcome-card">
            <div style="font-size:3rem;margin-bottom:1rem">📊</div>
            <div style="font-size:1.15rem;font-weight:700;color:#fff;margin-bottom:.8rem">Ready to analyze your own dataset</div>
            <div style="font-size:.9rem;line-height:1.75">
                Upload a CSV or Excel file containing timestamps, coordinates and weather/sensor values.<br>
                The page will clean the data, resample it to one-minute values, compute features, run the model and compare it to the McCree baseline.
            </div>
        </div>
        """, unsafe_allow_html=True)
