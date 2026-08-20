"""
pages/2_Expert_Mode.py  –  PAR Predictor · Expert Mode
────────────────────────────────────────────────────────
Full manual sensor input — enter every reading yourself.
Shows ML vs McCree comparison, feature importance and full feature table.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as _components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date, time as dtime

from core.features import compute_features
from core.predict  import (
    predict_par, mccree_estimate, get_feature_importance, is_model_available
)
from core.weather  import fetch_weather

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Expert Mode · ParPredict",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }

.panel-title {
    font-size: .78rem; font-weight: 700; color: #f39c12;
    text-transform: uppercase; letter-spacing: 1.5px;
    border-left: 3px solid #f39c12; padding-left: .5rem;
    margin: 1rem 0 .7rem 0;
}
.result-card {
    border-radius: 16px; padding: 1.5rem; text-align: center;
    border: 2px solid;
}
.big-num  { font-size: 3.2rem; font-weight: 900; line-height: 1; }
.unit     { font-size: .88rem;  color: #8892b0; margin-top: .25rem; }
.cap-lbl  { font-size: .72rem; color: #8892b0; text-transform: uppercase;
            letter-spacing: 1px; margin-bottom: .4rem; }
.sec-hdr  {
    font-size: .8rem; font-weight: 700; color: #2ecc71;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin: 1.2rem 0 .4rem 0; border-left: 3px solid #2ecc71;
    padding-left: .6rem;
}
.welcome-card {
    background:#1a1d2e; border:1px dashed #2a2d3e; border-radius:16px;
    padding:4rem 2rem; text-align:center; color:#8892b0;
}

@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .panel-title { font-size: .72rem; }
    .result-card { padding: 1.2rem; margin-bottom: 1rem; }
    .big-num { font-size: 2.6rem; }
    .unit, .cap-lbl, .sec-hdr { font-size: .78rem; }
    .step-card { min-height: auto; padding: 1.2rem; margin-bottom: 1rem; }
    section[data-testid="stHorizontalBlock"] { gap: 1rem !important; }
    div[data-testid="column"] > div:first-child { min-width: 100% !important; }
}
@media (max-width: 640px) {
    .block-container { padding-top: 1rem; }
    .panel-title { margin-bottom: .5rem; }
    .result-card { padding: 1rem; margin-bottom: 1rem; }
    .big-num { font-size: 2.2rem; }
    .unit, .cap-lbl, .sec-hdr { font-size: .75rem; }
    .welcome-card { padding: 2.5rem 1.2rem; margin-bottom: 1rem; }
    div[data-testid="column"] > div:first-child { min-width: 100% !important; }
    section[data-testid="stHorizontalBlock"] { gap: 1rem !important; }
}
</style>
""", unsafe_allow_html=True)

if "expert_result" not in st.session_state:
    st.session_state.expert_result = None

# ensure expert inputs persist between reruns
for key, value in {
    "e_lat":  51.6872,
    "e_lon":  14.4143,
    "e_alt":  84.0,
    "e_tz":   "Europe/Berlin",
    "e_date": date.today(),
    "e_time": dtime(datetime.now().hour, 0),
    "e_ghi":  450.0,
    "e_temp": 18.0,
    "e_rh":   65.0,
    "e_dwp":  8.0,
    "e_ws":   3.0,
    "e_wd":   180.0,
    "e_prec": 0.0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "expert_autofetch_temp" in st.session_state:
    fetched = st.session_state.pop("expert_autofetch_temp")
    st.session_state.e_ghi  = fetched.get("GHI_RC_01", st.session_state.e_ghi)
    st.session_state.e_temp = fetched.get("Temp_WS",    st.session_state.e_temp)
    st.session_state.e_rh   = fetched.get("RH_WS",      st.session_state.e_rh)
    st.session_state.e_dwp  = fetched.get("DWP_WS",     st.session_state.e_dwp)
    st.session_state.e_ws   = fetched.get("WS_WS",      st.session_state.e_ws)
    st.session_state.e_wd   = fetched.get("WD_WS",      st.session_state.e_wd)
    st.session_state.e_prec = fetched.get("PREC_INT_WS", st.session_state.e_prec)


def _apply_expert_autofetch():
    try:
        dt_sel = datetime.combine(
            st.session_state.e_date,
            st.session_state.e_time,
        )
        fetched = fetch_weather(
            st.session_state.e_lat,
            st.session_state.e_lon,
            dt_sel,
        )
        st.session_state.expert_autofetch_temp = fetched
        st.session_state.expert_autofetch_status = (
            "success", "✅ Weather auto-fetched successfully."
        )
    except Exception as e:
        st.session_state.expert_autofetch_status = (
            "error", f"❌ Open-Meteo fetch failed: {e}"
        )

# ── Helpers ───────────────────────────────────────────────────────────────────
def _gauge(value, max_val, color, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 12, "color": "#8892b0"}},
        number={"font": {"size": 32, "color": color}},
        gauge=dict(
            axis=dict(range=[0, max_val],
                      tickfont=dict(color="#8892b0", size=9)),
            bar=dict(color=color),
            bgcolor="rgba(26,29,46,0.8)",
            bordercolor="#2a2d3e",
            steps=[
                dict(range=[0, max_val * .2], color="rgba(52,73,94,.3)"),
                dict(range=[max_val * .2, max_val * .5],
                     color="rgba(46,204,113,.08)"),
                dict(range=[max_val * .5, max_val],
                     color="rgba(243,156,18,.08)"),
            ],
            threshold=dict(line=dict(color="white", width=2),
                           thickness=0.75, value=value),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaf6"),
        height=210,
        margin=dict(l=10, r=10, t=30, b=0),
    )
    return fig

def _importance_chart(importance):
    top = importance.head(12).sort_values()
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker=dict(color=top.values,
                    colorscale=[[0, "#2ecc71"], [1, "#f39c12"]]),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,46,0.6)",
        font=dict(color="#e8eaf6", size=11),
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                   title="Importance"),
        yaxis=dict(showgrid=False),
    )
    return fig

# ════════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("# ⚙️ Expert Mode — Full Sensor Input Dashboard")
st.caption(
    "Enter your own sensor readings for maximum accuracy · "
    "ML prediction vs McCree baseline · Feature importance"
)
st.divider()

if not is_model_available():
    st.error("⚠️ Model not found. Run `git lfs pull` from the project root.")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT: inputs (left) | results (right)
# ════════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 2.3], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
#  LEFT — INPUT PANEL
# ─────────────────────────────────────────────────────────────────────────────
with left:
    with st.container(border=True):

        # ── Location & Time ───────────────────────────────────────────────────
        st.markdown('<div class="panel-title">📍 Location & Time</div>',
                    unsafe_allow_html=True)
        lat = st.number_input("Latitude (°N)",  -90.0,  90.0,  51.6872,
                              format="%.4f", step=0.0001, key="e_lat")
        lon = st.number_input("Longitude (°E)", -180.0, 180.0, 14.4143,
                              format="%.4f", step=0.0001, key="e_lon")
        alt = st.number_input("Altitude (m)",    0.0, 8848.0, 84.0,
                              step=1.0, key="e_alt")
        tz  = st.text_input("Timezone (IANA)", "Europe/Berlin", key="e_tz")

        now = datetime.now()
        sel_date = st.date_input("Date",  date.today(), key="e_date")
        sel_time = st.time_input(
            "Local time",
            dtime(now.hour, 0),
            step=60,
            key="e_time",
            help="Cliquez dans le champ ou tapez l'heure au format HH:MM.",
        )
        dt_sel = datetime.combine(sel_date, sel_time)

        # ── Solar Irradiance ──────────────────────────────────────────────────
        st.markdown('<div class="panel-title">☀️ Solar Irradiance</div>',
                    unsafe_allow_html=True)
        ghi = st.slider("GHI — Global Horizontal Irradiance (W/m²)",
                        0.0, 1400.0, 450.0, 1.0, key="e_ghi")

        # ── Meteorological Sensors ─────────────────────────────────────────────
        st.markdown('<div class="panel-title">🌡️ Meteorological Sensors</div>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            temp = st.number_input("Temperature (°C)", -20.0, 50.0, 18.0,
                                   0.1, key="e_temp")
            rh   = st.number_input("Humidity (%)",      0.0, 100.0, 65.0,
                                   0.5, key="e_rh")
            dwp  = st.number_input("Dewpoint (°C)",    -20.0, 40.0,  8.0,
                                   0.1, key="e_dwp")
        with c2:
            ws   = st.number_input("Wind speed (m/s)",  0.0,  40.0, 3.0,
                                   0.1, key="e_ws")
            wd   = st.number_input("Wind dir (°)",       0.0, 360.0, 180.0,
                                   1.0, key="e_wd")
            prec = st.number_input("Precipitation (mm/h)", 0.0, 30.0, 0.0,
                                   0.1, key="e_prec")

        st.markdown("<br>", unsafe_allow_html=True)
        fetch_weather_btn = st.button(
            "🌦️  Auto-fetch weather from Open-Meteo",
            use_container_width=True,
            type="secondary",
            on_click=_apply_expert_autofetch,
        )
        if "expert_autofetch_status" in st.session_state:
            status_type, status_msg = st.session_state.expert_autofetch_status
            if status_type == "success":
                st.success(status_msg)
            else:
                st.error(status_msg)
        predict_btn = st.button("⚙️  Predict PAR",
                                use_container_width=True, type="primary")

    with st.expander("📋 Reference coordinates"):
        import pandas as _pd
        _coords = _pd.DataFrame({
            "Location": ["Laubsdorf DE","Nebelin DE","Paris FR","Tokyo JP"],
            "Lat":  [51.6872, 53.1183, 48.8566, 35.6762],
            "Lon":  [14.4143, 11.7461,  2.3522, 139.6503],
            "Alt m": [84, 50, 35, 40],
        })
        st.dataframe(_coords, hide_index=True, use_container_width=True)
        st.caption("Right-click Google Maps → copy lat, lon.")

# ─────────────────────────────────────────────────────────────────────────────
#  HANDLE PREDICT
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn:
    weather = {
        "GHI_RC_01":   ghi,
        "Temp_WS":     temp,
        "RH_WS":       rh,
        "DWP_WS":      dwp,
        "WS_WS":       ws,
        "WD_WS":       wd,
        "PREC_INT_WS": prec,
    }
    try:
        feat, is_day = compute_features(lat, lon, alt, dt_sel, weather, tz)
        par          = predict_par(feat) if is_day else 0.0
        mc           = mccree_estimate(ghi)
        imp          = get_feature_importance()
    except Exception as e:
        with right:
            st.error(f"❌ Prediction error: {e}")
        st.stop()

    st.session_state.expert_result = {
        "par": par, "mc": mc, "features": feat,
        "is_day": is_day, "imp": imp,
        "inputs": weather,
        "lat": lat, "lon": lon, "alt": alt, "dt": dt_sel, "tz": tz,
    }

    # Auto-scroll to results
    _components.html(
        '<script>'
        'setTimeout(function(){'
        '  var e=window.parent.document.getElementById("expert-results");'
        '  if(e) e.scrollIntoView({behavior:"smooth",block:"start"});'
        '},400);'
        '</script>',
        height=0,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  RIGHT — RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with right:
    res = st.session_state.expert_result

    if res is None:
        st.markdown("""
        <div class="welcome-card">
            <div style="font-size:3rem;margin-bottom:1rem">⚙️</div>
            <div style="font-size:1.15rem;font-weight:700;color:#fff;
                        margin-bottom:.8rem">Ready for expert prediction</div>
            <div style="font-size:.9rem;line-height:1.75">
                Enter your <strong style="color:#f39c12">sensor readings</strong>
                on the left, then click
                <strong style="color:#f39c12">Predict PAR</strong>.<br><br>
                Results include ML prediction, McCree comparison<br>
                and full feature importance analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Scroll anchor
        st.markdown('<div id="expert-results"></div>', unsafe_allow_html=True)

        par    = res["par"]
        mc     = res["mc"]
        ft     = res["features"]
        is_day = res["is_day"]
        imp    = res["imp"]

        # ── Location bar ──────────────────────────────────────────────────────
        st.markdown(
            f"**📌 {res['lat']:.4f}°N, {res['lon']:.4f}°E** &nbsp; "
            f"alt {res['alt']:.0f} m &nbsp;·&nbsp; `{res['tz']}` &nbsp;·&nbsp; "
            f"**{res['dt'].strftime('%Y-%m-%d %H:%M')}**"
        )

        if not is_day:
            st.info("🌙 **Night-time** — sun below horizon. PAR = 0.", icon="🌑")

        # ── Three result cards ────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            col_ml = "#2ecc71" if is_day else "#3498db"
            st.markdown(f"""
            <div class="result-card" style="background:linear-gradient(135deg,#0d2b1a,#0f1117);
                 border-color:{col_ml}">
                <div class="cap-lbl">🤖 ML Model (XGBoost)</div>
                <div class="big-num" style="color:{col_ml}">{par:.1f}</div>
                <div class="unit">µmol / m² / s</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="result-card" style="background:linear-gradient(135deg,#2d1a00,#0f1117);
                 border-color:#f39c12">
                <div class="cap-lbl">📐 McCree Baseline</div>
                <div class="big-num" style="color:#f39c12">{mc:.1f}</div>
                <div class="unit">µmol / m² / s</div>
                <div style="color:#8892b0;font-size:.75rem;margin-top:.4rem">
                    GHI × 0.45 × 4.57
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            diff = abs(par - mc)
            pct  = (diff / mc * 100) if mc > 1 else 0.0
            sign = "ML > McCree" if par > mc else "ML < McCree"
            diff_color = "#2ecc71" if par > mc else "#e74c3c"
            st.markdown(f"""
            <div class="result-card" style="background:#1a1d2e; border-color:#2a2d3e">
                <div class="cap-lbl">📊 Difference</div>
                <div class="big-num" style="color:{diff_color}">{diff:.1f}</div>
                <div class="unit">µmol / m² / s</div>
                <div style="color:#8892b0;font-size:.78rem;margin-top:.4rem">
                    {sign}<br>({pct:.1f} % relative)
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauges ────────────────────────────────────────────────────────────
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(_gauge(par, 1200, "#2ecc71",
                                   "ML Model (µmol/m²/s)"), width='stretch')
        with g2:
            st.plotly_chart(_gauge(mc, 1200, "#f39c12",
                                   "McCree Estimate (µmol/m²/s)"), width='stretch')

        # ── Feature importance ────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Feature Importance</div>',
                    unsafe_allow_html=True)
        if imp is not None:
            st.plotly_chart(_importance_chart(imp), width='stretch')
        else:
            st.info("Feature importance not available for this model type.")

        # ── Solar geometry ────────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Computed Solar Geometry</div>',
                    unsafe_allow_html=True)
        sg1, sg2, sg3, sg4 = st.columns(4)
        sg1.metric("Zenith",      f"{float(ft['zenith'].iloc[0]):.2f}°")
        sg2.metric("Elevation",   f"{float(ft['elevation'].iloc[0]):.2f}°")
        sg3.metric("Airmass",     f"{float(ft['airmass'].iloc[0]):.3f}")
        sg4.metric("Clearness kt",f"{float(ft['clearness_kt'].iloc[0]):.3f}")

        # ── Full feature table ────────────────────────────────────────────────
        with st.expander("🔍 Full feature vector (all 22 computed values)"):
            categories = {
                "Raw sensor":   ["GHI_RC_01","Temp_WS","RH_WS","DWP_WS","WS_WS",
                                 "WD_WS","PREC_INT_WS","PREC_DIFF_WS","PREC_WS",
                                 "Temp_RC_merged","Temp_RC_01"],
                "pvlib solar":  ["zenith","elevation","airmass","clearness_kt","dni"],
                "Wind cyclical":["wind_sin","wind_cos"],
                "Engineered":   ["is_raining","GHI_rolling_5min",
                                 "temp_diff","dew_depression"],
            }
            cat_map = {c: cat for cat, cols in categories.items() for c in cols}
            disp = ft.T.rename(columns={0: "Value"})
            disp["Value"]    = disp["Value"].round(5)
            disp["Category"] = disp.index.map(lambda x: cat_map.get(x, "Other"))
            st.dataframe(disp[["Category", "Value"]], use_container_width=True,
                         height=420)
