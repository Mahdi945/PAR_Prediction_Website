"""
pages/1_Normal_Mode.py  –  PAR Predictor · Normal Mode
────────────────────────────────────────────────────────
All inputs in main page (no sidebar). Coordinates only.
Results persist via session_state.
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

from core.weather  import fetch_weather, fetch_forecast
from core.features import compute_features
from core.predict  import predict_par, is_model_available

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Normal Mode · PAR Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }

.panel-title {
    font-size: .78rem; font-weight: 700; color: #2ecc71;
    text-transform: uppercase; letter-spacing: 1.5px;
    border-left: 3px solid #2ecc71; padding-left: .5rem;
    margin-bottom: .7rem;
}
.par-card {
    background: linear-gradient(135deg,#0d2b1a 0%,#0f1117 100%);
    border: 2px solid; border-radius: 18px;
    padding: 1.6rem; text-align: center;
}
.par-big  { font-size: 4rem; font-weight: 900; line-height: 1; }
.par-unit { font-size: .88rem; color: #8892b0; margin-top: .25rem; }
.par-cat  { font-size: 1rem; font-weight: 700; margin-top: .5rem; }
.wpill {
    background:#1a1d2e; border:1px solid #2a2d3e; border-radius:10px;
    padding:.55rem .7rem; text-align:center;
}
.wpill-val { font-size:1.3rem; font-weight:800; color:#fff; }
.wpill-lbl { font-size:.65rem; color:#8892b0; text-transform:uppercase;
             letter-spacing:1px; }
.dli-card {
    background:#1a1d2e; border:1px solid #2a2d3e; border-radius:14px;
    padding:1.1rem 1.3rem;
}
.dli-val  { font-size:1.9rem; font-weight:900; color:#f39c12; }
.dli-lbl  { font-size:.7rem; color:#8892b0; text-transform:uppercase;
            letter-spacing:1px; }
.welcome-card {
    background:#1a1d2e; border:1px dashed #2a2d3e; border-radius:16px;
    padding:4rem 2rem; text-align:center; color:#8892b0; margin-top: 1rem;
}

@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .panel-title { font-size: .72rem; }
    .par-card { padding: 1.2rem; margin-bottom: 1rem; }
    .par-big { font-size: 3rem; }
    .wpill, .dli-card, .welcome-card { padding: 1rem; margin-bottom: 1rem; }
    .wpill-val { font-size: 1.2rem; }
    .dli-val { font-size: 1.6rem; }
    .dli-lbl, .dli-card table td { font-size: .78rem; }
    section[data-testid="stHorizontalBlock"] { gap: 1rem !important; }
    div[data-testid="column"] > div:first-child { min-width: 100% !important; }
}
@media (max-width: 640px) {
    .block-container { padding-top: 1rem; }
    .par-card { padding: 1rem; }
    .par-big { font-size: 2.6rem; }
    .wpill { padding:.45rem .55rem; }
    .wpill-val { font-size: 1.1rem; }
    .dli-card { padding: .9rem; }
    .dli-val { font-size: 1.4rem; }
    .result-card { padding: 1rem; margin-bottom: 1rem; }
    .panel-title { margin-bottom: .5rem; }
    .wpill, .dli-card, .welcome-card { margin-bottom: 1rem; }
    div[data-testid="column"] > div:first-child { min-width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

if "normal_result" not in st.session_state:
    st.session_state.normal_result = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def par_category(par):
    if par < 50:   return "Very Low",  "#6c757d", "🌑"
    if par < 200:  return "Low",       "#3498db", "🌥️"
    if par < 400:  return "Moderate",  "#2ecc71", "⛅"
    if par < 700:  return "Good",      "#f39c12", "🌤️"
    return             "High",         "#e74c3c", "☀️"

def dli_today(fc_df):
    return round((fc_df["GHI"] * 2.06 * 3600).sum() / 1e6, 1)

def crop_advice(dli):
    if dli < 5:   return "🌿 Shade-tolerant crops (moss, ferns, microgreens)"
    if dli < 10:  return "🥬 Lettuce, spinach, herbs — ideal"
    if dli < 20:  return "🫑 Peppers, cucumbers, tomatoes (greenhouse)"
    if dli < 35:  return "🍅 Tomatoes, most fruiting crops — excellent"
    return              "🌻 Full-sun crops: sunflowers, corn, soybeans"

# ════════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("# 🌱 Normal Mode — PAR Nowcasting")
st.caption("Enter coordinates · Weather auto-fetched from Open-Meteo · Predicted by XGBoost")
st.divider()

if not is_model_available():
    st.error("⚠️ Model not found. Run `git lfs pull` from the project root.")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT: inputs (left 30%) | results (right 70%)
# ════════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 2.3], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
#  LEFT — INPUT PANEL
# ─────────────────────────────────────────────────────────────────────────────
with left:
    with st.container(border=True):

        # ── Coordinates ──────────────────────────────────────────────────────
        st.markdown('<div class="panel-title">📍 Coordinates</div>',
                    unsafe_allow_html=True)
        lat = st.number_input(
            "Latitude (°N)",
            min_value=-90.0, max_value=90.0,
            value=51.6872, step=0.0001, format="%.4f",
            help="Southern hemisphere → negative. Range: −90 to +90",
        )
        lon = st.number_input(
            "Longitude (°E)",
            min_value=-180.0, max_value=180.0,
            value=14.4143, step=0.0001, format="%.4f",
            help="Western hemisphere → negative. Range: −180 to +180",
        )
        alt = st.number_input(
            "Altitude (m)",
            min_value=0.0, max_value=8848.0,
            value=84.0, step=1.0,
            help="Used for precise solar geometry. Enter 0 if unknown.",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Date & Time ───────────────────────────────────────────────────────
        st.markdown('<div class="panel-title">🕐 Date & Time</div>',
                    unsafe_allow_html=True)
        now = datetime.now()
        sel_date = st.date_input("Date", value=date.today())
        sel_time = st.time_input(
            "Local time at that location",
            value=dtime(now.hour, 0),
            step=60,
            help="Cliquez dans le champ ou tapez l'heure au format HH:MM.",
        )
        dt_sel = datetime.combine(sel_date, sel_time)
        st.caption("ℹ️ Timezone is resolved automatically from coordinates.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Predict button ────────────────────────────────────────────────────
        predict_btn = st.button(
            "🌱  Predict PAR",
            use_container_width=True,
            type="primary",
        )

    # ── Reference coordinates ─────────────────────────────────────────────────
    with st.expander("📋 Example coordinates"):
        import pandas as _pd
        _coords = _pd.DataFrame({
            "Location": ["Laubsdorf DE","Nebelin DE","Paris FR",
                         "Cairo EG","Tokyo JP","São Paulo BR"],
            "Lat":  [51.6872, 53.1183, 48.8566, 30.0444, 35.6762, -23.5505],
            "Lon":  [14.4143, 11.7461,  2.3522, 31.2357,139.6503, -46.6333],
            "Alt m": [84, 50, 35, 23, 40, 760],
        })
        st.dataframe(_coords, hide_index=True, use_container_width=True)
        st.caption("Right-click on Google Maps → copy coordinates.")

# ─────────────────────────────────────────────────────────────────────────────
#  HANDLE PREDICT CLICK (runs before right column renders)
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn:
    progress = st.empty()
    with progress.container():
        with st.spinner("⏳ Fetching weather from Open-Meteo…"):
            try:
                weather    = fetch_weather(lat, lon, dt_sel)
                forecast_df = fetch_forecast(lat, lon)
                tz_str     = weather.get("_timezone", "UTC")
            except Exception as e:
                with right:
                    st.error(f"❌ Weather API error: {e}")
                st.stop()

        with st.spinner("⚙️ Computing solar geometry & predicting…"):
            try:
                feat, is_day = compute_features(
                    lat, lon, alt, dt_sel, weather, tz_str
                )
                par_val = predict_par(feat) if is_day else 0.0
            except Exception as e:
                with right:
                    st.error(f"❌ Prediction error: {e}")
                st.stop()

    progress.empty()

    st.session_state.normal_result = {
        "par": par_val, "weather": weather, "features": feat,
        "is_day": is_day, "forecast": forecast_df,
        "lat": lat, "lon": lon, "alt": alt, "dt": dt_sel, "tz": tz_str,
    }

    # Auto-scroll to results section
    _components.html(
        '<script>'
        'setTimeout(function(){'
        '  var e=window.parent.document.getElementById("par-results");'
        '  if(e) e.scrollIntoView({behavior:"smooth",block:"start"});'
        '},400);'
        '</script>',
        height=0,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  RIGHT — RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with right:
    res = st.session_state.normal_result

    if res is None:
        st.markdown("""
        <div class="welcome-card">
            <div style="font-size:3rem;margin-bottom:1rem">🌱</div>
            <div style="font-size:1.15rem;font-weight:700;color:#fff;
                        margin-bottom:.8rem">Ready to predict PAR</div>
            <div style="font-size:.9rem;line-height:1.75">
                Enter <strong style="color:#2ecc71">coordinates</strong>
                and <strong style="color:#2ecc71">date/time</strong> on the left,<br>
                then click <strong style="color:#2ecc71">Predict PAR</strong>.<br><br>
                Weather is fetched <em>automatically</em> for any location on Earth.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Scroll anchor — JS targets this element
        st.markdown('<div id="par-results"></div>', unsafe_allow_html=True)

        par    = res["par"]
        w      = res["weather"]
        ft     = res["features"]
        fc     = res["forecast"]
        is_day = res["is_day"]

        # ── Location info bar ─────────────────────────────────────────────────
        st.markdown(
            f"**📌 {res['lat']:.4f}°N, {res['lon']:.4f}°E** &nbsp; "
            f"alt {res['alt']:.0f} m &nbsp;·&nbsp; "
            f"`{res['tz']}` &nbsp;·&nbsp; "
            f"**{res['dt'].strftime('%Y-%m-%d %H:%M')}**"
        )

        if not is_day:
            st.info(
                "🌙 **Night-time** — sun is below the horizon. PAR = 0.",
                icon="🌑",
            )

        # ── Weather pills ─────────────────────────────────────────────────────
        pills = [
            (f"{w['GHI_RC_01']:.0f}",  "GHI",       "W/m²"),
            (f"{w['Temp_WS']:.1f}",    "Temp",       "°C"),
            (f"{w['RH_WS']:.0f}",      "Humidity",   "%"),
            (f"{w['PREC_INT_WS']:.1f}","Precip.",    "mm/h"),
            (f"{float(ft['clearness_kt'].iloc[0]):.2f}", "Clearness", "kt"),
        ]
        p_cols = st.columns(len(pills), gap="small")
        for col, (val, lbl, unit) in zip(p_cols, pills):
            with col:
                st.markdown(f"""
                <div class="wpill">
                    <div class="wpill-val">{val}</div>
                    <div class="wpill-lbl">{lbl}<br>({unit})</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── PAR card + DLI ────────────────────────────────────────────────────
        label, color, emoji = par_category(par)
        c_par, c_dli = st.columns([1.1, 1], gap="medium")

        with c_par:
            elev = float(ft["elevation"].iloc[0])
            st.markdown(f"""
            <div class="par-card" style="border-color:{color}">
                <div style="font-size:.72rem;color:#8892b0;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:.4rem">
                    🤖 XGBoost Prediction
                </div>
                <div class="par-big" style="color:{color}">{par:.1f}</div>
                <div class="par-unit">µmol / m² / s</div>
                <div class="par-cat" style="color:{color}">{emoji} {label}</div>
                <hr style="border-color:#2a2d3e;margin:.8rem 0">
                <table style="width:100%;font-size:.8rem;color:#8892b0">
                  <tr><td>Solar elevation</td>
                      <td style="color:#fff;text-align:right">{elev:.1f}°</td></tr>
                  <tr><td>Zenith</td>
                      <td style="color:#fff;text-align:right">
                          {float(ft["zenith"].iloc[0]):.1f}°</td></tr>
                  <tr><td>Airmass</td>
                      <td style="color:#fff;text-align:right">
                          {float(ft["airmass"].iloc[0]):.2f}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with c_dli:
            dli = dli_today(fc)
            st.markdown(f"""
            <div class="dli-card">
                <div class="dli-lbl">Daily Light Integral — today</div>
                <div class="dli-val">{dli}
                  <span style="font-size:.85rem;color:#8892b0">mol/m²/day</span>
                </div>
                <div style="margin-top:.7rem;font-size:.86rem;
                            color:#e8eaf6;line-height:1.6">
                    {crop_advice(dli)}
                </div>
            </div>
            <br>
            <div class="dli-card">
                <div class="dli-lbl">Solar &amp; precipitation</div>
                <table style="width:100%;font-size:.82rem;
                              color:#e8eaf6;margin-top:.4rem">
                  <tr>
                    <td style="color:#8892b0">Clearness kt</td>
                    <td style="text-align:right;color:#f39c12">
                        {float(ft["clearness_kt"].iloc[0]):.3f}</td>
                  </tr>
                  <tr>
                    <td style="color:#8892b0">DNI</td>
                    <td style="text-align:right">
                        {float(ft["dni"].iloc[0]):.0f} W/m²</td>
                  </tr>
                  <tr>
                    <td style="color:#8892b0">Raining</td>
                    <td style="text-align:right">
                        {"Yes 🌧️" if ft["is_raining"].iloc[0] else "No ☀️"}</td>
                  </tr>
                  <tr>
                    <td style="color:#8892b0">Dew depression</td>
                    <td style="text-align:right">
                        {float(ft["dew_depression"].iloc[0]):.1f} °C</td>
                  </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Daily forecast chart ──────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:.78rem;font-weight:700;color:#2ecc71;'
            'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.4rem">'
            "Today's Irradiance Forecast</div>",
            unsafe_allow_html=True,
        )
        par_fc = (fc["GHI"] * 2.06).clip(lower=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fc["time"], y=fc["GHI"],
            name="GHI (W/m²)", fill="tozeroy",
            line=dict(color="#f39c12", width=1.5),
            fillcolor="rgba(243,156,18,.12)",
        ))
        fig.add_trace(go.Scatter(
            x=fc["time"], y=par_fc,
            name="PAR est. (µmol/m²/s)",
            line=dict(color="#2ecc71", width=2),
        ))
        fig.add_vline(
            x=res["dt"].isoformat(), line_dash="dash",
            line_color="#ffffff", opacity=0.35,
            annotation_text="selected time",
            annotation_position="top left",
            annotation_font_color="#aaaaaa",
        )
        if par > 0:
            fig.add_trace(go.Scatter(
                x=[res["dt"]], y=[par],
                mode="markers",
                marker=dict(size=12, color="#2ecc71",
                            line=dict(color="#fff", width=2)),
                name=f"ML: {par:.1f} µmol/m²/s",
            ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,46,0.7)",
            font=dict(color="#e8eaf6"),
            height=270,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, tickformat="%H:%M"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch')

        # ── Expandables ───────────────────────────────────────────────────────
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            with st.expander("🔍 All computed features"):
                disp = ft.T.rename(columns={0: "Value"})
                disp["Value"] = disp["Value"].round(5)
                st.dataframe(disp, use_container_width=True)
        with col_exp2:
            with st.expander("🗺️ Location on map"):
                st.map(pd.DataFrame({"lat": [res["lat"]], "lon": [res["lon"]]}),
                       zoom=7)
