"""
pages/1_Normal_Mode.py  –  ParPredict · Normal Mode
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

from core.weather   import available_window
from core.cache     import fetch_weather, DateOutOfRangeError, WeatherServiceError
from core.features  import compute_features
from core.predict   import predict_par, model_status, model_card
from core.constants import MCCREE_FACTOR, SECONDS_PER_HOUR, MICROMOL_PER_MOL
from core.domain    import check_location, check_features, describe
from core.export    import download_bar, to_json_bytes, file_name
from core.places    import place_picker, local_hour, local_now, reference_timezone

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Normal Mode · ParPredict",
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

# The clock the page offers by default. Without a chosen place this is the
# visitor's own timezone, read from their browser — NOT the server's, which on
# Streamlit Cloud is UTC and left a visitor in Germany looking at 14:00 at 16:00.
_tz_ref = reference_timezone(st.session_state.get("nm_tz"), fallback="Europe/Berlin")
_win = available_window()

for _k, _v in {
    "nm_lat":  51.6872,
    "nm_lon":  14.4143,
    "nm_alt":  84.0,
    "nm_tz":   "",                       # filled by the place picker
    "nm_date": local_now(_tz_ref).date(),
    "nm_time": local_hour(_tz_ref),
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# A date left over from an older session moves out of the window as days pass.
st.session_state.nm_date = min(max(st.session_state.nm_date, _win.min_date), _win.max_date)

# ── Helpers ───────────────────────────────────────────────────────────────────
def par_category(par):
    if par < 50:   return "Very Low",  "#6c757d", "🌑"
    if par < 200:  return "Low",       "#3498db", "🌥️"
    if par < 400:  return "Moderate",  "#2ecc71", "⛅"
    if par < 700:  return "Good",      "#f39c12", "🌤️"
    return             "High",         "#e74c3c", "☀️"

def dli_for_day(fc_df):
    # DLI [mol/m²/day] = Σ_hours ( PAR [µmol/m²/s] × 3600 s ) / 1e6
    return round((fc_df["GHI"] * MCCREE_FACTOR * SECONDS_PER_HOUR).sum() / MICROMOL_PER_MOL, 1)

def crop_advice(dli):
    if dli < 5:   return "🌿 Shade-tolerant crops (moss, ferns, microgreens)"
    if dli < 10:  return "🥬 Lettuce, spinach, herbs — ideal"
    if dli < 20:  return "🫑 Peppers, cucumbers, tomatoes (greenhouse)"
    if dli < 35:  return "🍅 Tomatoes, most fruiting crops — excellent"
    return              "🌻 Full-sun crops: sunflowers, corn, soybeans"

# ════════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("# 🌱 Normal Mode — ParPredict")
st.caption("Enter coordinates · Weather from Open-Meteo (1940 → today +15 days) "
           "· Predicted by XGBoost")
st.divider()

_status = model_status()
if not _status.ok:
    st.error(f"⚠️ The prediction model is not usable here. {_status.detail}")
    st.stop()
_card = model_card()

# ════════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT: inputs (left 30%) | results (right 70%)
# ════════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 2.3], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
#  LEFT — INPUT PANEL
# ─────────────────────────────────────────────────────────────────────────────
with left:
    with st.container(border=True):

        # ── Find a place ─────────────────────────────────────────────────────
        st.markdown('<div class="panel-title">🔎 Find a place</div>',
                    unsafe_allow_html=True)
        place_picker(
            key="nm_place",
            lat_key="nm_lat", lon_key="nm_lon", alt_key="nm_alt",
            tz_key="nm_tz", date_key="nm_date", time_key="nm_time",
        )
        if st.session_state.get("nm_place_applied"):
            st.success(f"📍 {st.session_state['nm_place_applied']}"
                       + (f" · {st.session_state['nm_tz']}" if st.session_state.get("nm_tz") else ""))

        # ── Coordinates ──────────────────────────────────────────────────────
        st.markdown('<div class="panel-title">📍 Coordinates</div>',
                    unsafe_allow_html=True)
        st.caption("Filled in by the search above — still editable, so you can "
                   "nudge them from the town centre to your own field.")
        lat = st.number_input(
            "Latitude (°N)",
            min_value=-90.0, max_value=90.0,
            step=0.0001, format="%.4f", key="nm_lat",
            help="Southern hemisphere → negative. Range: −90 to +90",
        )
        lon = st.number_input(
            "Longitude (°E)",
            min_value=-180.0, max_value=180.0,
            step=0.0001, format="%.4f", key="nm_lon",
            help="Western hemisphere → negative. Range: −180 to +180",
        )
        alt = st.number_input(
            "Altitude (m)",
            min_value=0.0, max_value=8848.0,
            step=1.0, key="nm_alt",
            help="Used for precise solar geometry. Enter 0 if unknown.",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Date & Time ───────────────────────────────────────────────────────
        st.markdown('<div class="panel-title">🕐 Date & Time</div>',
                    unsafe_allow_html=True)
        win = _win
        sel_date = st.date_input(
            "Date",
            key="nm_date",
            min_value=win.min_date,
            max_value=win.max_date,
            help="Past dates use ERA5 reanalysis; today and future dates use "
                 "the forecast model.",
        )
        sel_time = st.time_input(
            "Local time at that location",
            step=60, key="nm_time",
            help="Click the field or type the hour as HH:MM. "
                 "Weather is hourly, so minutes are ignored.",
        )
        dt_sel = datetime.combine(sel_date, sel_time)
        _clock = local_now(_tz_ref)
        st.caption(
            f"🕒 Now in **{_tz_ref}**: {_clock:%H:%M} on {_clock:%Y-%m-%d}"
            + ("  — the zone of the place you picked." if st.session_state.get("nm_tz")
               else "  — your own timezone. Pick a place above to use its clock instead.")
        )
        st.caption(
            f"📅 Weather available **{win.min_date:%Y-%m-%d} → "
            f"{win.max_date:%Y-%m-%d}** — ERA5 archive up to yesterday, "
            f"forecast to today +15."
        )

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
                # One request: the point value and the day series share a source.
                weather     = fetch_weather(lat, lon, dt_sel)
                forecast_df = weather["_day_series"]
                tz_str      = weather.get("_timezone", "UTC")
            except DateOutOfRangeError as e:
                with right:
                    st.error(f"📅 {e}")
                st.stop()
            except WeatherServiceError as e:
                with right:
                    st.error(f"❌ Weather data unavailable: {e}")
                st.stop()
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

    # Has the model seen conditions like these? Two Brandenburg stations, 2024–2025.
    _loc_check = check_location(lat, lon)
    _domain    = describe(_loc_check, check_features(feat) if is_day else [])

    st.session_state.normal_result = {
        "par": par_val, "weather": weather, "features": feat,
        "is_day": is_day, "forecast": forecast_df,
        "lat": lat, "lon": lon, "alt": alt, "dt": dt_sel, "tz": tz_str,
        "source":       weather["_source"],
        "source_label": weather["_source_label"],
        "horizon":      weather["_horizon_days"],
        "matched_time": weather["_matched_time"],
        "missing":      weather["_missing"],
        "domain":       _domain,
        "nearest":      _loc_check.nearest,
        "distance_km":  _loc_check.distance_km,
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
                Weather is fetched <em>automatically</em> for any location on
                Earth — any date from <strong style="color:#2ecc71">1940</strong>
                up to <strong style="color:#2ecc71">15 days ahead</strong>.
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

        # ── Where these numbers come from ─────────────────────────────────────
        _matched = f"Matched hour: {res['matched_time'].replace('T', ' ')} local."
        if res["source"] == "archive":
            st.info(
                f"📜 **{res['source_label']}** — modelled, gridded reanalysis "
                f"(~25 km), not a station measurement. Against our two "
                f"pyranometers ERA5 read about 9–18 % high on GHI, and PAR "
                f"follows GHI almost one for one. {_matched}"
            )
        elif res["horizon"] == 0:
            st.caption(f"🛰️ {res['source_label']} · {_matched}")
        else:
            st.info(
                f"🔮 **{res['source_label']}** — forecast uncertainty grows "
                f"with the horizon. {_matched}"
            )

        if res["missing"]:
            st.warning(
                "⚠️ Open-Meteo had no value for: "
                + ", ".join(res["missing"])
                + " — typical values were used for those inputs."
            )

        for _note in res.get("domain", []):
            st.warning(f"🧭 {_note}")

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
            _mae   = _card["test_mae"]
            _ntest = _card["n_test"]
            _err_line = (
                f'<div style="font-size:.8rem;color:#8892b0;margin-top:.35rem" '
                f'title="Mean absolute error on {_ntest:,} held-out test rows from days the model never saw">'
                f'typical error ± {_mae:.0f} µmol/m²/s</div>'
            ) if is_day else ""
            st.markdown(f"""
            <div class="par-card" style="border-color:{color}">
                <div style="font-size:.72rem;color:#8892b0;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:.4rem">
                    🤖 XGBoost Prediction
                </div>
                <div class="par-big" style="color:{color}">{par:.1f}</div>
                <div class="par-unit">µmol / m² / s</div>
                {_err_line}
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
            dli = dli_for_day(fc)
            st.markdown(f"""
            <div class="dli-card">
                <div class="dli-lbl">Daily Light Integral — {res['dt']:%Y-%m-%d}</div>
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

        # ── Irradiance over the selected day ──────────────────────────────────
        if res["source"] == "archive":
            _chart_tag = "historical"
        elif res["horizon"] == 0:
            _chart_tag = "today"
        else:
            _chart_tag = f"forecast +{res['horizon']} d"
        st.markdown(
            '<div style="font-size:.78rem;font-weight:700;color:#2ecc71;'
            'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.4rem">'
            f"Irradiance — {res['dt']:%Y-%m-%d} ({_chart_tag})</div>",
            unsafe_allow_html=True,
        )
        par_fc = (fc["GHI"] * MCCREE_FACTOR).clip(lower=0)

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
            with st.expander("🗺️ Location on map", expanded=False):
                _zoom_label = st.radio(
                    "Zoom", ["Field", "Town", "Region"], index=1,
                    horizontal=True, key="nm_map_zoom",
                    help="How closely to frame the predicted point.",
                )
                _zoom = {"Field": 14, "Town": 11, "Region": 7}[_zoom_label]
                st.map(
                    pd.DataFrame({"lat": [res["lat"]], "lon": [res["lon"]]}),
                    latitude="lat", longitude="lon",
                    size=({"Field": 12, "Town": 90, "Region": 700})[_zoom_label],
                    color="#2ecc71", zoom=_zoom,
                )
                _where = st.session_state.get("nm_place_applied")
                st.caption(
                    ("📍 " + _where + " — " if _where else "📍 ")
                    + f"**{res['lat']:.4f}°N, {res['lon']:.4f}°E** · "
                    f"{res['alt']:.0f} m · `{res['tz']}`"
                )
                st.caption(f"Nearest training station: **{res.get('nearest', '—')}**, "
                           f"{res.get('distance_km', float('nan')):,.0f} km away.")

        # ── Export ────────────────────────────────────────────────────────────
        with st.expander("⬇ Export this prediction"):
            _summary = pd.DataFrame([{
                "timestamp_local":      res["dt"],
                "timezone":             res["tz"],
                "latitude":             res["lat"],
                "longitude":            res["lon"],
                "altitude_m":           res["alt"],
                "weather_source":       res["source_label"],
                "matched_hour_local":   res["matched_time"],
                "GHI_W_m2":             w.get("GHI_RC_01"),
                "temperature_C":        w.get("Temp_WS"),
                "humidity_pct":         w.get("RH_WS"),
                "dew_point_C":          w.get("DWP_WS"),
                "wind_speed_m_s":       w.get("WS_WS"),
                "wind_direction_deg":   w.get("WD_WS"),
                "precipitation_mm_h":   w.get("PREC_INT_WS"),
                "solar_elevation_deg":  float(ft["elevation"].iloc[0]),
                "clearness_index":      float(ft["clearness_kt"].iloc[0]),
                "is_daytime":           bool(is_day),
                "PAR_model_umol_m2_s":  par,
                "PAR_mccree_umol_m2_s": float(w.get("GHI_RC_01", 0.0)) * MCCREE_FACTOR,
                "typical_error_umol_m2_s": _card["test_mae"],
                "DLI_mccree_mol_m2_day": dli,
                "nearest_training_station": res.get("nearest"),
                "distance_to_training_km":  res.get("distance_km"),
            }])
            st.caption("Prediction summary — one row with inputs, sources and both estimates")
            download_bar(_summary, "prediction", key="nm_dl_summary", label="Summary",
                         formats=["csv", "xlsx", "json"])
            st.caption("Irradiance over the selected day — hourly GHI, temperature and rain from Open-Meteo")
            download_bar(fc, "day_series", key="nm_dl_day", label="Day series")
            st.caption("All 22 computed features — what the model actually saw")
            download_bar(ft, "features", key="nm_dl_features", label="Features",
                         formats=["csv", "xlsx", "json"])
            _report = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "request":  {"latitude": res["lat"], "longitude": res["lon"], "altitude_m": res["alt"],
                             "local_time": res["dt"], "timezone": res["tz"]},
                "weather":  {k: v for k, v in w.items() if not k.startswith("_")},
                "weather_source": {"kind": res["source"], "label": res["source_label"],
                                   "matched_hour": res["matched_time"], "horizon_days": res["horizon"],
                                   "substituted_inputs": res["missing"]},
                "prediction": {"PAR_model_umol_m2_s": par,
                               "PAR_mccree_umol_m2_s": float(w.get("GHI_RC_01", 0.0)) * MCCREE_FACTOR,
                               "is_daytime": bool(is_day), "DLI_mccree_mol_m2_day": dli},
                "domain_check": {"notes": res.get("domain", []), "nearest_training_station": res.get("nearest"),
                                 "distance_km": res.get("distance_km")},
                "features": ft.iloc[0].to_dict(),
                "model": _card,
            }
            st.download_button("⬇ Full report · JSON", data=lambda: to_json_bytes(_report),
                               file_name=file_name("report", "json"), mime="application/json",
                               key="nm_dl_report", use_container_width=True)
