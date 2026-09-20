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
    predict_par, mccree_estimate, get_feature_importance, model_status, model_card
)
from core.weather  import available_window
from core.cache    import fetch_weather, DateOutOfRangeError, WeatherServiceError
from core.domain   import check_location, check_features, describe
from core.export   import download_bar, to_json_bytes, file_name
from core.places   import (place_picker, local_clock, local_now, reference_timezone,
                           identify, KNOWN_SITES)

# Widget bounds, defined once and reused by both the sliders/number inputs and
# the auto-fetch clamp. A fetched value outside a widget's range (−31 °C in
# Yakutsk, 45 m/s wind) makes Streamlit raise on the next rerun.
_RANGES = {
    "e_ghi":  (0.0, 1400.0),
    "e_temp": (-20.0,  50.0),
    "e_rh":   (0.0,   100.0),
    "e_dwp":  (-20.0,  40.0),
    "e_ws":   (0.0,    40.0),
    "e_wd":   (0.0,   360.0),
    "e_prec": (0.0,    30.0),
}

# session_state key → the key fetch_weather() returns it under
_FETCH_KEYS = [
    ("e_ghi",  "GHI_RC_01",   "GHI"),
    ("e_temp", "Temp_WS",     "temperature"),
    ("e_rh",   "RH_WS",       "humidity"),
    ("e_dwp",  "DWP_WS",      "dew point"),
    ("e_ws",   "WS_WS",       "wind speed"),
    ("e_wd",   "WD_WS",       "wind direction"),
    ("e_prec", "PREC_INT_WS", "precipitation"),
]

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

# ensure expert inputs persist between reruns.
# The clock starts on the visitor's own timezone (from their browser), not the
# server's — on Streamlit Cloud that is UTC, which showed 14:00 to someone
# looking at a 16:00 wall clock. Once a place is picked, its zone takes over.
_tz_start = reference_timezone(None, fallback="Europe/Berlin")
for key, value in {
    "e_lat":  51.6872,
    "e_lon":  14.4143,
    "e_alt":  84.0,
    "e_tz":   _tz_start,
    "e_date": local_now(_tz_start).date(),
    "e_time": local_clock(_tz_start),
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

# A date left over from an older session could now sit outside the window
# (it moves forward every day), which makes st.date_input raise.
_win = available_window()
st.session_state.e_date = min(max(st.session_state.e_date, _win.min_date),
                             _win.max_date)

# Here the timezone is not just a caption: Expert Mode passes e_tz straight to
# compute_features, so a stale zone would place the sun in the wrong part of
# the sky. Whenever the coordinates move, the zone follows them.
_point = identify(st.session_state.e_lat, st.session_state.e_lon)
_at = (round(st.session_state.e_lat, 4), round(st.session_state.e_lon, 4))
if _point and _point.get("timezone") and st.session_state.get("e_tz_for") != _at:
    st.session_state.e_tz = _point["timezone"]
    st.session_state.e_tz_for = _at

if "expert_autofetch_temp" in st.session_state:
    fetched = st.session_state.pop("expert_autofetch_temp")

    clamped = []
    for key, src, label in _FETCH_KEYS:
        lo, hi = _RANGES[key]
        value  = float(fetched.get(src, st.session_state[key]))
        if value < lo or value > hi:
            clamped.append(f"{label} {value:.1f} → {min(max(value, lo), hi):.1f}")
            value = min(max(value, lo), hi)
        st.session_state[key] = value

    # Without this the typed timezone stays put and the solar geometry is
    # computed for the wrong zone.
    st.session_state.e_tz = fetched.get("_timezone", st.session_state.e_tz)

    notes = []
    if fetched.get("_missing"):
        notes.append("no Open-Meteo value for " + ", ".join(fetched["_missing"]))
    if clamped:
        notes.append("clamped to the input range: " + "; ".join(clamped))

    status_type = "warning" if notes else "success"
    message = (
        f"✅ Weather fetched — {fetched['_source_label']} · "
        f"{fetched['_matched_time'].replace('T', ' ')} "
        f"({fetched['_timezone']}). Timezone field updated."
    )
    if notes:
        message += "  \n⚠️ " + " · ".join(notes) + "."
    st.session_state.expert_autofetch_status = (status_type, message)


def _apply_expert_autofetch():
    try:
        dt_sel = datetime.combine(
            st.session_state.e_date,
            st.session_state.e_time,
        )
        st.session_state.expert_autofetch_temp = fetch_weather(
            st.session_state.e_lat,
            st.session_state.e_lon,
            dt_sel,
        )
        st.session_state.pop("expert_autofetch_status", None)
    except DateOutOfRangeError as e:
        st.session_state.expert_autofetch_status = ("error", f"📅 {e}")
    except WeatherServiceError as e:
        st.session_state.expert_autofetch_status = (
            "error", f"❌ Open-Meteo fetch failed: {e}"
        )
    except Exception as e:
        st.session_state.expert_autofetch_status = (
            "error", f"❌ Unexpected error: {e}"
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

_status = model_status()
if not _status.ok:
    st.error(f"⚠️ The prediction model is not usable here. {_status.detail}")
    st.stop()
_card = model_card()

# ════════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT: inputs (left) | results (right)
# ════════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 2.3], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
#  LEFT — INPUT PANEL
# ─────────────────────────────────────────────────────────────────────────────
with left:
    with st.container(border=True):

        # ── Find a place ──────────────────────────────────────────────────────
        st.markdown('<div class="panel-title">🔎 Find a place</div>',
                    unsafe_allow_html=True)
        place_picker(
            key="em_place",
            lat_key="e_lat", lon_key="e_lon", alt_key="e_alt",
            tz_key="e_tz", date_key="e_date", time_key="e_time",
            default=KNOWN_SITES[0],
        )
        if st.session_state.get("em_place_applied"):
            st.success(f"📍 {st.session_state['em_place_applied']} · {st.session_state['e_tz']}")

        # ── Location & Time ───────────────────────────────────────────────────
        st.markdown('<div class="panel-title">📍 Location & Time</div>',
                    unsafe_allow_html=True)
        st.caption("Filled in by the search above — still editable.")
        lat = st.number_input("Latitude (°N)",  -90.0,  90.0,
                              format="%.4f", step=0.0001, key="e_lat")
        lon = st.number_input("Longitude (°E)", -180.0, 180.0,
                              format="%.4f", step=0.0001, key="e_lon")
        alt = st.number_input("Altitude (m)",    0.0, 8848.0,
                              step=1.0, key="e_alt")
        if _point:
            st.caption(f"📍 These coordinates are in **{_point['display']}**")
        # No default: the zone is resolved from the coordinates above.
        tz  = st.text_input("Timezone (IANA)", key="e_tz",
                            help="Set from the coordinates, and by auto-fetch. "
                                 "Edit it only if you know better — it decides "
                                 "where the sun is.")

        # No positional default: session_state already seeds these keys, and
        # passing both makes Streamlit warn.
        sel_date = st.date_input(
            "Date",
            key="e_date",
            min_value=_win.min_date,
            max_value=_win.max_date,
            help="Past dates use ERA5 reanalysis; today and future dates use "
                 "the forecast model.",
        )
        sel_time = st.time_input(
            "Local time",
            step=60,
            key="e_time",
            help="Click the field or type the hour as HH:MM. "
                 "Weather is hourly, so minutes are ignored.",
        )
        dt_sel = datetime.combine(sel_date, sel_time)
        _clock = local_now(st.session_state.e_tz)
        st.caption(f"🕒 Now in **{st.session_state.e_tz}**: "
                   f"{_clock:%H:%M} on {_clock:%Y-%m-%d}")
        st.caption(
            f"🌦️ Auto-fetch covers **{_win.min_date:%Y-%m-%d} → "
            f"{_win.max_date:%Y-%m-%d}**. With readings entered by hand the "
            f"prediction works for any date — solar geometry is computed "
            f"locally, not fetched."
        )

        # ── Solar Irradiance ──────────────────────────────────────────────────
        st.markdown('<div class="panel-title">☀️ Solar Irradiance</div>',
                    unsafe_allow_html=True)
        ghi = st.slider("GHI — Global Horizontal Irradiance (W/m²)",
                        *_RANGES["e_ghi"], step=1.0, key="e_ghi")

        # ── Meteorological Sensors ─────────────────────────────────────────────
        st.markdown('<div class="panel-title">🌡️ Meteorological Sensors</div>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            temp = st.number_input("Temperature (°C)", *_RANGES["e_temp"],
                                   step=0.1, key="e_temp")
            rh   = st.number_input("Humidity (%)",     *_RANGES["e_rh"],
                                   step=0.5, key="e_rh")
            dwp  = st.number_input("Dewpoint (°C)",    *_RANGES["e_dwp"],
                                   step=0.1, key="e_dwp")
        with c2:
            ws   = st.number_input("Wind speed (m/s)", *_RANGES["e_ws"],
                                   step=0.1, key="e_ws")
            wd   = st.number_input("Wind dir (°)",     *_RANGES["e_wd"],
                                   step=1.0, key="e_wd")
            prec = st.number_input("Precipitation (mm/h)", *_RANGES["e_prec"],
                                   step=0.1, key="e_prec")

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
            elif status_type == "warning":
                st.warning(status_msg)
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

    _loc_check = check_location(lat, lon)
    st.session_state.expert_result = {
        "par": par, "mc": mc, "features": feat,
        "is_day": is_day, "imp": imp,
        "inputs": weather,
        "lat": lat, "lon": lon, "alt": alt, "dt": dt_sel, "tz": tz,
        "domain":      describe(_loc_check, check_features(feat) if is_day else []),
        "nearest":     _loc_check.nearest,
        "distance_km": _loc_check.distance_km,
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
            _mae   = _card["test_mae"]
            _ntest = _card["n_test"]
            _err_line = (
                f'<div style="color:#8892b0;font-size:.75rem;margin-top:.4rem" '
                f'title="Mean absolute error on {_ntest:,} held-out test rows from days the model never saw">'
                f'typical error ± {_mae:.0f} µmol/m²/s</div>'
            ) if is_day else ""
            st.markdown(f"""
            <div class="result-card" style="background:linear-gradient(135deg,#0d2b1a,#0f1117);
                 border-color:{col_ml}">
                <div class="cap-lbl">🤖 ML Model (XGBoost)</div>
                <div class="big-num" style="color:{col_ml}">{par:.1f}</div>
                <div class="unit">µmol / m² / s</div>
                {_err_line}
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

        # The domain notes are deliberately quiet. They matter, but a yellow
        # banner on every prediction outside Germany trains people to ignore
        # banners, and the same facts are in the map caption, in this expander
        # and in the JSON export.
        _notes = res.get("domain", [])
        if _notes:
            with st.expander(f"🧭 Model domain — {len(_notes)} note"
                             f"{'s' if len(_notes) != 1 else ''}"):
                for _n in _notes:
                    st.caption(_n)

        # ── Export ────────────────────────────────────────────────────────────
        with st.expander("⬇ Export this prediction"):
            _in = res["inputs"]
            _summary = pd.DataFrame([{
                "timestamp_local":      res["dt"],
                "timezone":             res["tz"],
                "latitude":             res["lat"],
                "longitude":            res["lon"],
                "altitude_m":           res["alt"],
                "GHI_W_m2":             _in.get("GHI_RC_01"),
                "temperature_C":        _in.get("Temp_WS"),
                "humidity_pct":         _in.get("RH_WS"),
                "dew_point_C":          _in.get("DWP_WS"),
                "wind_speed_m_s":       _in.get("WS_WS"),
                "wind_direction_deg":   _in.get("WD_WS"),
                "precipitation_mm_h":   _in.get("PREC_INT_WS"),
                "solar_elevation_deg":  float(ft["elevation"].iloc[0]),
                "clearness_index":      float(ft["clearness_kt"].iloc[0]),
                "is_daytime":           bool(is_day),
                "PAR_model_umol_m2_s":  par,
                "PAR_mccree_umol_m2_s": mc,
                "difference_umol_m2_s": par - mc,
                "typical_error_umol_m2_s": _card["test_mae"],
                "nearest_training_station": res.get("nearest"),
                "distance_to_training_km":  res.get("distance_km"),
            }])
            st.caption("Prediction summary — your readings, both estimates and their difference")
            download_bar(_summary, "expert_prediction", key="em_dl_summary", label="Summary",
                         formats=["csv", "xlsx", "json"])
            st.caption("All 22 computed features — what the model actually saw")
            download_bar(ft, "features", key="em_dl_features", label="Features",
                         formats=["csv", "xlsx", "json"])
            if imp is not None:
                _imp_df = imp.rename("importance").rename_axis("feature").reset_index()
                st.caption("Feature importance of the deployed model")
                download_bar(_imp_df, "feature_importance", key="em_dl_importance",
                             label="Importance", formats=["csv", "xlsx", "json"])
            _report = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "request":  {"latitude": res["lat"], "longitude": res["lon"], "altitude_m": res["alt"],
                             "local_time": res["dt"], "timezone": res["tz"]},
                "inputs":   dict(_in),
                "prediction": {"PAR_model_umol_m2_s": par, "PAR_mccree_umol_m2_s": mc,
                               "difference_umol_m2_s": par - mc, "is_daytime": bool(is_day)},
                "domain_check": {"notes": res.get("domain", []), "nearest_training_station": res.get("nearest"),
                                 "distance_km": res.get("distance_km")},
                "features": ft.iloc[0].to_dict(),
                "feature_importance": imp.to_dict() if imp is not None else None,
                "model": _card,
            }
            st.download_button("⬇ Full report · JSON", data=lambda: to_json_bytes(_report),
                               file_name=file_name("expert_report", "json"), mime="application/json",
                               key="em_dl_report", use_container_width=True)
