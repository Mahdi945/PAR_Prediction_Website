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
from core.cache     import (fetch_weather, fetch_ensemble,
                            DateOutOfRangeError, WeatherServiceError)
from core.features  import compute_features
from core.predict   import (predict_par, par_spread, mccree_estimate,
                            model_status, model_card)
from core.constants import MCCREE_FACTOR, SECONDS_PER_HOUR, MICROMOL_PER_MOL
from core.domain    import check_location, check_features, describe, error_note
from core.export    import download_bar, to_json_bytes, file_name
from core import theme
from core.html import block

# The palette for whichever appearance the visitor has chosen. Charts read
# it directly; the CSS below reads it through the var(--pp-*) variables.
T = theme.tokens()
from core.places    import (place_picker, local_clock, local_now, reference_timezone,
                            identify, KNOWN_SITES)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Normal Mode · ParPredict",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject()
# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }

.panel-title {
    font-size: .78rem; font-weight: 700; color: var(--pp-green-text);
    text-transform: uppercase; letter-spacing: 1.5px;
    border-left: 3px solid var(--pp-green); padding-left: .5rem;
    margin-bottom: .7rem;
}
.par-card {
    background: var(--pp-card-green);
    border: 2px solid; border-radius: 18px;
    padding: 1.6rem; text-align: center;
}
.par-big  { font-size: 4rem; font-weight: 900; line-height: 1; }
.par-unit { font-size: .88rem; color: var(--pp-muted); margin-top: .25rem; }
.par-cat  { font-size: 1rem; font-weight: 700; margin-top: .5rem; }
.wpill {
    background:var(--pp-surface); border:1px solid var(--pp-border); border-radius:10px;
    padding:.55rem .7rem; text-align:center;
}
.wpill-val { font-size:1.3rem; font-weight:800; color:var(--pp-text-strong); }
.wpill-lbl { font-size:.65rem; color:var(--pp-muted); text-transform:uppercase;
             letter-spacing:1px; }
.dli-card {
    background:var(--pp-surface); border:1px solid var(--pp-border); border-radius:14px;
    padding:1.1rem 1.3rem;
}
.dli-val  { font-size:1.9rem; font-weight:900; color:var(--pp-orange-text); }
.dli-lbl  { font-size:.7rem; color:var(--pp-muted); text-transform:uppercase;
            letter-spacing:1px; }
.welcome-card {
    background:var(--pp-surface); border:1px dashed var(--pp-border); border-radius:16px;
    padding:4rem 2rem; text-align:center; color:var(--pp-muted); margin-top: 1rem;
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
    "nm_time": local_clock(_tz_ref),
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# A date left over from an older session moves out of the window as days pass.
st.session_state.nm_date = min(max(st.session_state.nm_date, _win.min_date), _win.max_date)

# Coordinates -> place, so a hand-typed latitude also names its location and
# moves the clock. Re-resolved only when the coordinates actually change; a
# timezone the visitor set themselves is left alone until they move the point.
_point = identify(st.session_state.nm_lat, st.session_state.nm_lon)
_at = (round(st.session_state.nm_lat, 4), round(st.session_state.nm_lon, 4))
if _point and _point.get("timezone") and st.session_state.get("nm_tz_for") != _at:
    st.session_state.nm_tz = _point["timezone"]
    st.session_state.nm_tz_for = _at
    _tz_ref = _point["timezone"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def par_category(par):
    if par < 50:   return "Very Low",  "var(--pp-grey)", "🌑"
    if par < 200:  return "Low",       "var(--pp-blue)", "🌥️"
    if par < 400:  return "Moderate",  "var(--pp-green)", "⛅"
    if par < 700:  return "Good",      "var(--pp-orange)", "🌤️"
    return             "High",         "var(--pp-red)", "☀️"

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
            default=KNOWN_SITES[0],
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

        if _point:
            st.caption(f"📍 These coordinates are in **{_point['display']}**"
                       + (f" · {_point['timezone']}" if _point.get("timezone") else ""))
        else:
            st.caption("📍 This point could not be named — open sea, or the lookup "
                       "was unavailable. The coordinates are used exactly as given.")

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

        # How much does the irradiance forecast itself disagree with itself?
        # Only forecasts have an ensemble, and only daylight has a PAR worth
        # bounding. Any failure here leaves _spread as None and costs the user
        # nothing: an unavailable error bar must not cost them their prediction.
        _spread = None
        if is_day and weather["_horizon_days"] and weather["_horizon_days"] > 0:
            with st.spinner("🎲 Measuring forecast spread…"):
                try:
                    _ens = fetch_ensemble(lat, lon, dt_sel)
                    if _ens:
                        _spread = par_spread(
                            _ens["members"], lat=lat, lon=lon, alt=alt,
                            when=dt_sel, weather=weather, tz_str=tz_str,
                        )
                except Exception:                      # noqa: BLE001 - never fatal
                    _spread = None

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
        "spread":       _spread,
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
            <div style="font-size:1.15rem;font-weight:700;color:var(--pp-text-strong);
                        margin-bottom:.8rem">Ready to predict PAR</div>
            <div style="font-size:.9rem;line-height:1.75">
                Enter <strong style="color:var(--pp-green-text)">coordinates</strong>
                and <strong style="color:var(--pp-green-text)">date/time</strong> on the left,<br>
                then click <strong style="color:var(--pp-green-text)">Predict PAR</strong>.<br><br>
                Weather is fetched <em>automatically</em> for any location on
                Earth — any date from <strong style="color:var(--pp-green-text)">1940</strong>
                up to <strong style="color:var(--pp-green-text)">15 days ahead</strong>.
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
            # No claim about the horizon here any more: the card states the
            # measured ensemble spread, and that generalisation is wrong about
            # any particular day anyway - a clear high-pressure day is
            # predictable a week out, a convective one is not by lunchtime.
            st.info(f"🔮 **{res['source_label']}** · {_matched}")

        if res["missing"]:
            st.warning(
                "⚠️ Open-Meteo had no value for: "
                + ", ".join(res["missing"])
                + " — typical values were used for those inputs."
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
            _mae   = _card["test_mae"]
            _ntest = _card["n_test"]
            # error_note() scopes this to the weather shown, and states the
            # ensemble spread when one was available instead of the vague
            # sentence about the horizon.
            _sp = res.get("spread") or {}
            _err_line = (
                f'<div style="font-size:.8rem;color:var(--pp-muted);margin-top:.35rem" '
                f'title="Mean absolute error on {_ntest:,} held-out test rows from days the '
                f'model never saw, measured with the recorded weather as input. The forecast '
                f'spread is the standard deviation of the prediction across the ensemble '
                f'members, which is the uncertainty in the weather, not in the model.">'
                f'{error_note(_mae, res["horizon"], _sp.get("sd"), _sp.get("n"))}</div>'
            ) if is_day else ""

            # The project's whole claim is that the model beats PAR = 2.06 x GHI,
            # so the visitor should be able to see both numbers at once rather
            # than take the comparison on trust.
            _base = mccree_estimate(float(ft["GHI_RC_01"].iloc[0]))
            _diff = par - _base
            _base_line = (
                f'<div style="font-size:.88rem;color:var(--pp-muted);margin-top:.5rem" '
                f'title="The parameter-free physics formula the model is measured against: '
                f'PAR = {MCCREE_FACTOR} x GHI. It has nothing fitted to this data.">'
                f'<span style="color:var(--pp-text);font-weight:600">physics baseline</span> '
                f'<strong style="color:var(--pp-text-strong)">{_base:,.0f}</strong> '
                f'µmol/m²/s · '
                f'<strong style="color:var(--pp-text-strong)">{_diff:+,.0f}</strong> '
                f'vs the model</div>'
            ) if is_day else ""
            st.markdown(block(f"""
            <div class="par-card" style="border-color:{color}">
                <div style="font-size:.72rem;color:var(--pp-muted);text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:.4rem">
                    🤖 XGBoost Prediction
                </div>
                <div class="par-big" style="color:{color}">{par:.1f}</div>
                <div class="par-unit">µmol / m² / s</div>
                {_base_line}
                {_err_line}
                <div class="par-cat" style="color:{color}">{emoji} {label}</div>
                <hr style="border-color:var(--pp-border);margin:.8rem 0">
                <table style="width:100%;font-size:.8rem;color:var(--pp-muted)">
                  <tr><td>Solar elevation</td>
                      <td style="color:var(--pp-text-strong);text-align:right">{elev:.1f}°</td></tr>
                  <tr><td>Zenith</td>
                      <td style="color:var(--pp-text-strong);text-align:right">
                          {float(ft["zenith"].iloc[0]):.1f}°</td></tr>
                  <tr><td>Airmass</td>
                      <td style="color:var(--pp-text-strong);text-align:right">
                          {float(ft["airmass"].iloc[0]):.2f}</td></tr>
                </table>
            </div>
            """), unsafe_allow_html=True)

        with c_dli:
            dli = dli_for_day(fc)
            st.markdown(block(f"""
            <div class="dli-card">
                <div class="dli-lbl">Daily Light Integral — {res['dt']:%Y-%m-%d}</div>
                <div class="dli-val">{dli}
                  <span style="font-size:.85rem;color:var(--pp-muted)">mol/m²/day</span>
                </div>
                <div style="margin-top:.7rem;font-size:.86rem;
                            color:var(--pp-text);line-height:1.6">
                    {crop_advice(dli)}
                </div>
            </div>
            <br>
            <div class="dli-card">
                <div class="dli-lbl">Solar &amp; precipitation</div>
                <table style="width:100%;font-size:.82rem;
                              color:var(--pp-text);margin-top:.4rem">
                  <tr>
                    <td style="color:var(--pp-muted)">Clearness kt</td>
                    <td style="text-align:right;color:var(--pp-orange-text)">
                        {float(ft["clearness_kt"].iloc[0]):.3f}</td>
                  </tr>
                  <tr>
                    <td style="color:var(--pp-muted)">DNI</td>
                    <td style="text-align:right">
                        {float(ft["dni"].iloc[0]):.0f} W/m²</td>
                  </tr>
                  <tr>
                    <td style="color:var(--pp-muted)">Raining</td>
                    <td style="text-align:right">
                        {"Yes 🌧️" if ft["is_raining"].iloc[0] else "No ☀️"}</td>
                  </tr>
                  <tr>
                    <td style="color:var(--pp-muted)">Dew depression</td>
                    <td style="text-align:right">
                        {float(ft["dew_depression"].iloc[0]):.1f} °C</td>
                  </tr>
                </table>
            </div>
            """), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Irradiance over the selected day ──────────────────────────────────
        if res["source"] == "archive":
            _chart_tag = "historical"
        elif res["horizon"] == 0:
            _chart_tag = "today"
        else:
            _chart_tag = f"forecast +{res['horizon']} d"
        st.markdown(
            '<div style="font-size:.78rem;font-weight:700;color:var(--pp-green-text);'
            'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.4rem">'
            f"Irradiance — {res['dt']:%Y-%m-%d} ({_chart_tag})</div>",
            unsafe_allow_html=True,
        )
        par_fc = (fc["GHI"] * MCCREE_FACTOR).clip(lower=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fc["time"], y=fc["GHI"],
            name="GHI (W/m²)", fill="tozeroy",
            line=dict(color=T["orange"], width=1.5),
            fillcolor=T["orange-soft"],
        ))
        fig.add_trace(go.Scatter(
            x=fc["time"], y=par_fc,
            name="PAR est. (µmol/m²/s)",
            line=dict(color=T["green"], width=2),
        ))
        fig.add_vline(
            x=res["dt"].isoformat(), line_dash="dash",
            line_color=T["text-strong"], opacity=0.35,
            annotation_text="selected time",
            annotation_position="top left",
            annotation_font_color=T["muted"],
        )
        if par > 0:
            fig.add_trace(go.Scatter(
                x=[res["dt"]], y=[par],
                mode="markers",
                marker=dict(size=12, color=T["green"],
                            line=dict(color=T["text-strong"], width=2)),
                name=f"ML: {par:.1f} µmol/m²/s",
            ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=T["chart-plot"],
            font=dict(color=T["text"]),
            height=270,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, tickformat="%H:%M"),
            yaxis=dict(showgrid=True, gridcolor=T["chart-grid"]),
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
                    color=T["green"], zoom=_zoom,
                )
                _where = st.session_state.get("nm_place_applied")
                st.caption(
                    ("📍 " + _where + " — " if _where else "📍 ")
                    + f"**{res['lat']:.4f}°N, {res['lon']:.4f}°E** · "
                    f"{res['alt']:.0f} m · `{res['tz']}`"
                )
                st.caption(f"Nearest training station: **{res.get('nearest', '—')}**, "
                           f"{res.get('distance_km', float('nan')):,.0f} km away.")

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
