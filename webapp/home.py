"""
home.py  –  ParPredict · Home / Landing Page
"""

import base64
import streamlit as st
from pathlib import Path
from core.predict import model_status
from core import theme

# The palette for whichever appearance the visitor has chosen. Charts read
# it directly; the CSS below reads it through the var(--pp-*) variables.
T = theme.tokens()

# ── Load the logo ────────────────────────────────────────────────────────────
# logo.png wins when it is there, otherwise the original logo.svg — swapping
# the artwork is then a file change, not a code change. The mark is the badge
# alone: the hero prints "ParPredict" in type right underneath, so a lockup
# carrying its own wordmark would show the name twice (assets/logo_lockup.png
# is that version, if it is ever wanted instead).
_ASSETS = Path(__file__).parent / "assets"
_logo_img = ""
# The lockup writes "Par" in white, which disappears on a light background, so
# light mode gets a copy with that word inked dark. Same artwork otherwise.
_LOCKUP = "logo_lockup.png" if theme.is_dark() else "logo_lockup_light.png"
for _name, _mime in ((_LOCKUP, "image/png"),
                     ("logo_lockup.png", "image/png"),
                     ("logo.png", "image/png"),
                     ("logo.svg", "image/svg+xml")):
    _p = _ASSETS / _name
    if _p.exists():
        _logo_data = base64.b64encode(_p.read_bytes()).decode("ascii")
        _logo_img = (
            f'<img class="hero-logo" src="data:{_mime};base64,{_logo_data}" '
            f'alt="ParPredict" />'
        )
        break

theme.inject()
# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default Streamlit header padding */
.block-container { padding-top: 1.5rem; }

/* Hero */
/* Streamlit turns an <h1> in markdown into a heading COMPONENT: it wraps it in
   [data-testid="stHeadingWithActionElements"] and injects a hidden anchor link
   inside. Measured on the running app, that left the h1 281 px tall around a
   193 px image - 20 px padding above, 16 px below, and a 52.8 px empty line box
   for the anchor. Its styles outrank a plain .hero-title, hence !important:
   font-size 0 collapses the anchor's line box, the rest removes the padding. */
h1.hero-title {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 0 !important;
    line-height: 0 !important;
}
/* The name is drawn in the artwork, so the heading holds the image instead of
   type. It stays an <h1> so the page keeps a real heading for screen readers
   and for the document outline; the alt text carries the name. */
/* Selector deliberately long: Streamlit ships
   [data-testid="stMarkdownContainer"] img { max-width: 100% }, which is more
   specific than a lone .hero-logo and silently won - the logo rendered at its
   natural 520 px. .hero-title img.hero-logo outranks it without !important. */
.hero-title img.hero-logo {
    display: block; margin: 0 auto;
    width: 100%; max-width: 210px; height: auto;
}
/* Streamlit's own [data-testid="stMarkdownContainer"] p rule zeroes the top
   margin, so the gap under the logo has to be stated here or it collapses to
   nothing. This one number is the whole spacing between logo and tagline. */
p.hero-sub {
    font-size: 1.1rem; color: var(--pp-muted); line-height: 1.6;
    margin: 0.85rem 0 0.2rem 0 !important;
}

/* Mode cards */
.mode-card {
    background: linear-gradient(135deg, var(--pp-surface) 0%, var(--pp-bg) 100%);
    border-radius: 18px; padding: 2.2rem 1.8rem;
    text-align: center;
    height: 360px;
    width: 100%;
    max-width: 360px;
    display: flex; flex-direction: column;
    justify-content: space-between; align-items: center;
    transition: transform .2s, box-shadow .2s;
    box-sizing: border-box;
    overflow: hidden;
    margin: 0 auto;
}
.mode-card-green  { border: 2px solid var(--pp-green); }
.mode-card-orange { border: 2px solid var(--pp-orange); }
.mode-card:hover  { transform: translateY(-3px);
                    box-shadow: 0 12px 32px rgba(46,204,113,.25); }
a.mode-link       { text-decoration: none !important; display: block; width: 100%; height: 100%; }
a.mode-link:hover .mode-card-green  { box-shadow: 0 12px 32px rgba(46,204,113,.35); transform: translateY(-3px); }
a.mode-link:hover .mode-card-orange { box-shadow: 0 12px 32px rgba(243,156,18,.35); transform: translateY(-3px); }
.mode-icon  { font-size: 3.2rem; margin-bottom: .8rem; flex-shrink: 0; }
.mode-title { font-size: 1.5rem; font-weight: 800; color: var(--pp-text-strong);
              margin: .4rem 0; flex-shrink: 0; }
.mode-desc  { color: var(--pp-muted); font-size: .9rem; line-height: 1.65;
              flex: 1; overflow-wrap: anywhere; }

/* Force Streamlit columns that hold mode cards to equal height */
div[data-testid="column"] > div:first-child {
    height: 100%;
}

/* Stats strip */
.stats-bar {
    display: flex; justify-content: space-around;
    background: var(--pp-surface); border: 1px solid var(--pp-border);
    border-radius: 14px; padding: 1.4rem 1rem; margin: 2rem 0;
}
.stat-item { text-align: center; }
.stat-val { font-size: 1.9rem; font-weight: 900; color: var(--pp-green-text); }
.stat-lbl { font-size: .75rem; color: var(--pp-muted);
            text-transform: uppercase; letter-spacing: 1.5px; }

/* How-it-works steps */
.step-card {
    background: var(--pp-surface); border: 1px solid var(--pp-border);
    border-radius: 14px; padding: 1.4rem 1rem;
    text-align: center;
    height: 200px;                   /* fixed equal height for all cards */
    min-height: 200px;
    max-height: 200px;
    display: flex; flex-direction: column;
    justify-content: flex-start; align-items: center;
    box-sizing: border-box;
    overflow: hidden;
}
.step-icon  { font-size: 2rem; flex-shrink: 0; }
.step-title { font-size: 1rem; font-weight: 700; color: var(--pp-text-strong);
              margin: .5rem 0 .3rem 0; flex-shrink: 0; }
.step-desc  { color: var(--pp-muted); font-size: .78rem; line-height: 1.5;
              flex-shrink: 0; overflow: hidden; }

/* Warning / info banner */
.model-warn {
    background: var(--pp-warn-bg); border: 1px solid var(--pp-orange);
    border-radius: 10px; padding: .9rem 1.2rem;
    color: var(--pp-warn-text); font-size: .88rem;
}

/* Footer — stick to page bottom */
.footer-wrap {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: var(--pp-bg);
    border-top: 1px solid var(--pp-surface);
    text-align: center; color: var(--pp-faint);
    font-size: .76rem; padding: .6rem 1rem;
    z-index: 999;
}

@media (max-width: 900px) {
    .hero-title img.hero-logo { max-width: 180px; }
    .hero-sub { font-size: .98rem; }
    .stats-bar { flex-wrap: wrap; gap: .75rem; padding: 1rem; }
    .stat-item { width: 48%; margin-bottom: .75rem; }
    .mode-card, .step-card { height: auto; min-height: auto; padding: 1.4rem; margin-bottom: 1rem; }
    .mode-card { max-width: 100%; }
    .mode-title { font-size: 1.2rem; }
    .mode-desc { font-size: .82rem; }
    .step-title { font-size: .95rem; }
    .step-desc { font-size: .78rem; }
    div[data-testid="column"] > div:first-child { min-width: 100% !important; }
    section[data-testid="stHorizontalBlock"] { gap: 1rem !important; }
    a.mode-link { display: block; margin-bottom: 1rem; }
}
@media (max-width: 1200px) {
    .mode-card { max-width: 320px; }
}
@media (max-width: 640px) {
    .hero-title img.hero-logo { max-width: 155px; }
    .hero-sub { font-size: .9rem; }
    .stats-bar { flex-direction: column; }
    .stat-item { width: 100%; text-align: left; }
    .mode-card, .step-card { min-height: auto; }
    .step-card { padding: 1rem; }
    .mode-card { padding: 1.2rem; margin-bottom: 1rem; }
    .mode-icon { margin-bottom: .5rem; }
    .footer-wrap { font-size: .72rem; padding: .55rem .9rem; }
    div[data-testid="column"] > div:first-child { min-width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HERO SECTION
# ═══════════════════════════════════════════════════════════════════════════════
_, hero_col, _ = st.columns([1, 2.5, 1])
with hero_col:
    # The logo and the tagline are ONE markdown block on purpose. As two
    # separate st.markdown calls Streamlit inserts its own vertical gap between
    # them, which no CSS of ours was touching - that was the space under the
    # logo. Spacing here is the .hero-sub margin alone.
    st.markdown(
        f'''
    <div style="text-align:center">
        <h1 class="hero-title">{_logo_img}</h1>
        <p class="hero-sub">
            Predict Photosynthetically Active Radiation for Agrivoltaic Systems<br>
            <small style="color:var(--pp-faint)">
                Powered by XGBoost &nbsp;·&nbsp; pvlib &nbsp;·&nbsp;
                Open-Meteo &nbsp;·&nbsp; Hochschule Anhalt 2026
            </small>
        </p>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL STATUS BANNER
# ═══════════════════════════════════════════════════════════════════════════════
_status = model_status()
if not _status.ok:
    st.markdown(f"""
    <div class="model-warn">
        ⚠️ <strong>The prediction model is not usable on this deployment.</strong><br>
        <span style="font-size:.9rem">{_status.detail}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODE SELECTION CARDS  (entire card is clickable — no separate button)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### Choose your mode")
col_a, col_b, col_c = st.columns(3, gap="large")

with col_a:
    st.markdown("""
    <a href="Normal_Mode" target="_self" class="mode-link">
      <div class="mode-card mode-card-green">
        <div class="mode-icon">🌱</div>
        <div class="mode-title">Normal Mode</div>
        <div class="mode-desc">
            Search for any city or enter coordinates.<br>
            Weather is fetched <em>automatically</em> from Open-Meteo —
            any date from 1940 to 15 days ahead.
        </div>
        <div style="margin-top:1rem">
          <strong style="color:var(--pp-green-text)">3 inputs &nbsp;·&nbsp; One prediction</strong>
        </div>
      </div>
    </a>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <a href="Expert_Mode" target="_self" class="mode-link">
      <div class="mode-card mode-card-orange">
        <div class="mode-icon">⚙️</div>
        <div class="mode-title">Expert Mode</div>
        <div style="margin:.3rem 0 .7rem 0">
          <strong style="color:var(--pp-orange-text)">Full control &nbsp;·&nbsp; Full transparency</strong>
        </div>
        <div class="mode-desc">
            Enter your own sensor readings for maximum accuracy.<br>
            Inspect feature importance, McCree comparison,<br>
            and every intermediate computed value.
        </div>
      </div>
    </a>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <a href="Dataset_Upload" target="_self" class="mode-link">
      <div class="mode-card" style="border: 2px solid var(--pp-blue);">
        <div class="mode-icon">📊</div>
        <div class="mode-title">Dataset Upload</div>
        <div style="margin:.3rem 0 .7rem 0">
          <strong style="color:var(--pp-blue-text)">Batch analysis &nbsp;·&nbsp; Full benchmark</strong>
        </div>
        <div class="mode-desc">
            Upload a full time-series dataset with sensor readings and location.<br>
            Clean it, resample it, and compare model predictions to a baseline.
        </div>
      </div>
    </a>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Documentation ─────────────────────────────────────────────────────────────
# Directly under the three modes: the moment someone wonders which one they
# want is the moment the manual is worth offering.
_, _doc_col, _ = st.columns([1, 2.2, 1])
with _doc_col:
    st.markdown(
        '<div style="text-align:center;color:var(--pp-muted);font-size:.92rem;'
        'margin-bottom:.35rem">New here, or unsure which mode fits? '
        'The documentation explains every part of the site &mdash; '
        'in English and German.</div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/4_Documentation.py",
                 label="Open the documentation  ·  Dokumentation öffnen",
                 icon="📖", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODE COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:var(--pp-surface);border:1px solid var(--pp-border);border-radius:14px;
            padding:1.2rem 1.6rem;margin-bottom:1.4rem">
  <table style="width:100%;border-collapse:collapse;font-size:.9rem">
    <thead>
      <tr>
        <th style="color:var(--pp-muted);font-weight:600;padding:.4rem .8rem;
                   text-align:left;width:22%">Feature</th>
        <th style="color:var(--pp-green-text);font-weight:700;padding:.4rem .8rem;
                   text-align:center;width:39%">🌱 Normal Mode</th>
        <th style="color:var(--pp-orange-text);font-weight:700;padding:.4rem .8rem;
                   text-align:center;width:39%">⚙️ Expert Mode</th>
      </tr>
    </thead>
    <tbody style="color:var(--pp-text)">
      <tr style="border-top:1px solid var(--pp-border)">
        <td style="padding:.45rem .8rem;color:var(--pp-muted)">Who is it for?</td>
        <td style="padding:.45rem .8rem;text-align:center">Farmers, agronomists, general users</td>
        <td style="padding:.45rem .8rem;text-align:center">Researchers, engineers with on-site sensors</td>
      </tr>
      <tr style="border-top:1px solid var(--pp-border)">
        <td style="padding:.45rem .8rem;color:var(--pp-muted)">Inputs required</td>
        <td style="padding:.45rem .8rem;text-align:center">Location + Date/Time <em>(3 fields)</em></td>
        <td style="padding:.45rem .8rem;text-align:center">All sensor readings manually <em>(17+ fields)</em></td>
      </tr>
      <tr style="border-top:1px solid var(--pp-border)">
        <td style="padding:.45rem .8rem;color:var(--pp-muted)">Weather data</td>
        <td style="padding:.45rem .8rem;text-align:center">Auto-fetched via Open-Meteo API</td>
        <td style="padding:.45rem .8rem;text-align:center">You enter your own sensor values</td>
      </tr>
      <tr style="border-top:1px solid var(--pp-border)">
        <td style="padding:.45rem .8rem;color:var(--pp-muted)">Accuracy</td>
        <td style="padding:.45rem .8rem;text-align:center">Good (API weather ~hourly resolution)</td>
        <td style="padding:.45rem .8rem;text-align:center">Maximum (real on-site measurements)</td>
      </tr>
      <tr style="border-top:1px solid var(--pp-border)">
        <td style="padding:.45rem .8rem;color:var(--pp-muted)">Results shown</td>
        <td style="padding:.45rem .8rem;text-align:center">PAR gauge · DLI · Forecast chart · Crop advice</td>
        <td style="padding:.45rem .8rem;text-align:center">All of Normal + feature importance · McCree comparison · full feature table</td>
      </tr>
    </tbody>
  </table>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  STATS STRIP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="stats-bar">
  <div class="stat-item">
    <div class="stat-val">R²&nbsp;0.99</div>
    <div class="stat-lbl">Model Accuracy</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">15</div>
    <div class="stat-lbl">Model Features</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">1 min</div>
    <div class="stat-lbl">Native Resolution</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">2</div>
    <div class="stat-lbl">Monitoring Sites</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">190 K</div>
    <div class="stat-lbl">Training Rows (1-min)</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">Global</div>
    <div class="stat-lbl">Coverage via API</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  WHAT IS PAR?
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("📚  What is PAR and why does it matter for Agrivoltaics?"):
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("""
**PAR (Photosynthetically Active Radiation)**  
is the portion of sunlight in the 400–700 nm wavelength range that plants
use for photosynthesis.  It is measured in µmol/m²/s (quantum flux).

**Agrivoltaics (AgriPV)** combines solar energy production with agriculture
on the same land.  Solar panels intercept part of the incoming irradiance
(GHI), and the remaining PAR reaching the crops determines their growth,
water consumption and yield.

**The Problem**  
The classic McCree formula — `PAR ≈ 0.45 × GHI` — assumes a fixed spectral
ratio and ignores clouds, rain, solar angle and atmospheric conditions.
This leads to errors of up to **35 % nRMSE** in real deployments.
        """)
    with right:
        st.markdown("""
**This tool uses machine learning** trained on real 1-second sensor data from
agrivoltaic monitoring stations in Germany (Laubsdorf & Nebelin, 2024–2025).

| Method | R² | nRMSE |
|---|---|---|
| McCree (0.45 × GHI) | 0.75 | 35 % |
| Linear regression | 0.85 | 25 % |
| Neural Network (prototype) | 0.96 | 15.6 % |
| **This ML model (XGBoost)** | **0.99** | **~8 %** |

The model learns non-linear interactions between GHI, solar position,
humidity, precipitation and temperature to predict PAR accurately —
for **any location worldwide**, and for **any date** from 1940 to 15 days
ahead, via the Open-Meteo weather API.
        """)

# ═══════════════════════════════════════════════════════════════════════════════
#  HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### How it works")
steps = [
    ("📍", "Location",  "Enter a city name or latitude / longitude coordinates."),
    ("🌤️", "Weather",  "Open-Meteo supplies hourly GHI, temperature, humidity, wind and rain — 1940 to 15 days ahead."),
    ("☀️", "Solar Geometry", "pvlib computes zenith angle, airmass, clearness index and DNI — the same physics used in training."),
    ("🤖", "Predict",   "The XGBoost model infers PAR from all 22 features in milliseconds."),
    ("🌱", "Act",       "Use the PAR estimate and DLI forecast for irrigation scheduling, crop monitoring and yield forecasting."),
]
cols = st.columns(len(steps), gap="small")
for col, (icon, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-icon">{icon}</div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER  (fixed at page bottom)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-wrap">
    ParPredict &nbsp;·&nbsp; Hochschule Anhalt &nbsp;·&nbsp;
    Data Science Master Program 2026 &nbsp;&nbsp;·&nbsp;&nbsp;
    <strong>Creator:</strong> Mahdi Bey
</div>
""", unsafe_allow_html=True)
