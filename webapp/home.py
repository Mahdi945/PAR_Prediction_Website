"""
home.py  –  ParPredict · Home / Landing Page
"""

import base64
import streamlit as st
from pathlib import Path
from core.predict import is_model_available
from core.i18n import t, t_block

# ── Load logo SVG ────────────────────────────────────────────────────────────
_logo_path = Path(__file__).parent / "assets" / "logo.svg"
_logo = _logo_path.read_text(encoding="utf-8")
_logo_data = base64.b64encode(_logo.encode("utf-8")).decode("ascii")
_logo_img = f'<img src="data:image/svg+xml;base64,{_logo_data}" alt="ParPredict logo" width="170" height="170" style="display:block;margin:0 auto;" />'

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default Streamlit header padding */
.block-container { padding-top: 1.5rem; }

/* Hero */
.hero-title {
    font-size: 3.2rem; font-weight: 900; color: #ffffff;
    margin: 0; line-height: 1.1;
}
.hero-title span { color: #2ecc71; }
.hero-sub {
    font-size: 1.1rem; color: #8892b0;
    margin: 0.6rem 0 0.2rem 0; line-height: 1.6;
}

/* Mode cards */
.mode-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #0f1117 100%);
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
.mode-card-green  { border: 2px solid #2ecc71; }
.mode-card-orange { border: 2px solid #f39c12; }
.mode-card:hover  { transform: translateY(-3px);
                    box-shadow: 0 12px 32px rgba(46,204,113,.25); }
a.mode-link       { text-decoration: none !important; display: block; width: 100%; height: 100%; }
a.mode-link:hover .mode-card-green  { box-shadow: 0 12px 32px rgba(46,204,113,.35); transform: translateY(-3px); }
a.mode-link:hover .mode-card-orange { box-shadow: 0 12px 32px rgba(243,156,18,.35); transform: translateY(-3px); }
.mode-icon  { font-size: 3.2rem; margin-bottom: .8rem; flex-shrink: 0; }
.mode-title { font-size: 1.5rem; font-weight: 800; color: #fff;
              margin: .4rem 0; flex-shrink: 0; }
.mode-desc  { color: #8892b0; font-size: .9rem; line-height: 1.65;
              flex: 1; overflow-wrap: anywhere; }

/* Force Streamlit columns that hold mode cards to equal height */
div[data-testid="column"] > div:first-child {
    height: 100%;
}

/* Stats strip */
.stats-bar {
    display: flex; justify-content: space-around;
    background: #1a1d2e; border: 1px solid #2a2d3e;
    border-radius: 14px; padding: 1.4rem 1rem; margin: 2rem 0;
}
.stat-item { text-align: center; }
.stat-val { font-size: 1.9rem; font-weight: 900; color: #2ecc71; }
.stat-lbl { font-size: .75rem; color: #8892b0;
            text-transform: uppercase; letter-spacing: 1.5px; }

/* How-it-works steps */
.step-card {
    background: #1a1d2e; border: 1px solid #2a2d3e;
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
.step-title { font-size: 1rem; font-weight: 700; color: #fff;
              margin: .5rem 0 .3rem 0; flex-shrink: 0; }
.step-desc  { color: #8892b0; font-size: .78rem; line-height: 1.5;
              flex-shrink: 0; overflow: hidden; }

/* Warning / info banner */
.model-warn {
    background: #2d1a00; border: 1px solid #f39c12;
    border-radius: 10px; padding: .9rem 1.2rem;
    color: #f7c948; font-size: .88rem;
}

/* Footer — stick to page bottom */
.footer-wrap {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: #0f1117;
    border-top: 1px solid #1a1d2e;
    text-align: center; color: #5a6070;
    font-size: .76rem; padding: .6rem 1rem;
    z-index: 999;
}

@media (max-width: 900px) {
    .hero-title { font-size: 2.2rem; }
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
    .hero-title { font-size: 1.85rem; }
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
    st.markdown(
        f'<div style="text-align:center;margin-bottom:.5rem">{_logo_img}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"""
    <div style="text-align:center">
        <h1 class="hero-title">Par<span>Predict</span></h1>
        <p class="hero-sub">
            {t("Nowcasting Photosynthetically Active Radiation for Agrivoltaic Systems")}<br>
            <small style="color:#5a6070">
                {t("Powered by")} XGBoost &nbsp;·&nbsp; pvlib &nbsp;·&nbsp;
                Open-Meteo &nbsp;·&nbsp; Hochschule Anhalt 2026
            </small>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL STATUS BANNER
# ═══════════════════════════════════════════════════════════════════════════════
if not is_model_available():
    st.markdown(f"""
    <div class="model-warn">
        ⚠️ <strong>{t("Model file not found.")}</strong>
        {t("The XGBoost model is stored in Git LFS. Run")}
        <code>git lfs pull</code> {t("from the project root to download it, then refresh this page.")}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODE SELECTION CARDS  (entire card is clickable — no separate button)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"### {t('Choose your mode')}")
col_a, col_b, col_c = st.columns(3, gap="large")

with col_a:
    st.markdown(f"""
    <a href="Normal_Mode" target="_self" class="mode-link">
      <div class="mode-card mode-card-green">
        <div class="mode-icon">🌱</div>
        <div class="mode-title">{t("Normal Mode")}</div>
        <div class="mode-desc">
            {t('Search for any city or enter coordinates.<br>Weather is fetched <em>automatically</em> from Open-Meteo — any date from 1940 to 15 days ahead.')}
        </div>
        <div style="margin-top:1rem">
          <strong style="color:#2ecc71">{t("3 inputs &nbsp;·&nbsp; One prediction")}</strong>
        </div>
      </div>
    </a>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <a href="Expert_Mode" target="_self" class="mode-link">
      <div class="mode-card mode-card-orange">
        <div class="mode-icon">⚙️</div>
        <div class="mode-title">{t("Expert Mode")}</div>
        <div style="margin:.3rem 0 .7rem 0">
          <strong style="color:#f39c12">{t("Full control &nbsp;·&nbsp; Full transparency")}</strong>
        </div>
        <div class="mode-desc">
            {t('Enter your own sensor readings for maximum accuracy.<br>Inspect feature importance, McCree comparison,<br>and every intermediate computed value.')}
        </div>
      </div>
    </a>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown(f"""
    <a href="Dataset_Upload" target="_self" class="mode-link">
      <div class="mode-card" style="border: 2px solid #3498db;">
        <div class="mode-icon">📊</div>
        <div class="mode-title">{t("Dataset Upload")}</div>
        <div style="margin:.3rem 0 .7rem 0">
          <strong style="color:#3498db">{t("Batch analysis &nbsp;·&nbsp; Full benchmark")}</strong>
        </div>
        <div class="mode-desc">
            {t('Upload a full time-series dataset with sensor readings and location.<br>Clean it, resample it, and compare model predictions to a baseline.')}
        </div>
      </div>
    </a>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODE COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="background:#1a1d2e;border:1px solid #2a2d3e;border-radius:14px;
            padding:1.2rem 1.6rem;margin-bottom:1.4rem">
  <table style="width:100%;border-collapse:collapse;font-size:.9rem">
    <thead>
      <tr>
        <th style="color:#8892b0;font-weight:600;padding:.4rem .8rem;
                   text-align:left;width:22%">{t("Feature")}</th>
        <th style="color:#2ecc71;font-weight:700;padding:.4rem .8rem;
                   text-align:center;width:39%">🌱 {t("Normal Mode")}</th>
        <th style="color:#f39c12;font-weight:700;padding:.4rem .8rem;
                   text-align:center;width:39%">⚙️ {t("Expert Mode")}</th>
      </tr>
    </thead>
    <tbody style="color:#e8eaf6">
      <tr style="border-top:1px solid #2a2d3e">
        <td style="padding:.45rem .8rem;color:#8892b0">{t("Who is it for?")}</td>
        <td style="padding:.45rem .8rem;text-align:center">{t("Farmers, agronomists, general users")}</td>
        <td style="padding:.45rem .8rem;text-align:center">{t("Researchers, engineers with on-site sensors")}</td>
      </tr>
      <tr style="border-top:1px solid #2a2d3e">
        <td style="padding:.45rem .8rem;color:#8892b0">{t("Inputs required")}</td>
        <td style="padding:.45rem .8rem;text-align:center">{t('Location + Date/Time <em>(3 fields)</em>')}</td>
        <td style="padding:.45rem .8rem;text-align:center">{t('All sensor readings manually <em>(17+ fields)</em>')}</td>
      </tr>
      <tr style="border-top:1px solid #2a2d3e">
        <td style="padding:.45rem .8rem;color:#8892b0">{t("Weather data")}</td>
        <td style="padding:.45rem .8rem;text-align:center">{t("Auto-fetched via Open-Meteo API")}</td>
        <td style="padding:.45rem .8rem;text-align:center">{t("You enter your own sensor values")}</td>
      </tr>
      <tr style="border-top:1px solid #2a2d3e">
        <td style="padding:.45rem .8rem;color:#8892b0">{t("Accuracy")}</td>
        <td style="padding:.45rem .8rem;text-align:center">{t("Good (API weather ~hourly resolution)")}</td>
        <td style="padding:.45rem .8rem;text-align:center">{t("Maximum (real on-site measurements)")}</td>
      </tr>
      <tr style="border-top:1px solid #2a2d3e">
        <td style="padding:.45rem .8rem;color:#8892b0">{t("Results shown")}</td>
        <td style="padding:.45rem .8rem;text-align:center">{t("PAR gauge · DLI · Forecast chart · Crop advice")}</td>
        <td style="padding:.45rem .8rem;text-align:center">{t("All of Normal + feature importance · McCree comparison · full feature table")}</td>
      </tr>
    </tbody>
  </table>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  STATS STRIP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="stats-bar">
  <div class="stat-item">
    <div class="stat-val">R²&nbsp;0.99</div>
    <div class="stat-lbl">{t("Model Accuracy")}</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">15</div>
    <div class="stat-lbl">{t("Model Features")}</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">{t("1 min")}</div>
    <div class="stat-lbl">{t("Native Resolution")}</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">4</div>
    <div class="stat-lbl">{t("Monitoring Sites")}</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">~43M</div>
    <div class="stat-lbl">{t("Training Rows")}</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">{t("Global")}</div>
    <div class="stat-lbl">{t("Coverage via API")}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  WHAT IS PAR?
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander(t("📚  What is PAR and why does it matter for Agrivoltaics?")):
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(t_block("home.par_left", """
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
        """))
    with right:
        st.markdown(t_block("home.par_right", """
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
        """))

# ═══════════════════════════════════════════════════════════════════════════════
#  HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"### {t('How it works')}")
steps = ["Location", "Weather", "Solar Geometry", "Predict", "Act"]
icons = {"Location": "📍", "Weather": "🌤️", "Solar Geometry": "☀️", "Predict": "🤖", "Act": "🌱"}
cols = st.columns(len(steps), gap="small")
for col, step in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-icon">{icons[step]}</div>
            <div class="step-title">{t(f"step.{step}")}</div>
            <div class="step-desc">{t(f"step.{step}.desc")}</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER  (fixed at page bottom)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer-wrap">
    ParPredict &nbsp;·&nbsp; Hochschule Anhalt &nbsp;·&nbsp;
    Data Science Master Program 2026 &nbsp;&nbsp;·&nbsp;&nbsp;
    <strong>{t("Developers:")}</strong>
    Tristan Kühn &nbsp;·&nbsp; Ethan Miska &nbsp;·&nbsp;
    Mehdi Bey &nbsp;&nbsp;·&nbsp;&nbsp; <em>{t("Supervisor:")} Hugo Sanchez</em>
</div>
""", unsafe_allow_html=True)
