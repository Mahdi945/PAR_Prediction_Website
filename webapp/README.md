# PAR Predictor — Web Application

> **Nowcasting Photosynthetically Active Radiation for Agrivoltaic Systems**  
> Hochschule Anhalt · Data Science Master Program 2026  
> Team: Tristan Kühn · Ethan Miska · Mehdi Bey · Supervisor: Hugo Sanchez

---

## Overview

PAR Predictor is a Streamlit web application that predicts **PAR (Photosynthetically Active Radiation)** for any location worldwide using a trained XGBoost model.  It replaces the inaccurate McCree linear formula with a machine-learning pipeline that accounts for solar geometry, atmospheric conditions and weather — reducing prediction error from 35 % nRMSE down to ~8 %.

| Mode | Description |
|---|---|
| **Normal Mode** | Enter city/coordinates + time → weather auto-fetched → instant PAR prediction |
| **Expert Mode** | Full manual sensor input OR editable API values → PAR + McCree comparison + feature importance |

---

## Architecture

```
webapp/
├── app.py                     ← Home / landing page
├── pages/
│   ├── 1_Normal_Mode.py       ← Simple 3-input interface
│   └── 2_Expert_Mode.py       ← Full expert dashboard (API + Manual tabs)
├── core/
│   ├── __init__.py
│   ├── weather.py             ← Open-Meteo API client + geocoding
│   ├── features.py            ← Feature engineering pipeline (mirrors training notebooks)
│   └── predict.py             ← Model loading, PAR inference, McCree baseline
├── assets/
│   └── logo.svg               ← SVG logo (sun + plant + solar panel)
├── .streamlit/
│   └── config.toml            ← Dark theme configuration
└── requirements_web.txt       ← Python dependencies
```

**External dependencies at runtime:**

| Service | Purpose | Cost |
|---|---|---|
| [Open-Meteo](https://open-meteo.com) | Real-time weather (GHI, temp, humidity, wind, precipitation) | Free, no API key |
| [Open-Meteo Geocoding](https://open-meteo.com/en/docs/geocoding-api) | City name → lat/lon | Free, no API key |
| [pvlib](https://pvlib-python.readthedocs.io) | Solar geometry (zenith, airmass, clearness index, DNI) | Local library |

---

## Quick Start

### 1. Prerequisites

```bash
# From the project root — pull model files from Git LFS
git lfs pull
```

### 2. Install dependencies

```bash
cd webapp
pip install -r requirements_web.txt
```

### 3. Run locally

```bash
# from the webapp/ directory
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Feature Engineering

The feature pipeline in `core/features.py` exactly mirrors the training notebook `4_4_feature_engineering_All_Files.ipynb`.  For each prediction, 22 features are computed:

| Group | Features | Source |
|---|---|---|
| Raw sensor / weather (11) | `GHI_RC_01`, `Temp_WS`, `RH_WS`, `DWP_WS`, `WS_WS`, `WD_WS`, `PREC_INT_WS`, `PREC_DIFF_WS`, `PREC_WS`, `Temp_RC_merged`, `Temp_RC_01` | Open-Meteo API (or manual) |
| pvlib solar geometry (5) | `zenith`, `elevation`, `airmass`, `clearness_kt`, `dni` | pvlib (computed from lat/lon/time) |
| Wind cyclical (2) | `wind_sin`, `wind_cos` | Derived from `WD_WS` |
| Engineered (4) | `is_raining`, `GHI_rolling_5min`, `temp_diff`, `dew_depression` | Computed from above |

> **Note on `Temp_RC_merged`:** The reference cell temperature is not available from the weather API.  It is approximated using the NOCT model:  
> `Tc = Ta + (NOCT − 20) / 800 × GHI`  (NOCT = 45 °C for typical c-Si cells)

---

## Model

- **Type:** XGBoost Regressor (trained in `6_6_gradient_boosting_All_Files.ipynb`)
- **Target:** `PAR_PAR` [µmol/m²/s]
- **Training data:** ~300 k rows at 1-minute resolution, 2 German agrivoltaic sites (Laubsdorf + Nebelin, 2024–2025)
- **Performance:** R² ≈ 0.99, nRMSE ≈ 8 % (vs. McCree baseline: nRMSE ≈ 35 %)
- **Files (Git LFS):**
  - `data/results/xgboost_model_all_locations.pkl`
  - `data/processed/pkl_features_GradientBoosting/feature_names_all_locations.pkl`

---

## Global Capability

The app works worldwide because:

- **pvlib** computes solar position for any lat/lon/altitude on Earth.
- **Open-Meteo** provides free weather data for any global location.
- The feature engineering is fully location-agnostic.

**Limitation:** The model was trained on temperate-climate German data.  Predictions for tropical, desert or polar climates may be less accurate until the model is retrained with data from those regions.

---

## Deployment Options

### Streamlit Community Cloud (free)

1. Push the project to GitHub (ensure model files are in Git LFS).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app.
3. Point to `webapp/app.py`.
4. Set `requirements.txt` to `webapp/requirements_web.txt`.

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY webapp/requirements_web.txt .
RUN pip install -r requirements_web.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "webapp/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t par-predictor .
docker run -p 8501:8501 par-predictor
```

---

## McCree Baseline Formula

The classical linear approximation used for comparison:

```
PAR [µmol/m²/s] = GHI [W/m²]  ×  0.45  (PAR energy fraction)
                              ×  4.57  (µmol/J conversion for solar spectrum)
                = GHI × 2.06
```

The ML model significantly outperforms this estimate under cloudy, rainy or low-sun-angle conditions.

---

## Extending the App

- **Add a new site:** Update `LOCATION_COORDS` in `core/features.py` and retrain with new data.
- **Retrain model:** Run the Jupyter notebooks in `notebooks/Mahdi/` and replace the `.pkl` file.
- **Add feedback loop:** Store user-submitted actual PAR measurements in a SQLite database and trigger periodic retraining.
- **Add more crops:** Extend the `crop_advice()` function in `pages/1_Normal_Mode.py`.
