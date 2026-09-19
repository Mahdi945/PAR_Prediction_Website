# PAR Predictor — Web Application

> **Nowcasting Photosynthetically Active Radiation for Agrivoltaic Systems**  
> Hochschule Anhalt · Data Science Master Program 2026  
> Team: Tristan Kühn · Ethan Miska · Mehdi Bey · Supervisor: Hugo Sanchez

---

## Overview

PAR Predictor is a Streamlit web application that predicts **PAR (Photosynthetically Active Radiation)** for any location worldwide using a trained XGBoost model.  It replaces the inaccurate McCree linear formula with a machine-learning pipeline that accounts for solar geometry, atmospheric conditions and weather — reducing prediction error from 35 % nRMSE down to ~8 %.

| Mode | Description |
|---|---|
| **Normal Mode** | Enter coordinates + date/time (1940 → today + 15 days) → weather auto-fetched → PAR with its typical error, DLI, day curve, export |
| **Expert Mode** | Full manual sensor input (or auto-filled from Open-Meteo) → PAR + McCree comparison + feature importance, export |
| **Dataset Upload** | Upload a sensor file (CSV/TXT/Excel, ≤ 200 MB) → cleaned like the training data, scored per minute, compared with the baseline → download predictions, cleaned data, feature matrix and a JSON report in CSV / Excel / JSON / Parquet |

Every prediction states its **typical error** (held-out MAE, ± 29 µmol/m²/s) and is checked against the **training domain** — distance to the two Brandenburg stations, and inputs outside the training range — so a visitor knows how far to trust it.

---

## Architecture

```
webapp/
├── app.py                     ← Navigation router (run this)
├── home.py                    ← Landing page
├── pages/
│   ├── 1_Normal_Mode.py       ← Coordinates + date/time → prediction
│   ├── 2_Expert_Mode.py       ← Manual sensor input → prediction vs McCree
│   └── 3_Dataset_Upload.py    ← Whole files → cleaned, scored, exported
├── core/
│   ├── constants.py           ← McCree factor (mirror of src/constants.py)
│   ├── weather.py             ← Open-Meteo client: archive / forecast / geocoding
│   ├── cache.py               ← Streamlit-cached front for weather.py
│   ├── features.py            ← Feature engineering (single row + vectorised batch)
│   ├── predict.py             ← Model loading, training-faithful preprocessing, inference, model card
│   ├── domain.py              ← Training-domain check (distance, out-of-range inputs)
│   ├── export.py              ← CSV / Excel / JSON / Parquet / ZIP downloads
│   └── dataset.py             ← Upload pipeline: detect columns, clean, aggregate, score
├── assets/logo.svg
├── docs/                      ← Two-page presentation (HTML + PDF)
├── .streamlit/config.toml     ← Theme, 200 MB upload limit
└── requirements_web.txt       ← Pinned versions the tests were run with
```

Model files are read from the repository root (`data/results/xgboost_model_all_locations.pkl` and the
three small `data/processed/pkl_features_GradientBoosting/*.pkl` files: feature names, training medians,
clip bounds). `data/results/xgboost_metrics_all_locations.pkl` supplies the model card when present.

**External dependencies at runtime:**

| Service | Purpose | Cost |
|---|---|---|
| [Open-Meteo Forecast](https://open-meteo.com) | Weather for today and the next 15 days (GHI, temp, humidity, wind, precipitation) | Free, no API key |
| [Open-Meteo Archive](https://open-meteo.com/en/docs/historical-weather-api) | ERA5 reanalysis for past dates, back to 1940-01-01 | Free, no API key |
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

The feature pipeline in `core/features.py` mirrors the training notebook `4_feature_engineering_All_Files.ipynb`.  For each prediction, 22 columns are computed; the model consumes the 15 listed in `data/processed/pkl_features_GradientBoosting/feature_names_all_locations.pkl` (the others are shown in the UI only). `core/predict.py` raises if any of the 15 is missing rather than filling it with zero. The columns computed are:

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

- **Type:** XGBoost Regressor (trained in `6_model_training_All_Files.ipynb`)
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

The model is also **time-agnostic** — it maps GHI + weather + solar geometry to
PAR for any instant, so the covered period is set purely by the weather API:

| Requested date | Source | Coverage |
|---|---|---|
| Before today | Open-Meteo archive (ERA5 reanalysis) | 1940-01-01 → yesterday |
| Today and later | Open-Meteo forecast model | today → today + 15 days |

Dates outside that window are refused with an explicit message rather than
answered from a nearby date. Every result states which source produced it, and
Expert Mode still accepts manually entered readings for any date at all.

**Limitations:**

- The model was trained on temperate-climate German data (two stations in Brandenburg, April 2024 – May 2025). Predictions elsewhere are extrapolations — the app measures the distance to the nearest training station and flags any input outside the training range, but it cannot make the model know a climate it never saw.
- ERA5 is a modelled, gridded reanalysis (~25 km), not a station measurement — historical predictions inherit its spatial smoothing.
- Forecast accuracy degrades with the horizon; a +15-day GHI is a weather forecast, not a measurement.

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
