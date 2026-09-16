# PAR Predictor Web App – Team Documentation

## Introduction

This project is a Streamlit-based web application for predicting PAR (Photosynthetically Active Radiation), which is the part of solar radiation used by plants for photosynthesis. The goal is to provide a lightweight and user-friendly app that predicts PAR from weather and location conditions using a trained XGBoost machine learning model.

The application is built for agrivoltaic and solar research use cases. It allows a user to enter a location and time, fetch meteorological data automatically from an external API, transform the data into the same engineering features used in training, and predict PAR in real time.

---

## What we built

The website is not just a visual dashboard. It is a full prediction pipeline composed of:

- a user interface built with Streamlit
- an API layer that fetches weather and geolocation data
- a feature engineering pipeline that converts raw variables into model-ready inputs
- an XGBoost regressor that predicts PAR
- a comparison layer with the classical McCree baseline formula
- a dashboard for expert analysis and model interpretation

This means the web app sits between the user and the ML model, while also bringing in external live weather data.

---

## Project structure

```text
Project data science/
├── data/
│   ├── processed/
│   └── results/
│       └── xgboost_model_all_locations.pkl
├── notebooks/
│   └── Mahdi/
├── webapp/
│   ├── app.py
│   ├── home.py
│   ├── README.md
│   ├── requirements_web.txt
│   ├── pages/
│   │   ├── 1_Normal_Mode.py
│   │   ├── 2_Expert_Mode.py
│   │   └── 3_Dataset_Upload.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── weather.py
│   │   ├── features.py
│   │   └── predict.py
│   └── assets/
├── README.md
├── readmeapp.md
└── requirements.txt
```

### Main files

- `webapp/app.py`: Streamlit entry point and navigation organizer
- `webapp/home.py`: landing page with app description and model status
- `webapp/pages/1_Normal_Mode.py`: simple end-user mode for quick prediction
- `webapp/pages/2_Expert_Mode.py`: expert mode for manual input and model details
- `webapp/pages/3_Dataset_Upload.py`: dataset upload and inspection page
- `webapp/core/weather.py`: Open-Meteo API client
- `webapp/core/features.py`: transforms raw inputs into engineered features
- `webapp/core/predict.py`: loads the XGBoost model and computes PAR predictions

---

## Technology stack

The website uses the following technologies:

| Component | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Multi-page interactive web app |
| ML model | XGBoost | PAR prediction model |
| Data processing | Pandas, NumPy | Data handling and feature engineering |
| Solar calculations | pvlib | Solar geometry and irradiation-related variables |
| API calls | requests | HTTP requests to weather services |
| Model persistence | joblib | Loading the trained model and metadata |
| Visualization | Plotly / Streamlit charts | Gauges, bar plots, diagnostic visuals |
| Weather data source | Open-Meteo API | Real-time or forecast weather data |

The app is intentionally simple for deployment and for colleagues to use without needing a heavy backend framework.

---

## How the web app is connected to the model

The model is not embedded directly in the UI. Instead, the app follows a clean pipeline:

1. The user enters coordinates, time, and optionally weather values.
2. The UI calls the weather API if needed.
3. The app computes engineered features using the same logic as the training notebook.
4. The feature values are passed to `predict_par()`.
5. `predict_par()` loads the trained XGBoost model from disk.
6. The model returns a PAR estimate in µmol/m²/s.

The key model-loading logic is in `webapp/core/predict.py`:

- `load_model()` loads the saved model from `data/results/xgboost_model_all_locations.pkl`
- `predict_par(features_df)` builds the input matrix in the right feature order
- The model is then used to generate a prediction

There is also a robustness step to handle custom classes saved from a notebook environment. This is important because the training model was created in a notebook and serialized with joblib. The app includes a compatibility layer so the object can be loaded correctly in a fresh Python environment.

---

## How the app is linked to APIs

The app uses the Open-Meteo API for both weather and geocoding.

### Weather API

File: `webapp/core/weather.py`

The function `fetch_weather(lat, lon, dt)` does the following:

- sends a request to the Open-Meteo forecast API
- asks for hourly weather variables such as:
  - temperature
  - relative humidity
  - dew point
  - wind speed and direction
  - precipitation
  - shortwave radiation
- matches the requested timestamp to the nearest hour in the API response
- returns a dictionary with the variables needed by the model

### Geocoding API

The app can also translate a city name into coordinates using Open-Meteo geocoding. This makes the interface more user-friendly: users can search a place, and the app maps it to latitude and longitude before fetching weather.

This external API dependency is a key part of the app because the model needs location and time information to compute solar geometry and estimate PAR.

---

## How features are generated

The model does not work directly on raw weather values. The app first transforms them into engineered features that match the training data format.

This is handled in `webapp/core/features.py` and includes things such as:

- solar zenith and elevation
- air mass
- clearness index
- wind sine and cosine values
- rolling radiation values
- temperature difference variables
- dew depression
- rain flag

These features are necessary because the model has been trained on a specific feature set. If the app sends the wrong feature names or order, the prediction would be invalid.

---

## Model details

The main model is an XGBoost regressor trained on agrivoltaic data from several locations.

Important files:

- `data/results/xgboost_model_all_locations.pkl` — trained model file
- `data/processed/pkl_features_GradientBoosting/feature_names_all_locations.pkl` — expected feature names

The model is stored in Git LFS because it is too large for normal Git tracking. This is important for local setup and deployment.

The model output is PAR in µmol/m²/s, and the app also compares this value with the classical McCree estimate, which is a simpler formula-based approximation.

---

## User modes in the app

### Normal Mode
This is the simplified interface.

- user enters city or coordinates
- weather is fetched automatically
- features are generated
- prediction is computed and returned instantly

### Expert Mode
This is the advanced interface.

- user can manually edit weather values
- compare model prediction vs McCree baseline
- inspect feature importance
- see intermediate variables and diagnostics

### Dataset Upload Mode
This page allows data upload and exploration for teams working on model validation and feature analysis.

---

## Deployment on Streamlit Cloud

The app is intended to run on Streamlit Community Cloud. The deployment logic is simple because the project is already structured as a Streamlit app.

### Recommended deployment setup

- GitHub repository: push the project to GitHub
- Main app file: `webapp/app.py`
- Requirements file: `webapp/requirements_web.txt`
- Make sure Git LFS is enabled for the model file and any large data artifacts

### Important deployment notes

1. The app depends on the model file stored in `data/results/`.
2. That file must be available in the cloud deployment environment.
3. Git LFS must be configured in the repository so the model downloads correctly on deployment.
4. The Streamlit Cloud app should point to the correct working directory and requirements file.

### Typical commands for local execution

```bash
cd webapp
pip install -r requirements_web.txt
streamlit run app.py
```

If the model file is missing, the app will show a warning and ask the user to run:

```bash
git lfs pull
```

---

## End-to-end data flow

```text
User input
   ↓
Streamlit UI (Normal/Expert mode)
   ↓
Open-Meteo API / geocoding
   ↓
Weather values + location + time
   ↓
Feature engineering (pvlib + custom variables)
   ↓
XGBoost model prediction
   ↓
PAR result + comparison + charts
```

This shows clearly how the website, the API, and the model work together.

---

## Why this architecture is useful

This design is easy to maintain and easy for the team to extend:

- the frontend is separated from the modeling logic
- the API layer is isolated in one file
- the model is loaded in a dedicated prediction module
- feature processing mirrors the training pipeline
- deployment is straightforward because Streamlit handles the app interface

For teammates, this means the code is organized by responsibility instead of being mixed into one monolithic script.

---

## Important reminders for teammates

- Do not modify the model input feature order without updating the feature pipeline.
- Keep `core/features.py` and the training notebooks aligned.
- If the model is retrained, update the `.pkl` file and ensure Git LFS tracks it.
- The app is designed for live weather data; it is not a fully offline model pipeline.
- Streamlit Cloud deployment depends on proper dependency installation and LFS support.

---

## Summary

The web app is a full ML-based prediction product built around a real-time weather API and an XGBoost model. The front-end is developed in Streamlit, the weather layer uses Open-Meteo, the processing pipeline turns incoming data into training-compatible features, and the trained model is linked to the UI through `webapp/core/predict.py`.

In simple terms: the website collects the user request, gets weather data from the API, transforms it into model features, runs the XGBoost prediction, and displays the resulting PAR estimate in a clean dashboard.

This is the complete architecture of our current web application and deployment setup.
}
```

### Feature Engineering Process

`core/features.compute_features()` transforms weather into 20+ features:

```python
# Input weather dict → computations → output feature row
{
    # Direct mappings
    "GHI_RC_01": 450.0,
    "Temp_WS": 18.5,
    "RH_WS": 65.2,
    ...
    
    # Calculated solar geometry
    "zenith": 35.2,                # Solar zenith angle [°] (pvlib)
    "airmass": 1.58,               # Airmass (pvlib)
    "clearness_kt": 0.72,          # Clearness index (pvlib)
    
    # Derived/engineering
    "wind_sin": 0.0,               # sin(wind_direction)
    "wind_cos": -1.0,              # cos(wind_direction)
    "is_raining": 0,               # Binary flag if PREC > 0
    "GHI_rolling_5min": 445.0,     # 5-min rolling avg (or current)
    "temp_diff": 10.2,             # Temp - Dewpoint
    "dew_depression": 10.2         # Same as temp_diff
}
```

### Model Inference

```python
# In core/predict.py

1. Load XGBoost model (pickle)
   model = joblib.load("data/results/xgboost_model_all_locations.pkl")
   
2. Load feature names list
   feature_names = joblib.load("data/processed/.../feature_names_all_locations.pkl")
   
3. Reorder input features to match training order
   X = features_df[feature_names].values  # Must be exact order!
   
4. Run inference
   y_pred = model.predict(X)              # Returns array of PAR values
   
5. Extract & return single value
   par_prediction = y_pred[0]             # µmol/m²/s
```

### McCree Baseline (Expert Mode Only)

`core/predict.mccree_estimate()` is a simple physics-based formula:
```python
# Simplified McCree model: PAR = GHI × clearness_index × efficiency_factor
par_mccree = GHI × clearness_kt × 2.04
```
Used to **compare** ML prediction against a physics-based estimate.

---

## 6. Open-Meteo API Integration

### What is Open-Meteo?

- **Free** weather API (no key required)
- Returns hourly forecasts + historical data
- Covers global coordinates
- REST API (JSON responses)

### How It's Called

File: `core/weather.py`

```python
def fetch_weather(lat, lon, datetime):
    """
    Queries Open-Meteo API for weather at given location + time.
    
    Returns dict with keys:
      GHI_RC_01, Temp_WS, RH_WS, DWP_WS, WS_WS, WD_WS, PREC_INT_WS
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date.isoformat(),
        "end_date": date.isoformat(),
        "hourly": "direct_normal_irradiance,temperature,relative_humidity,..."
    }
    response = requests.get(url, params=params)
    # Parse JSON → extract hourly data → interpolate to requested time
    return weather_dict
```

### Geocoding

`core/weather.geocode_city()` converts city name to lat/lon:
```python
# "Berlin" → (52.52, 13.40)
url = "https://geocoding-api.open-meteo.com/v1/search"
params = {"name": "Berlin"}
```

---

## 7. Model Loading & Compatibility

### The XGBEnsemble Class

Why it exists:
- The training notebook created an `XGBEnsemble` class in its `__main__` scope
- When pickled, joblib stores the class reference as `__main__.XGBEnsemble`
- During unpickling, Python needs the class definition to exist
- Solution: Define the class in `core/predict.py` and inject it into `__main__` before unpickling

```python
# In core/predict.py:

class XGBEnsemble:
    """Ensemble of XGBoost models; predictions are averaged."""
    def __init__(self, models=None, **kwargs):
        self.models = models or []
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def predict(self, X):
        preds = np.mean([m.predict(X) for m in self.models], axis=0)
        return np.clip(preds, self.clip_min, self.clip_max)
    
    @property
    def feature_importances_(self):
        return np.mean([m.feature_importances_ for m in self.models], axis=0)

# Inject into __main__ BEFORE loading pickle
import __main__
setattr(__main__, "XGBEnsemble", XGBEnsemble)

# NOW load the model
model = joblib.load("data/results/xgboost_model_all_locations.pkl")
```

### Version Pinning (Critical!)

`webapp/requirements_web.txt` now pins exact versions:
```
xgboost==2.0.3
pandas==2.2.2
numpy==1.26.4
streamlit==1.40.2
```

**Why pinning matters:**
- XGBoost 3.0+ has breaking pickle changes; model trained with 2.0 may not load with 3.0
- pandas 3.0 dropped deprecated methods that older code might use
- Streamlit Cloud rebuilds from `requirements_web.txt` on every redeploy
- Unpinned dependencies = **random version upgrades = random breakage**

---

## 8. Application Deployment

### Local Development

```bash
# Clone repo
git clone <your-repo>
cd "Project data science"

# Pull Git LFS files (model + large data)
git lfs pull

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
cd webapp
pip install -r requirements_web.txt

# Run app
streamlit run app.py
# → Opens http://localhost:8501
```

### Streamlit Community Cloud Deployment

1. **Push code to GitHub** (repo must include `webapp/` folder)

2. **Model file considerations:**
   - If model is in Git LFS, Streamlit Cloud will pull it automatically
   - Or: upload model to external storage + fetch at startup

3. **Configure deployment:**
   - Go to https://share.streamlit.io
   - Connect GitHub repo
   - Set:
     - **Main file path:** `webapp/app.py`
     - **Requirements file:** `webapp/requirements_web.txt`
     - Python version: `3.12` (or latest stable)

4. **Environment variables (if needed):**
   - Add secrets in Streamlit Cloud dashboard (for API keys, etc.)
   - Access via `st.secrets["key_name"]`

### Production Best Practices

- ✅ **Pin exact versions** in `requirements_web.txt` (already done)
- ✅ **Keep model file** in Git LFS or cloud storage (not in repo if too large)
- ✅ **Log errors** from Open-Meteo calls (network can fail)
- ✅ **Cache API responses** to avoid quota limits
- ✅ **Test locally** before pushing (verify model loads, predictions run)

---

## 9. Important Files Reference

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `webapp/app.py` | Entry point, router | Sets page config, imports pages |
| `webapp/home.py` | Landing page | Checks model availability, displays info |
| `webapp/pages/1_Normal_Mode.py` | Simple prediction interface | Handles location input, auto-weather, prediction |
| `webapp/pages/2_Expert_Mode.py` | Advanced interface | Manual inputs, charts, McCree comparison, importance |
| `core/weather.py` | Open-Meteo API client | `fetch_weather()`, `geocode_city()`, `fetch_forecast()` |
| `core/features.py` | Feature engineering | `compute_features()` — transforms weather → ML features |
| `core/predict.py` | Model inference | `is_model_available()`, `load_model()`, `predict_par()`, `mccree_estimate()` |
| `requirements_web.txt` | **Exact dependency versions** | ✅ Pinned to prevent breakage |
| `data/results/xgboost_model_all_locations.pkl` | Trained XGBoost ensemble | Loaded by `predict_par()` |
| `data/processed/.../feature_names_all_locations.pkl` | Feature column order | Ensures correct input ordering to model |

---

## 10. Troubleshooting

### Error: "Model not found"
```
⚠️ Model not found. Run `git lfs pull` from the project root.
```
**Fix:**
```bash
git lfs pull
# Wait for large files to download
streamlit run app.py
```

### Error: "Failed to fetch weather"
```
❌ Open-Meteo fetch failed: <error message>
```
**Cause:** Network issue or API down  
**Fix:** Check internet, retry, or use Expert Mode to input weather manually

### Error: "Pickle incompatibility" / Model loads but prediction fails
**Likely cause:** XGBoost version mismatch  
**Fix:**
```bash
pip install -r requirements_web.txt --force-reinstall
```
Ensures XGBoost 2.0.3 is installed (matches model training version)

### Streamlit session state warnings (non-fatal)
If you see: `DeprecationWarning: st.number_input ... key=...`  
**Fix:** Already handled in code (no action needed)

---

## 11. Summary: App Workflow at a Glance

```
User accesses: http://localhost:8501 (or Streamlit Cloud URL)
       ↓
Home page: Select mode (Normal or Expert)
       ↓
[NORMAL MODE PATH]                    [EXPERT MODE PATH]
├─ Enter location + date/time          ├─ Enter location + date/time
├─ Click "Fetch Weather"               ├─ Manual weather inputs
├─ Auto-calls Open-Meteo API           ├─ Optional: "Auto-fetch"
├─ Computes features                   ├─ Computes features
├─ Loads model                         ├─ Loads model
├─ Predicts PAR                        ├─ Predicts PAR
├─ Shows gauge result                  ├─ Shows gauge + importance chart
└─ (Simple, fast)                      ├─ Shows McCree comparison
                                       └─ Shows feature table
                                          (Advanced, diagnostics)
       ↓
Result displayed: PAR in µmol/m²/s
       ↓
User can export, compare, or adjust inputs & re-predict
```

---

## 12. Key Technologies Recap

| Need | Solution | Version |
|------|----------|---------|
| Web UI | Streamlit | 1.40.2 |
| ML Model | XGBoost | 2.0.3 |
| Data handling | Pandas + NumPy | 2.2.2 / 1.26.4 |
| Solar math | pvlib | 0.11.1 |
| Weather API | Open-Meteo | (free, no auth) |
| Visualization | Plotly | 5.24.1 |
| Serialization | joblib | 1.4.2 |

---

**Last Updated:** 2026-08-19  
**Author:** Mahdi, Ethan, Tristan (Data Science Team)  
**Status:** ✅ Production-ready with pinned dependencies
