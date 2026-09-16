# PAR Predictor – Team Web App Documentation

## Overview

This project is a Streamlit web application that predicts PAR (Photosynthetically Active Radiation) for agrivoltaic and solar applications. The app takes a location and time, fetches weather data from a public API, transforms the data into the same feature format used during training, and runs a trained XGBoost model to estimate PAR in real time.

The main goal is to provide a simple interface for end users while keeping the technical flow transparent for developers and researchers.

---

## What the website does

The app includes:

- a simple prediction mode for quick usage
- an expert mode for manual entries and diagnostics
- a dataset upload view for analysis and validation
- automatic Open-Meteo weather fetch based on location/time
- a final PAR prediction using a trained XGBoost regressor
- a comparison with the classical McCree formula

This makes the application useful both for end users and for team members testing model behavior and feature engineering.

---

## Architecture

### Frontend
The frontend is built with Streamlit.

- `webapp/app.py` is the main navigation entry point
- `webapp/home.py` is the landing page
- `webapp/pages/1_Normal_Mode.py` is the quick prediction page
- `webapp/pages/2_Expert_Mode.py` is the advanced analysis page
- `webapp/pages/3_Dataset_Upload.py` is the dataset inspection page

### Backend and logic
The project separates logic into modules under `webapp/core/`.

- `weather.py` — fetches weather and geocoding data from Open-Meteo
- `features.py` — prepares model-ready features from raw inputs
- `predict.py` — loads the trained model and runs prediction logic

### Model and data assets
The model is stored in the project data folder, especially:

- `data/results/xgboost_model_all_locations.pkl`
- `data/processed/pkl_features_GradientBoosting/feature_names_all_locations.pkl`

The model file is large, so it is managed with Git LFS.

---

## Technology stack

| Area | Technology | Role |
|---|---|---|
| Web UI | Streamlit | Interactive multi-page app |
| ML | XGBoost | PAR prediction model |
| Data handling | Pandas, NumPy | Feature arrays and transformations |
| Solar calculations | pvlib | Solar geometry and irradiance-derived features |
| Weather API | Open-Meteo | Live weather and geocoding data |
| HTTP requests | requests | Calls to external APIs |
| Model serialization | joblib | Loads the trained XGBoost model |
| Visualization | Plotly, Streamlit charts | Charts, gauges, diagnostics |

---

## How the app works end-to-end

The full flow is:

1. User enters a city, coordinates, or weather values.
2. The app optionally calls Open-Meteo to fetch weather data.
3. The app computes engineered features such as solar elevation, airmass, clearness index, and wind-related variables.
4. These features are passed to `predict_par()`.
5. The `predict.py` module loads the XGBoost model from disk.
6. The model returns a PAR prediction in µmol/m²/s.
7. The result is shown in the dashboard and compared with a McCree baseline calculation.

---

## How the website connects to APIs

The main API integration is in `webapp/core/weather.py`.

### Weather API
The app calls Open-Meteo forecast data using latitude, longitude, and timestamp.

It requests values like:

- temperature
- humidity
- dew point
- wind speed
- wind direction
- precipitation
- shortwave radiation

The API response is matched to the closest hourly timestamp and converted into the variables expected by the model.

### Geocoding API
The app can also query the Open-Meteo geocoding service to convert a city name into latitude/longitude so the user does not need to enter coordinates manually.

This API layer is essential because the prediction model relies on meteorological and solar conditions tied to the selected place and time.

---

## How it connects to the XGBoost model

The model integration is in `webapp/core/predict.py`.

This module does several important things:

- defines compatibility logic for joblib-loaded notebook objects
- loads the trained XGBoost model from the project data folder
- loads the expected feature names used during training
- builds a DataFrame in the correct feature order
- runs the model prediction
- returns a PAR value as a float

The model is trained on engineered features, so the app must match the same feature order and names exactly. If anything changes in the feature pipeline, the prediction can become invalid.

---

## Feature engineering

Before the model predicts, the app computes features that reflect both weather conditions and solar geometry. These features include:

- zenith and elevation
- airmass
- clearness index
- wind sine/cosine transformations
- rolling radiation values
- temperature difference metrics
- dew depression
- rain indicator

This logic mirrors the training process, which is crucial because the model expects the same variables it learned from.

---

## Deployment on Streamlit Cloud

The app is designed to be deployed on Streamlit Community Cloud.

### Deployment steps

1. Push the repository to GitHub.
2. Enable Git LFS for the repository.
3. In Streamlit Cloud, create a new app.
4. Set the app file to `webapp/app.py`.
5. Set the requirements file to `webapp/requirements_web.txt`.
6. Make sure the model file is available through Git LFS.

### Local run

```bash
cd webapp
pip install -r requirements_web.txt
streamlit run app.py
```

If the model is missing locally, the app warns the user to run:

```bash
git lfs pull
```

---

## Why this architecture matters

This app is structured so that each part has one clear responsibility:

- Streamlit handles the user interface
- Open-Meteo provides the weather context
- feature engineering prepares model-ready input
- the XGBoost model produces the actual PAR prediction
- the dashboard shows results and comparisons

This separation makes the solution easier to maintain, debug, and deploy.

---

## Team notes

- The model should be retrained and replaced if the underlying dataset changes.
- Updating feature engineering requires checking the training notebooks and the app pipeline together.
- Git LFS must remain configured for the model artifact.
- The app depends on external weather data, so internet access is required at runtime.

---

## Summary

The website is a full ML-powered prediction system built around a Streamlit interface, live weather API access, custom feature engineering, and an XGBoost model. The app connects the user, the weather source, and the model into one workflow that produces a PAR estimate in real time.

This is the architecture the team should keep in mind when developing, maintaining, and deploying the project.
