# 📖 Web App Concept — How the Website Works

*A guided tour of the `webapp/` folder for someone who has never seen it before.*

This document explains **what every file does**, **how they connect**, and **how to use the website**. No prior knowledge is assumed — not of the project, and not of Streamlit.

> The trained model and the pipeline that produced it are explained in the root [`PROJECT_CONCEPT.md`](../PROJECT_CONCEPT.md). This document covers only the application that *uses* that model.

---

## 1. The one-sentence idea

> The notebooks produced a model that converts **GHI** (cheap, measured everywhere) into **PAR** (expensive, measured almost nowhere). This website is the door to that model: you give it a place and a time, it fetches the weather for you, and it answers with a PAR value.

Everything in this folder serves that sentence: asking the user for a location, collecting the weather, rebuilding exactly the 15 numbers the model was trained on, and presenting the answer honestly.

---

## 2. The mental model: one request, five stops

The most important thing to understand is that a prediction always travels the same path, no matter which page you are on:

```
   ① You                  a place, a date, an hour
       │
   ┌───▼────────────────────────────────────────────────────────┐
   │  ② core/weather.py    ask Open-Meteo what the weather was  │
   │                       (or will be) at that place and hour  │
   ├────────────────────────────────────────────────────────────┤
   │  ③ core/features.py   add the physics: where is the sun?   │
   │                       how clear is the sky?  → 22 numbers  │
   ├────────────────────────────────────────────────────────────┤
   │  ④ core/predict.py    hand the 15 the model knows to       │
   │                       XGBoost, get PAR back                │
   └───┬────────────────────────────────────────────────────────┘
       │
   ⑤ The page             show the number, a chart, and where
                          the weather came from
```

The **pages** (what you see) never do arithmetic themselves. They collect input and display results. All the real work lives in `core/`. This is the single most useful thing to know about the folder: *if you are looking for logic, it is in `core/`; if you are looking for layout, it is in `pages/`.*

---

## 3. File by file

### 📂 The entry point

| File | What it does |
|---|---|
| `app.py` | The **front door**. Run this, not the pages. It sets the page title and theme, styles the sidebar, and registers the four pages with `st.navigation()`. 58 lines, no logic. |
| `home.py` | The **landing page** — what the model is, what the three modes do, how it works. Pure presentation. |
| `assets/logo.svg` | The leaf-and-sun logo shown in the sidebar. |
| `.streamlit/config.toml` | Colours (the green `#2ecc71`, the dark background), and two practical settings: uploads up to 200 MB (the hosting container has ~2.7 GB of RAM and the pipeline needs several times the file size while it works), and a minimal toolbar so visitors do not see Streamlit's own Deploy button. |
| `requirements_web.txt` | The **exact** library versions the deployed site installs. Pinned on purpose — see §7. |

### 📂 `core/` — all of the logic

This is the part worth reading. Eight small modules, each with one job.

| File | One sentence | Key functions |
|---|---|---|
| `constants.py` | The McCree numbers (`0.45 × 4.57 = 2.06`) in one place, so the app and the notebooks cannot drift apart. | `MCCREE_FACTOR` |
| `weather.py` | Talks to Open-Meteo: picks the right endpoint for the requested date, and refuses dates it cannot serve. | `fetch_weather()`, `available_window()`, `geocode_city()` |
| `cache.py` | The same two calls, remembered: a repeated place-and-hour costs no network call and none of Open-Meteo's free quota (archive 24 h, forecast 1 h, place names 7 days). | `fetch_weather()`, `geocode_city()` |
| `features.py` | Turns raw weather into the numbers the model expects, adding solar geometry via `pvlib` — one row at a time, or a whole table in one vectorised pass. | `compute_features()`, `compute_features_batch()` |
| `predict.py` | Loads the trained model, feeds it exactly what it saw in training (median for gaps, training percentiles as clips), and says precisely why it cannot run when it cannot. | `predict_par()`, `predict_par_batch()`, `prepare_matrix()`, `model_status()`, `model_card()`, `mccree_estimate()` |
| `domain.py` | Asks whether the request looks like anything the model has seen: how far from the two Brandenburg stations, and which inputs sit outside their training range. | `check_location()`, `check_features()`, `describe()` |
| `export.py` | Turns any table into CSV, Excel, JSON or Parquet — built only when the visitor clicks, never on every rerun. | `convert()`, `download_bar()`, `to_zip_bytes()` |
| `dataset.py` | The whole pipeline in one function, for uploaded files: recognise the columns, clean like the notebooks, aggregate, featurise, score, report. | `prepare_dataset_for_prediction()`, `detect_columns()` |

Only `cache.py` and `export.py`'s one UI helper import Streamlit. Everything else in `core/` is plain Python and can be called from a script, a test, or a notebook, without a browser. That is why the test suite can check the app without launching it.

#### `weather.py` — where the data comes from

Open-Meteo runs **two** services, and the requested date decides which one is used:

| Requested date | Service | Covers |
|---|---|---|
| Before today | Archive (ERA5 reanalysis) | 1940-01-01 → yesterday |
| Today or later | Forecast model | today → today + 15 days |

Outside that window, the app raises `DateOutOfRangeError` **before sending any request**, and the page shows the covered range. The model itself has no time limit — this window is purely what the weather service can supply.

Two rules this module follows strictly:

- **Never substitute silently.** The requested hour must exist in the response and must carry a radiation value. If it does not, you get an error, not the nearest hour's reading. (An earlier version did snap to the nearest hour, which made a prediction for 2024 quietly return *today's* weather.)
- **Never let a network error crash a page.** Every call goes through `_get_json()`, which retries timeouts and converts everything else into a catchable `WeatherServiceError`.

#### `features.py` — the 22 numbers

`compute_features()` receives seven raw weather values and returns **22 columns**. The extra fifteen are computed, not measured:

```
measured by the weather service (7)      GHI, temperature, humidity, dew point,
                                         wind speed, wind direction, precipitation
computed by pvlib          (5)           zenith, elevation, airmass, clearness_kt, dni
computed by trigonometry   (2)           wind_sin, wind_cos
derived                    (8)           is_raining, temp_diff, dew_depression, …
```

Why encode wind direction as a sine and cosine? Because 359° and 1° are two degrees apart in reality but 358 apart as numbers. The model would read that gap as enormous. Splitting the angle into `sin` and `cos` makes the circle continuous again.

The model uses **15** of these 22. The rest are computed for display, or kept because the notebooks produced them.

#### `predict.py` — loading a model trained in a notebook

This module is longer than you would expect for "load a file and call `.predict()`", for one reason worth understanding.

The model was trained inside a Jupyter notebook, where the `XGBEnsemble` class (the three-seed average) was defined in the notebook's own scope. When `joblib` saved the model, it recorded the class as `__main__.XGBEnsemble` — "the class called XGBEnsemble, in whatever program is running". Loading it in the web app therefore fails, because the web app's `__main__` has no such class.

`_load_joblib_robust()` solves this by defining the same class and injecting it into `__main__` before unpickling, then retrying for any other notebook symbol the file happens to reference.

There is a deliberate limit: if it meets a symbol it does not recognise, it **raises instead of guessing**. A generic stub would let the site display `PAR = 0` with no error at all — a wrong answer that looks like a real one.

The same principle governs `predict_par()`: a missing feature column raises a `KeyError` rather than being filled with zero.

**What the model is actually fed.** Notebook 5 did two things to the training features before fitting: it replaced missing values with the training-set median, and it clipped the noisy meteorological channels (temperature, humidity, dew point, wind speed, rain, cell temperature) to their 1st–99th percentiles. Radiation and solar geometry were left untouched — their extremes are real. `prepare_matrix()` repeats exactly that at prediction time, from the same two small files the notebook saved (`train_medians_all_locations.pkl`, `clip_bounds_all_locations.pkl`), so a 36 °C afternoon is shown to the model as 30.5 °C — the hottest it ever learned from — and the page says so.

**Saying how good it is.** `model_card()` reads the held-out results saved by notebook 6 (MAE 29.4 vs 36.1 µmol/m²/s for the McCree formula, on 42,031 test rows from days the model never saw). The pages quote that number under every prediction as *typical error ± 29 µmol/m²/s*, so a visitor knows how much to trust the figure.

**When the model cannot run.** `model_status()` distinguishes three cases the old code lumped together: the file is missing; the file is a Git LFS *pointer* (a 130-byte text stub, which is what a host that does not resolve LFS ends up with); or the file exists but fails to load. Each page shows the matching sentence instead of a pickle traceback.

**Where the model file lives.** `predict.py` looks *outside* `webapp/`, at the repository root:

```
data/results/xgboost_model_all_locations.pkl                          the trained model
data/processed/pkl_features_GradientBoosting/feature_names_all_locations.pkl
                                                                      the 15 names, in order
```

That second file is the contract between the notebooks and the app. A model is only numbers — it knows "column 0", not "GHI". If the app built its columns in a different order, the model would read temperature as sunlight and return a confident, wrong answer. Loading the order from disk makes that impossible.

### 📂 `pages/` — what the visitor sees

Streamlit builds the sidebar from these filenames; the leading number sets the order.

| Page | For whom |
|---|---|
| `1_Normal_Mode.py` | Anyone. Three inputs, one answer. |
| `2_Expert_Mode.py` | Someone with their own sensor readings. |
| `3_Dataset_Upload.py` | Someone with a whole CSV file of measurements. |

---

## 4. The three modes, and how to use them

### 🌱 Normal Mode — "what is the PAR here, then?"

**You provide:** latitude, longitude, altitude, a date and an hour.
**The app provides:** everything else.

1. Enter coordinates. The table under *Example coordinates* has a few to copy, and right-clicking a spot in Google Maps gives you any others.
2. Pick a date between 1940 and 15 days from now, and an hour. The caption under the picker always shows the exact window.
3. Press **Predict PAR**.

You get the PAR value, the weather that produced it, the sun's position, the **Daily Light Integral** for that day with a crop recommendation, and a chart of the whole day with your chosen hour marked.

Above the results, one line always states **where the weather came from** — historical reanalysis, today's model, or a forecast and how many days out. This matters: a number from ERA5 for 1998 and a number from a +14-day forecast are not equally trustworthy, and the page never lets you forget which one you are looking at.

> Coordinates only — there is no city search on this page.

### ⚙️ Expert Mode — "here are my own readings"

**You provide:** everything, including GHI and all six weather values.

Use this when you have your own sensors, or want to ask a what-if question ("what if the sky cleared?"). Because nothing is fetched, **any date works** — the solar geometry is computed locally by `pvlib`.

The **Auto-fetch weather** button fills the fields from Open-Meteo as a starting point, then you adjust. It also fills the timezone field from the fetched location. Values outside a field's range are clamped, and the app tells you when it did so — a real reading like Yakutsk at −44.7 °C sits well outside the temperature input.

Expert Mode additionally shows the **McCree baseline** beside the model prediction, and the model's **feature importances** — which inputs it actually relies on.

### 📊 Dataset Upload — "score my whole file"

Upload a CSV, TXT or Excel file (up to 200 MB / 3 million rows) with a timestamp column and sensor readings. The page works in three numbered steps:

**1 · Upload.** The delimiter and encoding are sniffed (German exports with `;` and decimal commas are fine), and only the first 2,000 rows are read for a preview.

**2 · Which column is what.** The header names are matched loosely — `ghi`, `GHI_RC_01`, `Globalstrahlung`, `irradiance` and `shortwave_radiation` are all understood, and so are German names like `Lufttemperatur` or `Windrichtung` — and every guess can be corrected in a drop-down. Only a timestamp and a GHI column are required; a sensor that is not in the file is replaced by its training median, and the page says so. A measured PAR column (`PAR`, `PPFD`, …) is optional and enables the accuracy report. Only the mapped columns are read from disk, so a wide file with four useful columns costs a quarter of the memory.

**3 · Station location and timezone.** Coordinates are guessed from the file name when it starts with a place, the timezone is validated against the IANA list (with suggestions when it is misspelt), and `0°, 0°` is refused unless you insist — it is in the Atlantic.

Then the full pipeline from the notebooks runs in one pass (`core/dataset.py`), every step a column operation — a 145,000-row file takes about a second:

```
timestamps in any format, epoch or with offset  →  local wall time (dd.mm.yyyy read day-first)
sentinel values (-99999, 3276.7, …)             →  missing
physically impossible readings                   →  rows dropped   (notebook 2)
GHI ≤ 30 W/m² (night and twilight)               →  rows dropped   (notebook 2)
duplicate timestamps                             →  dropped        (notebook 2)
gaps up to 3 h                                   →  carried forward (ffill/bfill, notebook 2)
longer gaps                                      →  training median, counted and reported
wind direction                                   →  sin/cos BEFORE the minute average (notebook 4)
1-second rows                                    →  1-minute means
                                                 →  22 features per row, one pvlib pass
                                                 →  one model call for the whole table
```

The wind detail matters: averaging bearings of 355° and 5° gives 180° — a south wind out of a north wind. Training averaged the sine and cosine instead (a *resultant vector*), and so does the upload pipeline now.

**What comes back.** Row counts at every stage, a **cleaning report** (which step removed how many rows, and why), the model-vs-baseline **metrics table** when a measured PAR column was mapped (MAE, RMSE, nRMSE, MBE, R² for both), a time-series chart, a **training-domain check** (how far the station is from Laubsdorf/Nebelin, and which inputs leave the training range in what share of rows), and the model's feature importance.

**What you can take home** — the *Download* panel, in CSV, Excel (≤ 1,048,575 rows), JSON or Parquet:

| File | Contents |
|---|---|
| Predictions | one row per scored minute: model PAR, McCree PAR, their difference, the weather that went in, solar geometry, measured PAR if given |
| Cleaned data | your file after the cleaning steps, at its native resolution, before aggregation |
| Feature matrix | the 22 computed features per minute — exactly what the model saw, for reproducing the result elsewhere |
| Report (JSON) | settings, column mapping, every cleaning step with counts, metrics, domain check, model card |
| Everything (ZIP) | the three tables as CSV plus the report and a README |

Files are generated when you click, not while the page renders, so a two-million-row result does not get serialised four times every time a widget changes.

---

## 5. How one number travels through the app

Follow a single prediction end to end.

**①** You enter `51.6872, 14.4143`, altitude `84`, `2024-06-15` at `12:00`, and press Predict.

**②** `fetch_weather()` sees the date is in the past, so it calls the **archive** endpoint with `start_date = end_date = 2024-06-15` and `timezone=auto`. Open-Meteo replies with 24 hourly rows and the timezone `Europe/Berlin`. The row labelled `2024-06-15T12:00` is taken — exactly that one, or an error. GHI is `131 W/m²`.

**③** `compute_features()` localises the timestamp to `Europe/Berlin` (not UTC — that mistake shifts the sun by two hours), then asks `pvlib` where the sun was: zenith, elevation, airmass. It computes how clear the sky was, splits out the direct beam, and encodes the wind direction. The result is one row of 22 numbers.

**④** `predict_par()` reads the 15 feature names from disk, selects those columns **in that order**, and calls the ensemble. Three XGBoost models each answer; their average is the prediction.

**⑤** The page shows PAR with its *typical error ± 29 µmol/m²/s*, the source line *"Historical — ERA5 reanalysis… Matched hour: 2024-06-15 12:00"*, and the whole day's curve — drawn from the same response, so the chart and the number cannot disagree. If the location is far from the training stations, or an input lies outside what the model was trained on, a warning says so in one sentence. An *Export* expander offers the summary, the day series, the features and a JSON report.

---

## 6. Two ideas that matter more than the code

### Train/serve consistency — the model must be fed what it was taught

A model learns one input format: 15 numbers, in one order, computed one way. Anything that later asks it for a prediction must reproduce that **exactly** — same feature order, same units, same solar-geometry calculation, same timezone handling.

If the calculation drifts even slightly, the model still returns a number, and the number still looks plausible. Nothing crashes. That is what makes this class of bug dangerous, and it is why `core/features.py` mirrors notebook 4 step for step, why the feature order is read from a file rather than typed out, and why the app raises instead of filling a gap with zero.

### Silence is the enemy

Every design decision in `core/` points the same way: **an error the user can see is better than a number the user cannot check.** A missing hour, an unknown symbol in the model file, a feature the app cannot compute, a date outside the API's range — all of them stop and say so. None of them quietly return something plausible.

This is worth more than it sounds. A crash gets fixed in an hour. A number that is quietly wrong can survive a whole presentation.

### Say how sure you are

A model trained on two German stations will answer a question about Nairobi without blinking — decision trees do not extrapolate, they repeat their nearest training leaf. So every prediction now carries two honesty devices: the **typical error** (the held-out MAE, ± 29 µmol/m²/s) printed under the number, and a **training-domain check** (`core/domain.py`) that names the nearest training station and its distance, and lists any input that lies outside the range the model learned from — together with what the model will treat it as. Neither changes the number; both change how much a reader should trust it.

---

## 7. Running and deploying it

```bash
# from the repository root — the model files are stored with Git LFS
git lfs pull

cd webapp
pip install -r requirements_web.txt
python -m streamlit run app.py
```

The site opens at `http://localhost:8501`.

**Two requirements files, and which one counts.** `webapp/requirements_web.txt` holds the exact versions that are known to work together locally. The *deployed* site installs the repository's root `requirements.txt` instead, which is deliberately unpinned — Streamlit Cloud builds on Python 3.14, and several of the local pins have no 3.14 wheels. So the live site tracks current releases. If you add a widget, check it is not brand new, or the site can break on a version the pins never covered.

**Deployment.** The app is published from a separate branch that contains only `webapp/` plus the two model files it loads. The `data/` paths in `predict.py` must stay valid there, which is why those two files travel with it.

**After a push that adds a new function to `core/`.** Streamlit Cloud re-runs the page scripts when it sees a commit, but keeps modules it has already imported. New page code can therefore meet an old `core/` module and fail with `ImportError`, even though the repository is correct. The cure is a process restart: *Manage app → Reboot*, or any edit to the root `requirements.txt`, which forces a rebuild.

---

## 8. Where to start reading

| If you want to… | Open |
|---|---|
| see it run | `app.py`, then `python -m streamlit run app.py` |
| understand a prediction | `core/features.py`, then `core/predict.py` |
| understand the date limits | `core/weather.py`, `available_window()` |
| understand the upload pipeline | `core/dataset.py`, `prepare_dataset_for_prediction()` |
| understand the warnings under a prediction | `core/domain.py`, then `core/predict.py`, `prepare_matrix()` |
| add a download somewhere | `core/export.py`, `download_bar()` |
| change what a page looks like | the matching file in `pages/` |
| understand the model itself | the root [`PROJECT_CONCEPT.md`](../PROJECT_CONCEPT.md) |

---

<sub>Hochschule Anhalt · Data Science Master · 2026 · Web application for the PAR prediction model</sub>
