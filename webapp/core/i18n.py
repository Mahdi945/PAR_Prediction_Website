"""
core/i18n.py
────────────
Minimal EN/DE translation layer for the ParPredict UI.

Design: the dictionary key IS the English source text, so a call like
    t("Predict PAR")
reads naturally at the call site and no separate key namespace has to be
invented or kept in sync. A missing translation falls back to the English
text itself — the page never breaks because of a missing key.

No network call, no API key, nothing that can fail during a demo: the
current language is read from st.session_state["lang"], set once by the
toggle in app.py and shared by every page in the session.
"""
from __future__ import annotations

import streamlit as st

DEFAULT_LANG = "en"

# ── EN → DE ──────────────────────────────────────────────────────────────────
TRANSLATIONS: dict[str, str] = {

    # ═══════════════════════════════ app.py — navigation ═══════════════════════
    "Home": "Startseite",
    "Normal Mode": "Normal-Modus",
    "Expert Mode": "Experten-Modus",
    "Dataset Upload": "Datensatz-Upload",

    # ═══════════════════════════════ home.py ════════════════════════════════════
    "Nowcasting Photosynthetically Active Radiation for Agrivoltaic Systems":
        "Vorhersage der photosynthetisch aktiven Strahlung für Agri-Photovoltaik-Anlagen",
    "Powered by": "Basiert auf",

    "Model file not found.":
        "Modelldatei nicht gefunden.",
    "The XGBoost model is stored in Git LFS. Run":
        "Das XGBoost-Modell liegt in Git LFS. Führen Sie",
    "from the project root to download it, then refresh this page.":
        "im Projektverzeichnis aus, um es herunterzuladen, und laden Sie die Seite neu.",

    "Choose your mode": "Modus wählen",

    "Search for any city or enter coordinates.<br>Weather is fetched <em>automatically</em> from Open-Meteo — any date from 1940 to 15 days ahead.":
        "Stadt suchen oder Koordinaten eingeben.<br>Das Wetter wird <em>automatisch</em> von Open-Meteo geladen — jedes Datum von 1940 bis 15 Tage im Voraus.",
    "3 inputs &nbsp;·&nbsp; One prediction": "3 Eingaben &nbsp;·&nbsp; eine Vorhersage",

    "Full control &nbsp;·&nbsp; Full transparency": "Volle Kontrolle &nbsp;·&nbsp; volle Transparenz",
    "Enter your own sensor readings for maximum accuracy.<br>Inspect feature importance, McCree comparison,<br>and every intermediate computed value.":
        "Eigene Sensormesswerte für maximale Genauigkeit eingeben.<br>Feature Importance, McCree-Vergleich<br>und jeden berechneten Zwischenwert einsehen.",

    "Batch analysis &nbsp;·&nbsp; Full benchmark": "Stapelverarbeitung &nbsp;·&nbsp; vollständiger Vergleich",
    "Upload a full time-series dataset with sensor readings and location.<br>Clean it, resample it, and compare model predictions to a baseline.":
        "Vollständigen Zeitreihen-Datensatz mit Sensormesswerten und Standort hochladen.<br>Bereinigen, resamplen und die Modellvorhersagen mit einer Baseline vergleichen.",

    "Feature": "Merkmal",
    "Who is it for?": "Für wen ist es gedacht?",
    "Farmers, agronomists, general users": "Landwirt:innen, Agronom:innen, allgemeine Nutzer:innen",
    "Researchers, engineers with on-site sensors": "Forschende, Ingenieur:innen mit Sensoren vor Ort",
    "Inputs required": "Benötigte Eingaben",
    "Location + Date/Time <em>(3 fields)</em>": "Standort + Datum/Uhrzeit <em>(3 Felder)</em>",
    "All sensor readings manually <em>(17+ fields)</em>": "Alle Sensormesswerte manuell <em>(17+ Felder)</em>",
    "Weather data": "Wetterdaten",
    "Auto-fetched via Open-Meteo API": "Automatisch über die Open-Meteo-API",
    "You enter your own sensor values": "Sie geben eigene Sensorwerte ein",
    "Accuracy": "Genauigkeit",
    "Good (API weather ~hourly resolution)": "Gut (API-Wetter, ca. stündliche Auflösung)",
    "Maximum (real on-site measurements)": "Maximal (echte Messungen vor Ort)",
    "Results shown": "Angezeigte Ergebnisse",
    "PAR gauge · DLI · Forecast chart · Crop advice": "PAR-Anzeige · DLI · Prognose-Diagramm · Pflanzenempfehlung",
    "All of Normal + feature importance · McCree comparison · full feature table":
        "Alles aus Normal + Feature Importance · McCree-Vergleich · vollständige Merkmalstabelle",

    "Model Accuracy": "Modellgenauigkeit",
    "Model Features": "Modell-Merkmale",
    "Native Resolution": "Native Auflösung",
    "1 min": "1 Min.",
    "Monitoring Sites": "Messstationen",
    "Training Rows": "Trainingszeilen",
    "Coverage via API": "Abdeckung über API",
    "Global": "Weltweit",

    "📚  What is PAR and why does it matter for Agrivoltaics?":
        "📚  Was ist PAR und warum ist es für Agri-Photovoltaik wichtig?",
    "home.par_left": (
        "**PAR (Photosynthetically Active Radiation)**  \n"
        "ist der Teil des Sonnenlichts im Wellenlängenbereich 400–700 nm, den\n"
        "Pflanzen für die Photosynthese nutzen. Gemessen wird er in µmol/m²/s\n"
        "(Quantenflussdichte).\n\n"
        "**Agri-Photovoltaik (Agri-PV)** verbindet Solarstromerzeugung mit\n"
        "Landwirtschaft auf derselben Fläche. Solarmodule fangen einen Teil der\n"
        "einfallenden Strahlung (GHI) ab, und das verbleibende PAR, das die\n"
        "Pflanzen erreicht, bestimmt Wachstum, Wasserverbrauch und Ertrag.\n\n"
        "**Das Problem**  \n"
        "Die klassische McCree-Formel — `PAR ≈ 0,45 × GHI` — geht von einem\n"
        "festen spektralen Verhältnis aus und ignoriert Wolken, Regen,\n"
        "Sonnenstand und Atmosphärenbedingungen. In realen Anlagen führt das\n"
        "zu Fehlern von bis zu **35 % nRMSE**."
    ),
    "home.par_right": (
        "**Dieses Werkzeug nutzt maschinelles Lernen**, trainiert auf echten\n"
        "Sekundendaten von Agri-Photovoltaik-Messstationen in Deutschland\n"
        "(Laubsdorf & Nebelin, 2024–2025).\n\n"
        "| Methode | R² | nRMSE |\n"
        "|---|---|---|\n"
        "| McCree (0,45 × GHI) | 0,75 | 35 % |\n"
        "| Lineare Regression | 0,85 | 25 % |\n"
        "| Neuronales Netz (Prototyp) | 0,96 | 15,6 % |\n"
        "| **Dieses ML-Modell (XGBoost)** | **0,99** | **~8 %** |\n\n"
        "Das Modell lernt nichtlineare Zusammenhänge zwischen GHI, Sonnenstand,\n"
        "Luftfeuchtigkeit, Niederschlag und Temperatur, um PAR präzise\n"
        "vorherzusagen — für **jeden Ort weltweit** und für **jedes Datum**\n"
        "von 1940 bis 15 Tage im Voraus, über die Open-Meteo-Wetter-API."
    ),

    "How it works": "So funktioniert es",
    "step.Location": "Standort",
    "step.Location.desc": "Stadtname oder Breiten-/Längengrad eingeben.",
    "step.Weather": "Wetter",
    "step.Weather.desc": "Open-Meteo liefert stündliche GHI-, Temperatur-, Luftfeuchtigkeits-, Wind- und Niederschlagswerte — ERA5-Reanalyse für vergangene Daten, das Vorhersagemodell für heute und die nächsten 15 Tage.",
    "step.Solar Geometry": "Sonnenstand",
    "step.Solar Geometry.desc": "pvlib berechnet Zenitwinkel, Luftmasse, Klarheitsindex und DNI — dieselbe Physik wie im Training.",
    "step.Predict": "Vorhersage",
    "step.Predict.desc": "Das XGBoost-Modell berechnet PAR aus allen 22 Merkmalen in Millisekunden.",
    "step.Act": "Handeln",
    "step.Act.desc": "PAR-Schätzung und DLI-Prognose für Bewässerungsplanung, Pflanzenüberwachung und Ertragsprognose nutzen.",

    "Developers:": "Entwickler:innen:",
    "Supervisor:": "Betreuer:",

    # ═══════════════════════════════ shared across pages ════════════════════════
    "⚠️ Model not found. Run `git lfs pull` from the project root.":
        "⚠️ Modell nicht gefunden. Führen Sie `git lfs pull` im Projektverzeichnis aus.",
    "Latitude (°N)": "Breitengrad (°N)",
    "Southern hemisphere → negative. Range: −90 to +90":
        "Südhalbkugel → negativ. Bereich: −90 bis +90",
    "Longitude (°E)": "Längengrad (°E)",
    "Western hemisphere → negative. Range: −180 to +180":
        "Westhalbkugel → negativ. Bereich: −180 bis +180",
    "Altitude (m)": "Höhe (m)",
    "Used for precise solar geometry. Enter 0 if unknown.":
        "Für eine präzise Sonnenstandsberechnung. 0 eingeben, falls unbekannt.",
    "Date": "Datum",
    "Past dates use ERA5 reanalysis; today and future dates use the forecast model.":
        "Vergangene Daten nutzen die ERA5-Reanalyse; heute und zukünftige Daten nutzen das Vorhersagemodell.",
    "Right-click on Google Maps → copy coordinates.":
        "Rechtsklick in Google Maps → Koordinaten kopieren.",
    "Right-click Google Maps → copy lat, lon.":
        "Rechtsklick in Google Maps → Breite/Länge kopieren.",
    "🌙 **Night-time** — sun is below the horizon. PAR = 0.":
        "🌙 **Nachtzeit** — die Sonne steht unter dem Horizont. PAR = 0.",
    "🌙 **Night-time** — sun below horizon. PAR = 0.":
        "🌙 **Nachtzeit** — die Sonne steht unter dem Horizont. PAR = 0.",
    "Location": "Standort",
    "Zenith": "Zenit",
    "Elevation": "Höhenwinkel",
    "Airmass": "Luftmasse",
    "Clearness kt": "Klarheitsindex kt",

    # ═══════════════════════════════ pages/1_Normal_Mode.py ═════════════════════
    "Enter coordinates · Weather from Open-Meteo (1940 → today +15 days) · Predicted by XGBoost":
        "Koordinaten eingeben · Wetter von Open-Meteo (1940 → heute +15 Tage) · Vorhersage von XGBoost",
    "📍 Coordinates": "📍 Koordinaten",
    "🕐 Date & Time": "🕐 Datum & Uhrzeit",
    "Local time at that location": "Ortszeit am Standort",
    "Click the field or type the hour as HH:MM. Weather is hourly, so minutes are ignored.":
        "Feld anklicken oder die Uhrzeit als HH:MM eingeben. Wetterdaten sind stündlich, Minuten werden ignoriert.",
    "normal.weather_window_caption": (
        "📅 Wetterdaten verfügbar **{min} → {max}** — ERA5-Archiv bis gestern, "
        "Vorhersage bis heute +15. Die Zeitzone wird automatisch aus den "
        "Koordinaten bestimmt."
    ),
    "🌱  Predict PAR": "🌱  PAR vorhersagen",
    "📋 Example coordinates": "📋 Beispiel-Koordinaten",
    "⏳ Fetching weather from Open-Meteo…": "⏳ Wetterdaten werden von Open-Meteo geladen…",
    "⚙️ Computing solar geometry & predicting…": "⚙️ Sonnenstand wird berechnet & Vorhersage läuft…",
    "❌ Weather data unavailable:": "❌ Wetterdaten nicht verfügbar:",
    "❌ Weather API error:": "❌ Fehler bei der Wetter-API:",
    "❌ Prediction error:": "❌ Fehler bei der Vorhersage:",
    "Ready to predict PAR": "Bereit für die PAR-Vorhersage",
    "normal.welcome_card": (
        "Geben Sie links <strong style=\"color:#2ecc71\">Koordinaten</strong>\n"
        "und <strong style=\"color:#2ecc71\">Datum/Uhrzeit</strong> ein,<br>\n"
        "und klicken Sie dann auf <strong style=\"color:#2ecc71\">PAR vorhersagen</strong>.<br><br>\n"
        "Das Wetter wird <em>automatisch</em> für jeden Ort der Welt geladen —\n"
        "jedes Datum von <strong style=\"color:#2ecc71\">1940</strong>\n"
        "bis <strong style=\"color:#2ecc71\">15 Tage im Voraus</strong>."
    ),
    "Matched hour:": "Zugeordnete Stunde:",
    "local.": "Ortszeit.",
    "— modelled, gridded reanalysis, not station measurements.":
        "— modellierte, gerasterte Reanalyse, keine Stationsmessung.",
    "— forecast uncertainty grows with the horizon.":
        "— die Vorhersageunsicherheit wächst mit dem Zeithorizont.",
    "⚠️ Open-Meteo had no value for:": "⚠️ Open-Meteo hatte keinen Wert für:",
    "— typical values were used for those inputs.":
        "— für diese Eingaben wurden typische Werte verwendet.",
    "GHI": "GHI",
    "Temp": "Temp.",
    "Humidity": "Luftfeuchte",
    "Precip.": "Niederschlag",
    "Clearness": "Klarheit",
    "🤖 XGBoost Prediction": "🤖 XGBoost-Vorhersage",
    "Solar elevation": "Sonnenhöhe",
    "Very Low": "Sehr niedrig",
    "Low": "Niedrig",
    "Moderate": "Mäßig",
    "Good": "Gut",
    "High": "Hoch",
    "Daily Light Integral —": "Daily Light Integral (Tageslichtsumme) —",
    "🌿 Shade-tolerant crops (moss, ferns, microgreens)":
        "🌿 Schattenverträgliche Pflanzen (Moos, Farne, Microgreens)",
    "🥬 Lettuce, spinach, herbs — ideal":
        "🥬 Salat, Spinat, Kräuter — ideal",
    "🫑 Peppers, cucumbers, tomatoes (greenhouse)":
        "🫑 Paprika, Gurken, Tomaten (Gewächshaus)",
    "🍅 Tomatoes, most fruiting crops — excellent":
        "🍅 Tomaten, die meisten Fruchtgemüse — hervorragend",
    "🌻 Full-sun crops: sunflowers, corn, soybeans":
        "🌻 Vollsonnen-Pflanzen: Sonnenblumen, Mais, Soja",
    "Solar &amp; precipitation": "Sonne &amp; Niederschlag",
    "DNI": "DNI",
    "Raining": "Regen",
    "Yes 🌧️": "Ja 🌧️",
    "No ☀️": "Nein ☀️",
    "Dew depression": "Taupunktdifferenz",
    "historical": "historisch",
    "today": "heute",
    "Irradiance —": "Einstrahlung —",
    "selected time": "gewählte Uhrzeit",
    "🔍 All computed features": "🔍 Alle berechneten Merkmale",
    "Value": "Wert",
    "🗺️ Location on map": "🗺️ Standort auf der Karte",
    "Coordinates": "Koordinaten",
    "Date & Time": "Datum & Uhrzeit",
    "Predict PAR": "PAR vorhersagen",
    "XGBoost Prediction": "XGBoost-Vorhersage",
    "forecast": "Prognose",
    "d": "T.",
    "PAR est.": "PAR gesch.",
    "ML": "ML",

    # ═══════════════════════════════ pages/2_Expert_Mode.py ═════════════════════
    "Enter your own sensor readings for maximum accuracy · ML prediction vs McCree baseline · Feature importance":
        "Eigene Sensormesswerte für maximale Genauigkeit eingeben · ML-Vorhersage vs. McCree-Baseline · Feature Importance",
    "Full Sensor Input Dashboard": "Vollständiges Sensor-Eingabe-Dashboard",
    "no Open-Meteo value for": "kein Open-Meteo-Wert für",
    "clamped to the input range:": "auf den Eingabebereich begrenzt:",
    "Weather fetched —": "Wetter abgerufen —",
    "Timezone field updated.": "Zeitzonenfeld aktualisiert.",
    "Open-Meteo fetch failed:": "Open-Meteo-Abruf fehlgeschlagen:",
    "Unexpected error:": "Unerwarteter Fehler:",
    "📍 Location & Time": "📍 Standort & Zeit",
    "☀️ Solar Irradiance": "☀️ Solarstrahlung",
    "🌡️ Meteorological Sensors": "🌡️ Meteorologische Sensoren",
    "Timezone (IANA)": "Zeitzone (IANA)",
    "Filled in automatically by auto-fetch.": "Wird beim Auto-Abruf automatisch ausgefüllt.",
    "Local time": "Ortszeit",
    "expert.autofetch_caption": (
        "🌦️ Der Auto-Abruf deckt **{min} → {max}** ab. Bei manuell "
        "eingegebenen Messwerten funktioniert die Vorhersage für jedes "
        "Datum — der Sonnenstand wird lokal berechnet, nicht abgerufen."
    ),
    "GHI — Global Horizontal Irradiance (W/m²)": "GHI — Globalstrahlung (W/m²)",
    "Temperature (°C)": "Temperatur (°C)",
    "Humidity (%)": "Luftfeuchtigkeit (%)",
    "Dewpoint (°C)": "Taupunkt (°C)",
    "Wind speed (m/s)": "Windgeschwindigkeit (m/s)",
    "Wind dir (°)": "Windrichtung (°)",
    "Precipitation (mm/h)": "Niederschlag (mm/h)",
    "🌦️  Auto-fetch weather from Open-Meteo": "🌦️  Wetter automatisch von Open-Meteo abrufen",
    "⚙️  Predict PAR": "⚙️  PAR vorhersagen",
    "📋 Reference coordinates": "📋 Referenz-Koordinaten",
    "Ready for expert prediction": "Bereit für die Experten-Vorhersage",
    "expert.welcome_card": (
        "Geben Sie links Ihre <strong style=\"color:#f39c12\">Sensormesswerte</strong>\n"
        "ein und klicken Sie dann auf\n"
        "<strong style=\"color:#f39c12\">PAR vorhersagen</strong>.<br><br>\n"
        "Die Ergebnisse umfassen die ML-Vorhersage, den McCree-Vergleich<br>\n"
        "und eine vollständige Feature-Importance-Analyse."
    ),
    "🤖 ML Model (XGBoost)": "🤖 ML-Modell (XGBoost)",
    "📐 McCree Baseline": "📐 McCree-Baseline",
    "📊 Difference": "📊 Differenz",
    "ML > McCree": "ML > McCree",
    "ML < McCree": "ML < McCree",
    "% relative)": "% relativ)",
    "ML Model (µmol/m²/s)": "ML-Modell (µmol/m²/s)",
    "McCree Estimate (µmol/m²/s)": "McCree-Schätzung (µmol/m²/s)",
    "Feature Importance": "Feature Importance",
    "Feature importance not available for this model type.":
        "Feature Importance ist für diesen Modelltyp nicht verfügbar.",
    "Computed Solar Geometry": "Berechneter Sonnenstand",
    "🔍 Full feature vector (all 22 computed values)":
        "🔍 Vollständiger Merkmalsvektor (alle 22 berechneten Werte)",
    "Raw sensor": "Rohe Sensordaten",
    "pvlib solar": "pvlib Sonnenstand",
    "Wind cyclical": "Wind (zyklisch)",
    "Engineered": "Abgeleitet",
    "Category": "Kategorie",
    "Other": "Sonstiges",
    "Location & Time": "Standort & Zeit",
    "Solar Irradiance": "Solarstrahlung",
    "Meteorological Sensors": "Meteorologische Sensoren",
    "Auto-fetch weather from Open-Meteo": "Wetter automatisch von Open-Meteo abrufen",
    "ML Model (XGBoost)": "ML-Modell (XGBoost)",
    "McCree Baseline": "McCree-Baseline",
    "Difference": "Differenz",

    # ═══════════════════════════════ pages/3_Dataset_Upload.py ══════════════════
    "Upload a full sensor dataset, clean it, resample to one-minute values, and benchmark the PAR model against a baseline.":
        "Vollständigen Sensordatensatz hochladen, bereinigen, auf Minutenwerte resamplen und das PAR-Modell gegen eine Baseline vergleichen.",
    "⚠️ Model not found. Run `git lfs pull` from the project root before running this page.":
        "⚠️ Modell nicht gefunden. Führen Sie `git lfs pull` im Projektverzeichnis aus, bevor Sie diese Seite verwenden.",
    "📁 Upload Dataset": "📁 Datensatz hochladen",
    "Choose a CSV or Excel file": "CSV- oder Excel-Datei wählen",
    "The dataset should include timestamp, latitude, longitude and weather/sensor columns.":
        "Der Datensatz sollte Zeitstempel, Breitengrad, Längengrad und Wetter-/Sensorspalten enthalten.",
    "Max file size: 10 GB": "Maximale Dateigröße: 10 GB",
    "Could not load the uploaded file:": "Die hochgeladene Datei konnte nicht gelesen werden:",
    "dataset.loaded_rows": "{n} Zeilen aus {name} geladen.",
    "🧭 Column Mapping": "🧭 Spaltenzuordnung",
    "Timestamp column": "Zeitstempel-Spalte",
    "Latitude, longitude, and altitude are entered below. The app will infer values from the filename if available.":
        "Breitengrad, Längengrad und Höhe werden unten eingegeben. Die App versucht, Werte aus dem Dateinamen abzuleiten, falls möglich.",
    "Observed PAR column (optional — needed for error metrics)":
        "Gemessene PAR-Spalte (optional — für Fehlermetriken benötigt)",
    "Select the column containing measured PAR [µmol/m²/s]. Required to compute MAE, R², etc.":
        "Spalte mit dem gemessenen PAR-Wert [µmol/m²/s] auswählen. Wird für MAE, R² usw. benötigt.",
    "⚙️ Processing Settings": "⚙️ Verarbeitungseinstellungen",
    "dataset.inferred_city": "Aus dem Dateinamen abgeleitete Stadt: {city}",
    "⚠️ Could not look up coordinates automatically — enter latitude, longitude and altitude below.":
        "⚠️ Koordinaten konnten nicht automatisch ermittelt werden — bitte Breitengrad, Längengrad und Höhe unten eingeben.",
    "dataset.inferred_location": "Verwendeter abgeleiteter Standort: {display}",
    "Timezone": "Zeitzone",
    "📍 Coordinates (editable)": "📍 Koordinaten (editierbar)",
    "Adjust the latitude used for prediction.": "Für die Vorhersage verwendeten Breitengrad anpassen.",
    "Adjust the longitude used for prediction.": "Für die Vorhersage verwendeten Längengrad anpassen.",
    "Adjust the altitude used for prediction.": "Für die Vorhersage verwendete Höhe anpassen.",
    "The deployed XGBoost model is trained at 1-minute resolution; this is enforced for consistent predictions.":
        "Das eingesetzte XGBoost-Modell wurde mit Minutenauflösung trainiert; diese wird für konsistente Vorhersagen erzwungen.",
    "🚀 Run cleaning, feature engineering and prediction":
        "🚀 Bereinigung, Feature Engineering und Vorhersage starten",
    "Processing the uploaded dataset... This may take a few moments.":
        "Der hochgeladene Datensatz wird verarbeitet … das kann einen Moment dauern.",
    "🔄 Re-run processing": "🔄 Verarbeitung erneut ausführen",
    "Re-processing dataset with current settings...": "Datensatz wird mit aktuellen Einstellungen erneut verarbeitet …",
    "Processing failed:": "Verarbeitung fehlgeschlagen:",
    "## 📈 Results": "## 📈 Ergebnisse",
    "Rows uploaded": "Hochgeladene Zeilen",
    "Rows after cleaning": "Zeilen nach Bereinigung",
    "Rows after resampling": "Zeilen nach Resampling",
    "📊 Baseline vs Model — Full Metrics Comparison": "📊 Baseline vs. Modell — vollständiger Metrikvergleich",
    "dataset.evaluated_caption": "Ausgewertet auf {n} Tageszeilen mit PAR > 0 — entspricht den Trainingsbedingungen.",
    "Metric": "Metrik",
    "Baseline (McCree)": "Baseline (McCree)",
    "Model (XGBoost)": "Modell (XGBoost)",
    "Winner": "Sieger",
    "R²": "R²",
    "nRMSE (%)": "nRMSE (%)",
    "RMSE (µmol/m²/s)": "RMSE (µmol/m²/s)",
    "MAE  (µmol/m²/s)": "MAE  (µmol/m²/s)",
    "nMBE (%)": "nMBE (%)",
    "MBE  (µmol/m²/s)": "MBE  (µmol/m²/s)",
    "MAE improvement (%)": "MAE-Verbesserung (%)",
    "RMSE improvement (%)": "RMSE-Verbesserung (%)",
    "✅ Model": "✅ Modell",
    "⚠️ Baseline": "⚠️ Baseline",
    "✅ Yes": "✅ Ja",
    "⚠️ No": "⚠️ Nein",
    "✅ ~0": "✅ ~0",
    "⚠️ Bias": "⚠️ Verzerrung",
    "ℹ️ Accuracy metrics require an **Observed PAR column** — select one above and re-run to compute MAE, RMSE, MAPE and R².":
        "ℹ️ Für Genauigkeitsmetriken wird eine **gemessene PAR-Spalte** benötigt — oben auswählen und erneut ausführen, um MAE, RMSE, MAPE und R² zu berechnen.",
    "📉 Performance Comparison": "📉 Leistungsvergleich",
    "Observed PAR": "Gemessenes PAR",
    "PAR (µmol/m²/s)": "PAR (µmol/m²/s)",
    "✨ Feature Importance": "✨ Feature Importance",
    "Importance": "Wichtigkeit",
    "📋 Preview of Processed Data": "📋 Vorschau der verarbeiteten Daten",
    "Timestamp": "Zeitstempel",
    "Model PAR": "Modell-PAR",
    "Baseline PAR": "Baseline-PAR",
    "No processed rows were generated. Please verify your columns and timestamp values.":
        "Es wurden keine verarbeiteten Zeilen erzeugt. Bitte Spalten und Zeitstempel prüfen.",
    "Ready to analyze your own dataset": "Bereit, Ihren eigenen Datensatz zu analysieren",
    "Upload Dataset": "Datensatz hochladen",
    "Column Mapping": "Spaltenzuordnung",
    "Processing Settings": "Verarbeitungseinstellungen",
    "Coordinates (editable)": "Koordinaten (editierbar)",
    "Latitude": "Breitengrad",
    "Longitude": "Längengrad",
    "Altitude": "Höhe",
    "Run cleaning, feature engineering and prediction":
        "Bereinigung, Feature Engineering und Vorhersage starten",
    "Re-run processing": "Verarbeitung erneut ausführen",
    "Results": "Ergebnisse",
    "Baseline vs Model — Full Metrics Comparison": "Baseline vs. Modell — vollständiger Metrikvergleich",
    "Performance Comparison": "Leistungsvergleich",
    "Preview of Processed Data": "Vorschau der verarbeiteten Daten",
    "dataset.welcome_card": (
        "CSV- oder Excel-Datei mit Zeitstempeln, Koordinaten und Wetter-/Sensorwerten hochladen.<br>\n"
        "Die Seite bereinigt die Daten, resampelt sie auf Minutenwerte, berechnet die Merkmale, "
        "führt das Modell aus und vergleicht es mit der McCree-Baseline."
    ),
}


def t(text: str) -> str:
    """Return *text* in the current language.

    The dictionary key is the English source string; a missing entry (or an
    unrecognised language) falls back to the English text unchanged, so a
    forgotten translation is a cosmetic gap, never a crash.
    """
    if st.session_state.get("lang", DEFAULT_LANG) != "de":
        return text
    return TRANSLATIONS.get(text, text)


def current_lang() -> str:
    return st.session_state.get("lang", DEFAULT_LANG)


def t_block(key: str, english: str) -> str:
    """Translate a long or templated block by a short dictionary key.

    t() uses the English text itself as the lookup key, which does not
    work for multi-paragraph markdown or strings containing .format()
    placeholders. Here the caller passes both: *key* selects the German
    replacement, and *english* — the real content, kept inline at the call
    site so it stays readable and diffable — is what is shown in English
    mode, and what is returned if *key* is ever missing from the dictionary.
    """
    if st.session_state.get("lang", DEFAULT_LANG) != "de":
        return english
    return TRANSLATIONS.get(key, english)


def language_toggle() -> None:
    """Render the EN/DE switch. Call once, from the sidebar, in app.py.

    Streamlit reruns the whole script on every interaction, so changing
    st.session_state['lang'] here is immediately visible to every page in
    the same run — no page-to-page synchronisation needed.
    """
    st.session_state.setdefault("lang", DEFAULT_LANG)
    options = ["en", "de"]
    labels = {"en": "🇬🇧 English", "de": "🇩🇪 Deutsch"}
    st.segmented_control(
        "Language / Sprache",
        options=options,
        format_func=lambda code: labels[code],
        key="lang",
        label_visibility="collapsed",
    )
