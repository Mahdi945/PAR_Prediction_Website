"""
core/i18n.py
────────────
German for the whole interface, without an API key.

The translations are written once and committed. Nothing is fetched at runtime:
no key, no latency, no new way for the page to fail, and the domain terms are
right — *Globalstrahlung*, *Taupunkt*, *Tageslichtsumme*, *Klarheitsindex* — where
a general-purpose engine produces something a German agronomist would not say.

Three decisions worth knowing:

**Keyed by the English text, not by an identifier.** A missing entry falls back
to readable English rather than printing its own key. An earlier attempt at this
keyed on short names and put `home.par_left` on the page when an entry was
missing.

**Applied at the render boundary.** `install()` wraps Streamlit's own text
functions, so the pages need no edits and no call site can be forgotten. Text
that is not in the catalogue passes through untouched.

**Never applied to anything structural.** Only the label, body and help of a
widget are translated. Column names, dict keys, widget `key=` arguments and
session-state names are left alone — renaming a DataFrame column by language
crashed this app once already.

    language()  → "en" | "de"
    translate() → text, with any known English replaced by German
    install()   → wrap Streamlit once, at startup
    switcher()  → the sidebar control
"""

from __future__ import annotations

import re

import streamlit as st

__all__ = ["language", "translate", "install", "switcher", "LANGUAGES"]

LANGUAGES = {"en": "English", "de": "Deutsch"}
_KEY = "pp_lang"


# ═════════════════════════════════════════════════════════════════════════════
#  The catalogue: English source -> German
# ═════════════════════════════════════════════════════════════════════════════

CATALOG: dict[str, str] = {
    # Words that are the same in German - Zoom, Region, Format, Detail,
    # Sensor - are deliberately absent: an entry that changes nothing still
    # costs a regex on every string the page renders.
    # ── Navigation, shared ───────────────────────────────────────────────────
    "Home": "Start",
    "Normal Mode": "Normal-Modus",
    "Expert Mode": "Experten-Modus",
    "Dataset Upload": "Datei-Upload",
    "Documentation": "Dokumentation",
    "Appearance": "Darstellung",
    "Language": "Sprache",
    "Changes the whole app, and is remembered in this browser.":
        "Ändert die gesamte Anwendung und wird in diesem Browser gespeichert.",

    # ── Home ─────────────────────────────────────────────────────────────────
    "### Choose your mode": "### Wählen Sie Ihren Modus",
    "### How it works": "### So funktioniert es",
    "Open the documentation  ·  Dokumentation öffnen":
        "Dokumentation öffnen  ·  Open the documentation",
    "New here, or unsure which mode fits? The documentation explains every part of the site &mdash; in English and German.":
        "Neu hier oder unsicher, welcher Modus passt? Die Dokumentation erklärt jeden Teil der Anwendung &mdash; auf Deutsch und Englisch.",
    "📚  What is PAR and why does it matter for Agrivoltaics?":
        "📚  Was ist PAR und warum ist es für Agri-Photovoltaik wichtig?",
    "Search for any city or enter coordinates.":
        "Nach einer Stadt suchen oder Koordinaten eingeben.",
    "3 inputs": "3 Eingaben",
    "One prediction": "Eine Vorhersage",
    "Full control": "Volle Kontrolle",
    "Full transparency": "Volle Transparenz",
    "Batch analysis": "Stapelverarbeitung",
    "Full benchmark": "Vollständiger Vergleich",
    "Model Accuracy": "Modellgenauigkeit",
    "Model Features": "Modellmerkmale",
    "Native Resolution": "Native Auflösung",
    "Monitoring Sites": "Messstationen",
    "Coverage via API": "Abdeckung über API",
    "Results shown": "Angezeigte Ergebnisse",
    "Model file not found.": "Modelldatei nicht gefunden.",
    "The prediction model is not usable on this deployment.":
        "Das Vorhersagemodell ist in dieser Installation nicht verwendbar.",

    # ── Shared controls ──────────────────────────────────────────────────────
    "Latitude (°N)": "Breitengrad (°N)",
    "Longitude (°E)": "Längengrad (°E)",
    "Altitude (m)": "Höhe (m)",
    "Date": "Datum",
    "Local time at that location": "Ortszeit an diesem Ort",
    "Local time": "Ortszeit",
    "Timezone (IANA)": "Zeitzone (IANA)",
    "Search a city or region": "Stadt oder Region suchen",
    "Start typing a place…": "Ort eintippen …",
    "Southern hemisphere → negative. Range: −90 to +90":
        "Südhalbkugel → negativ. Bereich: −90 bis +90",
    "Western hemisphere → negative. Range: −180 to +180":
        "Westhalbkugel → negativ. Bereich: −180 bis +180",
    "Used for precise solar geometry. Enter 0 if unknown.":
        "Für die genaue Sonnengeometrie. Bei Unkenntnis 0 eingeben.",
    "Past dates use ERA5 reanalysis; today and future dates use the forecast model.":
        "Vergangene Daten stammen aus der ERA5-Reanalyse; heute und später aus dem Vorhersagemodell.",
    "Click the field or type the hour as HH:MM. Weather is hourly, so minutes are ignored.":
        "Feld anklicken oder Stunde als HH:MM eingeben. Die Wetterdaten sind stündlich, Minuten werden ignoriert.",
    "Filled in by the search above — still editable, so you can nudge them from the town centre to your own field.":
        "Aus der Suche oben übernommen — weiterhin änderbar, damit Sie vom Ortsmittelpunkt zu Ihrem Feld korrigieren können.",
    "Filled in by the search above — still editable.":
        "Aus der Suche oben übernommen — weiterhin änderbar.",
    "Type a town or region and press Enter; the matches appear in this same list. Coordinates, altitude and timezone come from the GeoNames database via Open-Meteo.":
        "Ort oder Region eintippen; die Treffer erscheinen in derselben Liste. Koordinaten, Höhe und Zeitzone stammen aus der GeoNames-Datenbank über Open-Meteo.",
    "📋 Example coordinates": "📋 Beispielkoordinaten",
    "📋 Reference coordinates": "📋 Referenzkoordinaten",
    "Right-click on Google Maps → copy coordinates.":
        "In Google Maps rechtsklicken → Koordinaten kopieren.",
    "Right-click Google Maps → copy lat, lon.":
        "In Google Maps rechtsklicken → Breite und Länge kopieren.",
    "Location": "Ort",

    # ── Normal Mode ──────────────────────────────────────────────────────────
    "# 🌱 Normal Mode — ParPredict": "# 🌱 Normal-Modus — ParPredict",
    "Enter coordinates · Weather from Open-Meteo (1940 → today +15 days) · Predicted by XGBoost":
        "Koordinaten eingeben · Wetter von Open-Meteo (1940 → heute +15 Tage) · Vorhersage mit XGBoost",
    "🌱  Predict PAR": "🌱  PAR vorhersagen",
    "⚙️  Predict PAR": "⚙️  PAR vorhersagen",
    "Ready to predict PAR": "Bereit für die PAR-Vorhersage",
    "Ready for expert prediction": "Bereit für die Experten-Vorhersage",
    "🌙 **Night-time** — sun is below the horizon. PAR = 0.":
        "🌙 **Nacht** — die Sonne steht unter dem Horizont. PAR = 0.",
    "🌙 **Night-time** — sun below horizon. PAR = 0.":
        "🌙 **Nacht** — Sonne unter dem Horizont. PAR = 0.",
    "📍 This point could not be named — open sea, or the lookup was unavailable. The coordinates are used exactly as given.":
        "📍 Dieser Punkt konnte nicht benannt werden — offenes Meer oder die Suche war nicht erreichbar. Die Koordinaten werden genau so verwendet.",
    "🔍 All computed features": "🔍 Alle berechneten Merkmale",
    "🗺️ Location on map": "🗺️ Ort auf der Karte",
    "How closely to frame the predicted point.":
        "Wie eng der vorhergesagte Punkt gezeigt wird.",
    "Field": "Feld",
    "Town": "Ort",
    "Daily Light Integral": "Tageslichtsumme",
    "Solar elevation": "Sonnenhöhe",
    "Zenith": "Zenit",
    "Airmass": "Luftmasse",
    "Clearness kt": "Klarheitsindex kt",
    "Raining": "Regen",
    "Dew depression": "Taupunktdifferenz",
    "Humidity": "Luftfeuchte",
    "Temp": "Temperatur",
    "Precip.": "Niederschlag",
    "Clearness": "Klarheit",
    "typical error": "typischer Fehler",
    "model error": "Modellfehler",
    "forecast ensemble spread": "Streuung des Vorhersage-Ensembles",
    "physics baseline": "physikalische Referenz",
    "vs the model": "gegenüber dem Modell",
    "for the weather shown": "für das angezeigte Wetter",
    "for the readings shown": "für die angezeigten Messwerte",
    "the forecast adds its own error on top": "hinzu kommt der Fehler der Vorhersage selbst",
    "Very Low": "Sehr niedrig",
    "Low": "Niedrig",
    "Moderate": "Mittel",
    "Good": "Gut",
    "High": "Hoch",

    # ── Expert Mode ──────────────────────────────────────────────────────────
    "# ⚙️ Expert Mode — Full Sensor Input Dashboard":
        "# ⚙️ Experten-Modus — vollständige Sensoreingabe",
    "Enter your own sensor readings for maximum accuracy · ML prediction vs McCree baseline · Feature importance":
        "Eigene Sensorwerte für höchste Genauigkeit · ML-Vorhersage gegen McCree-Referenz · Merkmalswichtigkeit",
    "GHI — Global Horizontal Irradiance (W/m²)": "GHI — Globalstrahlung (W/m²)",
    "Temperature (°C)": "Temperatur (°C)",
    "Humidity (%)": "Luftfeuchte (%)",
    "Dewpoint (°C)": "Taupunkt (°C)",
    "Wind speed (m/s)": "Windgeschwindigkeit (m/s)",
    "Wind dir (°)": "Windrichtung (°)",
    "Precipitation (mm/h)": "Niederschlag (mm/h)",
    "🌦️  Auto-fetch weather from Open-Meteo":
        "🌦️  Wetter automatisch von Open-Meteo laden",
    "Set from the coordinates, and by auto-fetch. Type to filter the list. Change it only if you know better — it decides where the sun is.":
        "Aus den Koordinaten gesetzt und vom automatischen Laden. Zum Filtern tippen. Nur ändern, wenn Sie es besser wissen — sie bestimmt den Sonnenstand.",
    "Feature Importance": "Merkmalswichtigkeit",
    "Computed Solar Geometry": "Berechnete Sonnengeometrie",
    "Feature importance not available for this model type.":
        "Für diesen Modelltyp ist keine Merkmalswichtigkeit verfügbar.",
    "🔍 Full feature vector (all 22 computed values)":
        "🔍 Vollständiger Merkmalsvektor (alle 22 berechneten Werte)",
    "Feature importance of the deployed model":
        "Merkmalswichtigkeit des eingesetzten Modells",
    "McCree Baseline": "McCree-Referenz",
    "Difference": "Differenz",
    "Raw sensor": "Rohe Sensorwerte",
    "Wind cyclical": "Wind zyklisch",
    "Engineered": "Abgeleitet",
    "Category": "Kategorie",
    "Value": "Wert",

    # ── Dataset Upload ───────────────────────────────────────────────────────
    "# 📊 Dataset Upload Mode": "# 📊 Datei-Upload",
    "Upload a sensor file, clean it the way the training data was cleaned, score every minute with the model, and download the results in the format you need.":
        "Sensordatei hochladen, wie die Trainingsdaten bereinigen, jede Minute mit dem Modell bewerten und die Ergebnisse im gewünschten Format herunterladen.",
    "## 📈 Results": "## 📈 Ergebnisse",
    "Choose a CSV, TXT or Excel file": "CSV-, TXT- oder Excel-Datei wählen",
    "One row per measurement. Needs a timestamp and a GHI column; other weather sensors improve the prediction, and a measured PAR column enables the accuracy report.":
        "Eine Zeile je Messung. Erforderlich sind ein Zeitstempel und eine GHI-Spalte; weitere Wettersensoren verbessern die Vorhersage, eine gemessene PAR-Spalte schaltet den Genauigkeitsbericht frei.",
    "Ready to score your own data": "Bereit, Ihre eigenen Daten zu bewerten",
    "Guessed from the header names — correct anything that is wrong. Only the mapped columns are read, so unrelated columns cost nothing.":
        "Aus den Spaltennamen erraten — bitte Falsches korrigieren. Nur zugeordnete Spalten werden gelesen, andere kosten nichts.",
    "🚀 Clean, engineer features and predict":
        "🚀 Bereinigen, Merkmale berechnen und vorhersagen",
    "Rows uploaded": "Hochgeladene Zeilen",
    "Rows after cleaning": "Zeilen nach Bereinigung",
    "1-minute bins scored": "Bewertete Minutenintervalle",
    "Native resolution": "Native Auflösung",
    "Period": "Zeitraum",
    "Timestamp *": "Zeitstempel *",
    "Local wall time at the station, an ISO string with offset, or a Unix epoch.":
        "Ortszeit an der Station, ISO-Zeit mit Zeitzone oder Unix-Zeitstempel.",
    "GHI — global horizontal irradiance, W/m² *": "GHI — Globalstrahlung, W/m² *",
    "The model's main input. Required.": "Die wichtigste Eingangsgröße. Erforderlich.",
    "Measured PAR, µmol/m²/s (optional)": "Gemessene PAR, µmol/m²/s (optional)",
    "Enables MAE / RMSE / R² of the model and the baseline on your data.":
        "Schaltet MAE / RMSE / R² für Modell und Referenz auf Ihren Daten frei.",
    "A sensor that is not in your file is replaced by the training median for every row. The prediction still works, with somewhat lower accuracy.":
        "Ein Sensor, der in Ihrer Datei fehlt, wird in jeder Zeile durch den Trainingsmedian ersetzt. Die Vorhersage funktioniert weiterhin, mit etwas geringerer Genauigkeit.",
    "Choose the GHI column — without it the model has nothing to work with.":
        "GHI-Spalte wählen — ohne sie hat das Modell keine Grundlage.",
    "Timezone of the timestamps (IANA)": "Zeitzone der Zeitstempel (IANA)",
    "Enter the station coordinates — solar geometry depends on them.":
        "Stationskoordinaten eingeben — die Sonnengeometrie hängt davon ab.",
    "🧹 Cleaning report — what was removed, and why":
        "🧹 Bereinigungsbericht — was entfernt wurde und warum",
    "Cleaning report": "Bereinigungsbericht",
    "Missing sensor values and how they were filled":
        "Fehlende Sensorwerte und wie sie gefüllt wurden",
    "🧭 Training-domain check — is this data like what the model learned from?":
        "🧭 Prüfung des Trainingsbereichs — ähneln diese Daten dem Gelernten?",
    "Every model input stays within the training range in every daytime bin.":
        "Alle Eingangsgrößen bleiben in jedem Tagesintervall im Trainingsbereich.",
    "Trees do not extrapolate: outside its training range the model answers as it would at the edge.":
        "Entscheidungsbäume extrapolieren nicht: Außerhalb des Trainingsbereichs antwortet das Modell wie am Rand.",
    "✨ Feature importance (gain, averaged over the 3 seeds)":
        "✨ Merkmalswichtigkeit (Gain, über die 3 Seeds gemittelt)",
    "📋 Preview of the scored data (first 20 bins)":
        "📋 Vorschau der bewerteten Daten (erste 20 Intervalle)",
    "One row per scored 1-minute bin: model and baseline PAR, the weather that went in, solar geometry and — if you provided it — the measured PAR.":
        "Eine Zeile je bewertetem Minutenintervall: PAR von Modell und Referenz, die eingegangenen Wetterwerte, die Sonnengeometrie und — sofern angegeben — die gemessene PAR.",
    "Your data after the cleaning steps, at its native resolution, before aggregation. Filled values are included; the cleaning report says how many.":
        "Ihre Daten nach der Bereinigung, in nativer Auflösung, vor der Aggregation. Gefüllte Werte sind enthalten; die Anzahl steht im Bereinigungsbericht.",
    "Exactly what the model saw — the 22 computed features per 1-minute bin, so results can be reproduced outside this app.":
        "Genau das, was das Modell gesehen hat — die 22 berechneten Merkmale je Minutenintervall, damit Ergebnisse außerhalb dieser Anwendung reproduzierbar sind.",
    "Settings, column mapping, every cleaning step with counts, metrics, the domain check and the model card — everything needed to cite or audit this run.":
        "Einstellungen, Spaltenzuordnung, jeder Bereinigungsschritt mit Zahlen, Metriken, Bereichsprüfung und Modellangaben — alles, um diesen Lauf zu zitieren oder zu prüfen.",
    "Predictions, cleaned data and feature matrix as CSV plus the JSON report, in one archive.":
        "Vorhersagen, bereinigte Daten und Merkmalsmatrix als CSV samt JSON-Bericht in einem Archiv.",
    "The file has fewer than two columns — it needs at least a timestamp and a GHI column.":
        "Die Datei hat weniger als zwei Spalten — mindestens ein Zeitstempel und eine GHI-Spalte sind nötig.",
    "0° N, 0° E is in the Atlantic Ocean — yes, the station really is there.":
        "0° N, 0° O liegt im Atlantik — ja, die Station steht wirklich dort.",
    "ℹ️ Accuracy metrics need a **measured PAR column** — map one in step 2 and run again.":
        "ℹ️ Für Genauigkeitsmetriken ist eine **gemessene PAR-Spalte** nötig — in Schritt 2 zuordnen und erneut ausführen.",
    "ℹ️ Fewer than 10 daytime bins had a measured PAR > 0, so no accuracy metrics were computed.":
        "ℹ️ Weniger als 10 Tagesintervalle hatten eine gemessene PAR > 0, daher wurden keine Genauigkeitsmetriken berechnet.",
    "❌ The server ran out of memory while processing this file. Upload a smaller file (one month at a time works well).":
        "❌ Dem Server ging beim Verarbeiten dieser Datei der Speicher aus. Bitte eine kleinere Datei hochladen (ein Monat auf einmal funktioniert gut).",
    "Other weather sensors (optional)": "Weitere Wettersensoren (optional)",
    "Predictions": "Vorhersagen",
    "Cleaned data": "Bereinigte Daten",
    "Feature matrix": "Merkmalsmatrix",
    "Everything": "Alles",
    "Report": "Bericht",
    "Download": "Herunterladen",
    "Summary": "Zusammenfassung",
    "Features": "Merkmale",
    "Importance": "Wichtigkeit",
    "Day series": "Tagesverlauf",
    "Metric": "Kennzahl",
    "Winner": "Besser",
    "Step": "Schritt",
    "Removed": "Entfernt",
    "Missing": "Fehlend",
    "Fallback": "Ersatzwert",
    "Input": "Eingangsgröße",
    "Unit": "Einheit",

    # ── Exports ──────────────────────────────────────────────────────────────
    "⬇ Export this prediction": "⬇ Diese Vorhersage exportieren",
    "⬇ Full report · JSON": "⬇ Vollständiger Bericht · JSON",
    "⬇ Report · JSON": "⬇ Bericht · JSON",
    "⬇ Everything · ZIP": "⬇ Alles · ZIP",
    "Prediction summary — one row with inputs, sources and both estimates":
        "Zusammenfassung — eine Zeile mit Eingaben, Quellen und beiden Schätzungen",
    "Prediction summary — your readings, both estimates and their difference":
        "Zusammenfassung — Ihre Messwerte, beide Schätzungen und ihre Differenz",
    "Irradiance over the selected day — hourly GHI, temperature and rain from Open-Meteo":
        "Strahlung am gewählten Tag — stündliche GHI, Temperatur und Niederschlag von Open-Meteo",
    "All 22 computed features — what the model actually saw":
        "Alle 22 berechneten Merkmale — was das Modell tatsächlich gesehen hat",
    "Excel export needs the openpyxl package.":
        "Der Excel-Export benötigt das Paket openpyxl.",
    "Parquet export needs the pyarrow package.":
        "Der Parquet-Export benötigt das Paket pyarrow.",

    # ── Landing page prose ───────────────────────────────────────────────────
    # These sit inside HTML blocks, wrapped across lines; matching normalises
    # whitespace so the key can be written as one sentence.
    "Predict Photosynthetically Active Radiation for Agrivoltaic Systems":
        "Photosynthetisch aktive Strahlung für Agri-Photovoltaik vorhersagen",
    # One entry including the <em>, because a lone word carries no space and so
    # is never fragment-matched - that guard is what keeps file paths intact.
    "Weather is fetched <em>automatically</em> from Open-Meteo — any date from 1940 to 15 days ahead.":
        "Das Wetter wird <em>automatisch</em> von Open-Meteo geladen — für jedes Datum von 1940 bis 15 Tage im Voraus.",
    "from Open-Meteo — any date from 1940 to 15 days ahead.":
        "von Open-Meteo geladen — für jedes Datum von 1940 bis 15 Tage im Voraus.",
    "Enter your own sensor readings for maximum accuracy.":
        "Eigene Sensorwerte für höchste Genauigkeit eingeben.",
    "Inspect feature importance, McCree comparison, and every intermediate computed value.":
        "Merkmalswichtigkeit, McCree-Vergleich und jeden Zwischenwert einsehen.",
    "Upload a full time-series dataset with sensor readings and location.":
        "Vollständige Zeitreihe mit Sensorwerten und Standort hochladen.",
    "Clean it, resample it, and compare model predictions to a baseline.":
        "Bereinigen, neu abtasten und die Modellvorhersagen mit einer Referenz vergleichen.",

    # comparison table
    "Best for": "Am besten geeignet für",
    "Farmers, agronomists, general users":
        "Landwirte, Agrarwissenschaftler, allgemeine Nutzer",
    "Researchers, engineers with on-site sensors":
        "Forschende und Ingenieure mit eigenen Sensoren",
    "Inputs required": "Erforderliche Eingaben",
    "All sensor readings manually": "Alle Sensorwerte manuell",
    "Auto-fetched via Open-Meteo API": "Automatisch über die Open-Meteo-API",
    "You enter your own sensor values": "Sie geben Ihre eigenen Sensorwerte ein",
    "Accuracy": "Genauigkeit",
    "Good (API weather ~hourly resolution)":
        "Gut (API-Wetter mit etwa stündlicher Auflösung)",
    "Maximum (real on-site measurements)":
        "Höchste (echte Messungen vor Ort)",
    "PAR gauge · DLI · Forecast chart · Crop advice":
        "PAR-Anzeige · Tageslichtsumme · Tagesverlauf · Kulturempfehlung",
    "All of Normal + feature importance · McCree comparison · full feature table":
        "Alles aus dem Normal-Modus + Merkmalswichtigkeit · McCree-Vergleich · vollständige Merkmalstabelle",
    "Training Rows (1-min)": "Trainingszeilen (1 Min.)",

    # how it works
    "Enter a city name or latitude / longitude coordinates.":
        "Einen Ortsnamen oder Breiten- und Längengrad eingeben.",
    "Open-Meteo supplies hourly GHI, temperature, humidity, wind and rain — 1940 to 15 days ahead.":
        "Open-Meteo liefert stündlich Globalstrahlung, Temperatur, Luftfeuchte, Wind und Niederschlag — von 1940 bis 15 Tage im Voraus.",
    "Solar Geometry": "Sonnengeometrie",
    "Predict": "Vorhersagen",
    "Act": "Handeln",
    "Weather": "Wetter",

    # the PAR explainer
    "is the portion of sunlight in the 400–700 nm wavelength range that plants use for photosynthesis. It is measured in µmol/m²/s (quantum flux).":
        "ist der Anteil des Sonnenlichts im Wellenlängenbereich 400–700 nm, den Pflanzen für die Photosynthese nutzen. Gemessen wird er in µmol/m²/s (Quantenstromdichte).",
    "**Agrivoltaics (AgriPV)** combines solar energy production with agriculture on the same land. Solar panels intercept part of the incoming irradiance (GHI), and the remaining PAR reaching the crops determines their growth, water consumption and yield.":
        "**Agri-Photovoltaik (AgriPV)** verbindet Stromerzeugung und Landwirtschaft auf derselben Fläche. Die Module fangen einen Teil der einfallenden Globalstrahlung ab; die verbleibende PAR, die die Kulturen erreicht, bestimmt Wachstum, Wasserverbrauch und Ertrag.",
    "**The Problem**": "**Das Problem**",
    "The model learns non-linear interactions between GHI, solar position, humidity, precipitation and temperature to predict PAR accurately —":
        "Das Modell lernt nichtlineare Zusammenhänge zwischen Globalstrahlung, Sonnenstand, Luftfeuchte, Niederschlag und Temperatur, um PAR genau vorherzusagen —",
}

# Longest first, so a long sentence is matched before a word inside it.
_FRAGMENTS = sorted(CATALOG.items(), key=lambda kv: -len(kv[0]))
_MIN_FRAGMENT = 8


def _compile(source: str):
    """Match this phrase even when it is broken across lines.

    Prose inside the HTML blocks is wrapped and indented, so the literal string
    never appears contiguously. Every run of whitespace in the catalogue key
    therefore matches any run of whitespace in the page.

    The `probe` is the longest word in the phrase, used as a cheap pre-filter:
    if it is absent the regex is never run, which keeps a few hundred entries
    from being searched on every markdown call.
    """
    parts = source.split()
    if not parts:
        return None
    probe = max(parts, key=len)
    pattern = r"\s+".join(re.escape(w) for w in parts)
    return probe, re.compile(pattern)


_FLAT = {" ".join(k.split()): v for k, v in CATALOG.items()}
_COMPILED = []
for _src, _dst in _FRAGMENTS:
    if len(_src) >= _MIN_FRAGMENT and " " in _src:
        _c = _compile(_src)
        if _c:
            _COMPILED.append((_c[0], _c[1], _dst))


# ═════════════════════════════════════════════════════════════════════════════
#  Which language
# ═════════════════════════════════════════════════════════════════════════════

def _browser_language() -> str:
    """navigator.language, via st.context.locale. English unless it says German."""
    try:
        loc = st.context.locale
    except Exception:
        return "en"
    return "de" if isinstance(loc, str) and loc.lower().startswith("de") else "en"


def language() -> str:
    """The visitor's choice, else what their browser asked for."""
    chosen = st.session_state.get(_KEY)
    if chosen in LANGUAGES:
        return chosen
    return _browser_language()


def translate(text):
    """English in, German out — where the catalogue knows the phrase.

    An exact match wins. Otherwise known sentences are replaced inside longer
    text, which is how prose embedded in an HTML block gets translated without
    the surrounding markup being touched.
    """
    if not isinstance(text, str) or not text or language() != "de":
        return text
    hit = CATALOG.get(text)
    if hit is not None:
        return hit
    stripped = text.strip()
    hit = CATALOG.get(stripped)
    if hit is not None:
        return text.replace(stripped, hit)
    # Only prose gets fragment substitution. Without this guard, "Documentation"
    # would be replaced inside "pages/4_Documentation.py" and the page link
    # would stop resolving.
    if len(text) < _MIN_FRAGMENT or " " not in text:
        return text
    # Normalised lookup: a whole block that differs only in wrapping.
    flat = " ".join(text.split())
    hit = _FLAT.get(flat)
    if hit is not None:
        return hit
    for probe, rx, dst in _COMPILED:
        if probe in text:
            text = rx.sub(lambda _m, d=dst: d, text)
    return text


# ═════════════════════════════════════════════════════════════════════════════
#  Applying it
# ═════════════════════════════════════════════════════════════════════════════

# (function name, how many leading positional arguments are visible text)
_TEXT_FIRST = ("markdown", "caption", "write", "header", "subheader", "title",
               "button", "radio", "selectbox", "number_input", "text_input",
               "text_area", "date_input", "time_input", "file_uploader",
               "checkbox", "toggle", "slider", "multiselect", "expander",
               "metric", "info", "warning", "error", "success",
               "download_button", "page_link", "color_picker")
_LIST_FIRST = ("tabs",)
_PAGE_FIRST = ("page_link",)
_INSTALLED = "_pp_i18n_wrapped"


def _wrap(fn, first_is_list=False, skip_first=False, offset=0):
    """Wrap a text function so its visible arguments pass through the catalogue.

    `offset` is 1 when patching the class rather than the module: an unbound
    method receives `self` first, so the label is one place further along.
    Getting this wrong is silent - st.markdown() translates, col.markdown() and
    st.sidebar.radio() quietly do not, because the DeltaGenerator itself is
    handed to translate() and comes back unchanged.
    """
    if getattr(fn, _INSTALLED, False):
        return fn

    def wrapper(*args, **kwargs):
        args = list(args)
        i = offset
        if len(args) > i and not skip_first:
            if first_is_list and isinstance(args[i], (list, tuple)):
                args[i] = [translate(x) for x in args[i]]
            else:
                args[i] = translate(args[i])
        for name in ("label", "help", "body", "text", "placeholder"):
            if name in kwargs:
                kwargs[name] = translate(kwargs[name])
        return fn(*args, **kwargs)

    setattr(wrapper, _INSTALLED, True)
    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    return wrapper


def install() -> None:
    """Wrap Streamlit's text functions once per process.

    Both the module attributes (``st.markdown``) and the class methods
    (``col.markdown``, ``st.sidebar.radio``) are patched: ``st.markdown`` is a
    bound method captured at import, so patching only the class would miss it,
    and patching only the module would miss every column and the sidebar.
    """
    from streamlit.delta_generator import DeltaGenerator

    for name in _TEXT_FIRST + _LIST_FIRST:
        is_list = name in _LIST_FIRST
        # page_link takes the page itself first; only its label is text.
        skip = name in _PAGE_FIRST
        method = getattr(DeltaGenerator, name, None)
        if method is not None and not getattr(method, _INSTALLED, False):
            setattr(DeltaGenerator, name, _wrap(method, is_list, skip, offset=1))
        fn = getattr(st, name, None)
        if fn is not None and not getattr(fn, _INSTALLED, False):
            setattr(st, name, _wrap(fn, is_list, skip))

    # Sidebar entries are built from st.Page(title=...), which is a class rather
    # than a text function, so it needs its own small wrapper.
    page_cls = getattr(st, "Page", None)
    if page_cls is not None and not getattr(page_cls, _INSTALLED, False):
        def _page(*args, **kwargs):
            if "title" in kwargs:
                kwargs["title"] = translate(kwargs["title"])
            return page_cls(*args, **kwargs)
        setattr(_page, _INSTALLED, True)
        st.Page = _page


def switcher(*, key: str = _KEY, location=None) -> None:
    """Language control. Sits beside Appearance in the sidebar."""
    target = location if location is not None else st.sidebar
    options = list(LANGUAGES)
    if key not in st.session_state:
        st.session_state[key] = _browser_language()
    target.radio(
        "Language",
        options,
        format_func=lambda c: LANGUAGES[c],
        key=key,
        horizontal=True,
        help="Switches the whole interface, documentation included.",
    )
