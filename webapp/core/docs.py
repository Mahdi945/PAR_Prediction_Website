"""
core/docs.py
────────────
The user manual, in English and German.

Content only - pages/4_Documentation.py renders it. Keeping it here means the
two languages sit side by side in one file, so a change to one is obvious when
the other is not updated with it.

Every figure is filled in from core.predict.model_card() at render time rather
than typed into the prose. Retrain the model and the manual follows; nobody has
to remember that the MAE is quoted in four places.
"""

from __future__ import annotations

from .constants import MCCREE_FACTOR, PAR_ENERGY_FRACTION, PAR_QUANTUM_EFFICACY
from .predict import model_card, TRAINING_STATIONS

LANGUAGES = {"en": "English", "de": "Deutsch"}


def _german_number(value: str) -> str:
    """1,234.5 -> 1.234,5 — German writes the separators the other way round."""
    return value.replace(",", "").replace(".", ",").replace("", ".")


def _facts(lang: str = "en") -> dict:
    """The numbers the manual quotes, from the deployed model's own record."""
    c = model_card()
    facts = {
        "mae": f"{c['test_mae']:.1f}",
        "mae_round": f"{c['test_mae']:.0f}",
        "base_mae": f"{c['baseline_mae']:.1f}",
        "rmse": f"{c['test_rmse']:.1f}",
        "base_rmse": f"{c['baseline_rmse']:.1f}",
        "nrmse": f"{c['test_nrmse']:.1f}",
        "r2": f"{c['test_r2']:.4f}",
        "base_r2": f"{c['baseline_r2']:.4f}",
        "gain": f"{c['mae_gain_pct']:.1f}",
        "n_train": f"{c['n_train']:,}",
        "n_val": f"{c['n_val']:,}",
        "n_test": f"{c['n_test']:,}",
        "n_feat": c["n_features"],
        # From core.constants, never retyped: tests/test_mccree_factor.py fails
        # the build if the factor is written into a source file by hand.
        "mccree": f"{MCCREE_FACTOR:.2f}",
        "par_frac": f"{PAR_ENERGY_FRACTION:.2f}",
        "par_eff": f"{PAR_QUANTUM_EFFICACY:.2f}",
    }
    if lang == "de":
        facts = {k: _german_number(v) if isinstance(v, str) else v
                 for k, v in facts.items()}
    facts["stations"] = " & ".join(TRAINING_STATIONS)
    facts["n_stations"] = len(TRAINING_STATIONS)
    return facts


# ═════════════════════════════════════════════════════════════════════════════
#  English
# ═════════════════════════════════════════════════════════════════════════════

EN = [
    ("what", "What ParPredict does", """
**PAR** — Photosynthetically Active Radiation — is the part of sunlight a plant
can actually use for photosynthesis: the band between **400 and 700 nm**. It is
what decides how fast a crop grows, and in **agrivoltaics** — solar panels and
farming on the same land — it is what you need to know before you decide how
much shade a crop can take.

The problem is practical. A PAR sensor is expensive and rare. A **GHI** sensor
(Global Horizontal Irradiance — total sunlight on a flat surface, in W/m²) is
cheap and sits at almost every weather station.

ParPredict converts the one into the other. Give it a place and a time, and it
predicts PAR in **µmol/m²/s** from GHI, six weather readings and the position of
the sun.

There is a simple physical rule for this already, the **McCree relation**:

> PAR ≈ GHI × {par_frac} × {par_eff} = GHI × **{mccree}**

That rule is the yardstick this whole project is measured against. The app shows
it next to every prediction so you can see for yourself whether the model earned
its place.
"""),

    ("words", "The words on the screen", """
| Term | Unit | What it means |
|---|---|---|
| **PAR** | µmol/m²/s | The light a plant can photosynthesise with. What the app predicts. |
| **GHI** | W/m² | Total sunlight hitting a flat surface. The model's main input. |
| **DLI** | mol/m²/day | Daily Light Integral — all the PAR a crop receives in one day. The figure growers actually plan with. |
| **Clearness index (kt)** | — | 0 to 1. How much of the sunlight at the top of the atmosphere reached the ground. Low = overcast, ~0.7 = clear sky. |
| **Solar elevation** | ° | How high the sun stands. Below 0.5° the app reports PAR = 0. |
| **Air mass** | — | How much atmosphere the light travelled through. 1 at the zenith, large near the horizon. |
| **MAE** | µmol/m²/s | Mean Absolute Error — the average size of a miss. |
| **nRMSE** | % | Error relative to the average PAR, so it can be compared between sites. |
"""),

    ("normal", "Normal Mode — the quick answer", """
For anyone who wants a number without owning a single instrument.

1. **Find a place.** Start typing a city or region; matches appear as you type.
   Choosing one fills in latitude, longitude, altitude and the timezone. The
   coordinates stay editable, because a geocoded town centre is not your field.
2. **Pick a date and time.** The clock starts at the current time *at that
   place*, not at yours. Weather is hourly, so minutes are ignored.
3. **Press Predict PAR.**

**What comes back**

- the predicted PAR, with its **typical error** underneath
- the **DLI** for that day, and which crops that suits
- the weather that went in: GHI, temperature, humidity, rain, clearness
- an irradiance curve for the whole day with your chosen hour marked
- where the numbers came from — archive, today, or forecast — and the exact hour matched
- the location on a map, and how far it is from the nearest training station
"""),

    ("expert", "Expert Mode — your own readings", """
For anyone holding sensor data of their own.

Enter every input by hand: GHI, air temperature, humidity, dew point, wind speed
and direction, rain. **Auto-fetch** fills them from Open-Meteo first if you would
rather start from real weather and adjust.

The timezone matters more here than anywhere else in the app: this page passes it
straight to the solar-geometry calculation, so a wrong zone puts the sun in the
wrong part of the sky. It is set from the coordinates automatically and follows
them when you move the point.

Alongside the prediction you get the **McCree baseline**, the difference between
the two, the model's **feature importance**, the computed solar geometry, and the
full 22-value feature vector the model was shown.
"""),

    ("upload", "Dataset Upload — a whole file at once", """
For scoring a logger export rather than a single moment.

**What it accepts.** CSV, TXT or Excel, up to **200 MB** and **3,000,000 rows** —
about a month of one-second logging. Any delimiter, decimal commas, timestamps as
local time, ISO with offset or Unix epoch, and `dd.mm.yyyy` read day-first.

**Three steps**

1. **Upload** — the delimiter and encoding are detected for you.
2. **Map the columns** — guessed from the header names in English *and* German
   (`Globalstrahlung`, `Lufttemperatur`, `Windrichtung` are all recognised).
   Only a timestamp and a GHI column are required. A measured PAR column is
   optional and turns on the accuracy report.
3. **Give the station's position** — guessed from the file name where possible.

**What it does** — the same cleaning the training data received: logger sentinels
removed, physically impossible readings dropped, night rows filtered, duplicate
timestamps removed, short gaps carried forward, wind averaged as a vector, then
one-minute means, the {n_feat} model features, and one model call.

**What you get back** — row counts at every stage, a cleaning report saying what
was removed and why, model-versus-baseline metrics if you supplied measured PAR,
a chart, a training-domain check, and downloads.
"""),

    ("weather", "Where the weather comes from", """
All weather comes from **Open-Meteo**, free and without an API key.

| Your date | Source | Covers |
|---|---|---|
| Before today | ERA5 reanalysis archive | 1940-01-01 → yesterday |
| Today and later | Forecast model | today → today + 15 days |

Outside that window the app **refuses and says so**, rather than quietly using a
nearby date. Every result states which source produced it and which hour was
matched.

**Two honest caveats.**

*Historical dates* come from ERA5, a modelled reanalysis on a roughly 25 km grid
— not a measurement at your field. Compared against our own two pyranometers it
read about **9–18 % high on GHI**, and PAR follows GHI almost one for one. On a
past date that bias is larger than the model's own error.

*Forecast dates* are a weather forecast. Their uncertainty grows with the horizon.
"""),

    ("model", "How the prediction is made", """
The model is **{model_name}**, trained on **{n_train}** one-minute rows from
{n_stations} research stations in Brandenburg, Germany ({stations}), with
{n_val} rows for validation and **{n_test}** held back for the final test.

It reads **{n_feat} features**:

- **8 measured** — GHI, air temperature, humidity, dew point, wind speed, two
  rain channels, and the reference-cell temperature
- **5 from solar physics**, computed with `pvlib` — zenith, elevation, air mass,
  clearness index, direct normal irradiance
- **2 for wind direction**, encoded as sine and cosine so that 359° and 1° are
  neighbours rather than opposites

Two choices are worth knowing about:

**The split is by whole days, not by rows.** Readings a minute apart are nearly
identical, so a random split would let the model recognise a neighbouring minute
instead of learning the physics. Measured that way it looked 38 % better than the
baseline; measured honestly, {gain} %.

**PAR is forced to rise with GHI.** A monotone constraint means more sunlight can
never produce less predicted PAR, whatever the other inputs do.
"""),

    ("accuracy", "How accurate it is — and when not to trust it", """
On **{n_test}** rows from days the model had never seen:

| | McCree formula | ParPredict |
|---|---|---|
| MAE (µmol/m²/s) | {base_mae} | **{mae}** |
| RMSE (µmol/m²/s) | {base_rmse} | **{rmse}** |
| R² | {base_r2} | **{r2}** |

The model is **{gain} % more accurate** than the physical formula on average
error. That gap is the entire justification for having a model at all.

**What "typical error ± {mae_round}" means.** It is the mean absolute error, so
about **two rows in three** land within that distance of the truth. It is not a
95 % guarantee — that would be roughly ±100. The error is also not uniform: in
absolute terms it grows with the light, while in relative terms it shrinks, from
around a quarter of the value in near-darkness to under 2 % in full sun.

**Where to be careful**

- **Far from Brandenburg.** The model has seen two German lowland sites. Anywhere
  else is extrapolation. The app measures the distance to the nearest training
  station and notes it under *Model domain*.
- **Outside the training range.** Decision trees do not extrapolate — beyond the
  values they were trained on they answer as they would at the edge. Inputs that
  fall outside are listed, and whether they were clipped.
- **On historical dates**, ERA5's own bias dominates (see above).
"""),

    ("downloads", "Taking the results with you", """
Every page exports, in **CSV, Excel, JSON or Parquet**.

| From | You get |
|---|---|
| Normal / Expert Mode | a one-row summary, the day's hourly series, all 22 computed features, and a JSON report |
| Dataset Upload | predictions per minute, your cleaned file, the feature matrix, a JSON report, or all of it as one ZIP |

Files are built when you press the button, not while the page draws, so a large
result costs nothing until you ask for it. The JSON report carries the settings,
every cleaning step with counts, the metrics, the domain check and the model
card — enough to cite or audit a run later.
"""),

    ("appearance", "Appearance and accessibility", """
**Light or dark** — the switch is in the sidebar under *Appearance*. It changes
the whole app, and your browser remembers it.

The layout reflows for tablets and phones. Colours were chosen for contrast
rather than for looks: the brand green fails legibility on white, so text uses a
darker green in light mode while borders and bars keep the brighter one.
"""),

    ("faq", "Common questions", """
**Why is my prediction 0?**
The sun is below the horizon at that place and time. PAR really is zero.

**Why does the wind look wrong?**
Open-Meteo reports wind at 10 m; station anemometers usually sit lower, so its
numbers run higher than the training data. Wind is the 13th most important of
{n_feat} features and moves the prediction by about 0.1 %, so this does not
matter in practice.

**Why is a whole file scored in seconds when a notebook takes minutes?**
The notebooks *build* the model — a hundred candidates are trained to find the
best one. The app only *uses* the finished model, and holds everything in memory
instead of writing files between stages.

**Can I predict for a greenhouse or under panels?**
Only by entering the GHI measured *there* in Expert Mode. The model converts the
light that arrives; it does not know what your structure does to it.

**Is my uploaded file stored?**
No. It is processed in memory for your session and never written to disk.
"""),

    ("credits", "Credits", """
Built at **Hochschule Anhalt**, Data Science Master Programme 2026.
Creator: **Mahdi Bey**.

Model: XGBoost. Solar geometry: `pvlib`. Weather: Open-Meteo. Interface:
Streamlit. Place names: the GeoNames database via Open-Meteo, and BigDataCloud
for the reverse lookup.

The McCree relation is from K. J. McCree (1972), which is where the factors
{par_frac} and {par_eff} come from.
"""),
]


# ═════════════════════════════════════════════════════════════════════════════
#  Deutsch
# ═════════════════════════════════════════════════════════════════════════════

DE = [
    ("what", "Was ParPredict macht", """
**PAR** — photosynthetisch aktive Strahlung — ist der Teil des Sonnenlichts, den
eine Pflanze tatsächlich für die Photosynthese nutzen kann: der Bereich zwischen
**400 und 700 nm**. Er bestimmt, wie schnell eine Kultur wächst, und in der
**Agri-Photovoltaik** — Solarmodule und Landwirtschaft auf derselben Fläche —
muss man ihn kennen, bevor man entscheidet, wie viel Schatten eine Kultur
verträgt.

Das Problem ist praktischer Natur: Ein PAR-Sensor ist teuer und selten. Ein
**GHI**-Sensor (Globalstrahlung — die gesamte Sonneneinstrahlung auf eine
waagerechte Fläche, in W/m²) ist günstig und steht an fast jeder Wetterstation.

ParPredict rechnet das eine in das andere um. Ort und Zeitpunkt eingeben, und die
Anwendung sagt PAR in **µmol/m²/s** aus GHI, sechs Wettergrößen und dem
Sonnenstand vorher.

Dafür gibt es bereits eine einfache physikalische Regel, die **McCree-Beziehung**:

> PAR ≈ GHI × {par_frac} × {par_eff} = GHI × **{mccree}**

An dieser Regel misst sich das gesamte Projekt. Die Anwendung zeigt sie neben
jeder Vorhersage, damit Sie selbst sehen, ob das Modell seinen Platz verdient.
"""),

    ("words", "Die Begriffe auf dem Bildschirm", """
| Begriff | Einheit | Bedeutung |
|---|---|---|
| **PAR** | µmol/m²/s | Das Licht, mit dem eine Pflanze Photosynthese betreiben kann. Das sagt die Anwendung vorher. |
| **GHI** | W/m² | Globalstrahlung auf eine waagerechte Fläche. Die wichtigste Eingangsgröße. |
| **DLI** | mol/m²/Tag | Tageslichtsumme — die gesamte PAR-Menge eines Tages. Damit planen Gärtnereien tatsächlich. |
| **Klarheitsindex (kt)** | — | 0 bis 1. Wie viel der Strahlung am Oberrand der Atmosphäre den Boden erreicht hat. Niedrig = bedeckt, ~0,7 = klarer Himmel. |
| **Sonnenhöhe** | ° | Wie hoch die Sonne steht. Unter 0,5° meldet die Anwendung PAR = 0. |
| **Luftmasse** | — | Wie viel Atmosphäre das Licht durchquert hat. 1 im Zenit, groß nahe am Horizont. |
| **MAE** | µmol/m²/s | Mittlerer absoluter Fehler — wie groß ein Fehlschlag im Mittel ausfällt. |
| **nRMSE** | % | Fehler bezogen auf den mittleren PAR-Wert, dadurch zwischen Standorten vergleichbar. |
"""),

    ("normal", "Normal Mode — die schnelle Antwort", """
Für alle, die einen Wert brauchen, ohne ein einziges Messgerät zu besitzen.

1. **Ort suchen.** Stadt oder Region eintippen; die Treffer erscheinen beim
   Tippen. Eine Auswahl füllt Breite, Länge, Höhe und Zeitzone. Die Koordinaten
   bleiben editierbar — ein geokodierter Ortsmittelpunkt ist nicht Ihr Feld.
2. **Datum und Uhrzeit wählen.** Die Uhr startet bei der aktuellen Zeit *an
   diesem Ort*, nicht bei Ihrer. Die Wetterdaten sind stündlich, Minuten werden
   daher ignoriert.
3. **Predict PAR** drücken.

**Was zurückkommt**

- der vorhergesagte PAR-Wert, darunter der **typische Fehler**
- die **Tageslichtsumme (DLI)** und welche Kulturen dazu passen
- die eingegangenen Wetterwerte: GHI, Temperatur, Luftfeuchte, Regen, Klarheit
- der Strahlungsverlauf des ganzen Tages mit markierter Stunde
- die Herkunft der Zahlen — Archiv, heute oder Vorhersage — und die getroffene Stunde
- der Ort auf der Karte und die Entfernung zur nächsten Trainingsstation
"""),

    ("expert", "Expert Mode — eigene Messwerte", """
Für alle, die eigene Sensordaten haben.

Alle Eingaben von Hand: GHI, Lufttemperatur, Luftfeuchte, Taupunkt,
Windgeschwindigkeit und -richtung, Niederschlag. **Auto-fetch** füllt die Felder
zunächst aus Open-Meteo, falls Sie von echten Wetterdaten ausgehen und dann
anpassen möchten.

Die Zeitzone ist hier wichtiger als irgendwo sonst: Diese Seite gibt sie direkt
an die Berechnung des Sonnenstands weiter, eine falsche Zone setzt die Sonne also
an die falsche Stelle des Himmels. Sie wird automatisch aus den Koordinaten
gesetzt und folgt ihnen, wenn Sie den Punkt verschieben.

Neben der Vorhersage erhalten Sie die **McCree-Referenz**, die Differenz zwischen
beiden, die **Merkmalswichtigkeit** des Modells, die berechnete Sonnengeometrie
und den vollständigen Merkmalsvektor aus 22 Werten.
"""),

    ("upload", "Dataset Upload — eine ganze Datei", """
Um einen Logger-Export zu bewerten statt eines einzelnen Zeitpunkts.

**Was akzeptiert wird.** CSV, TXT oder Excel, bis **200 MB** und **3.000.000
Zeilen** — etwa ein Monat bei Sekundentakt. Beliebiges Trennzeichen,
Dezimalkommas, Zeitstempel als Ortszeit, ISO mit Zeitzone oder Unix-Epoche;
`TT.MM.JJJJ` wird tagesweise zuerst gelesen.

**Drei Schritte**

1. **Hochladen** — Trennzeichen und Kodierung werden erkannt.
2. **Spalten zuordnen** — aus den Spaltennamen erraten, englisch *und* deutsch
   (`Globalstrahlung`, `Lufttemperatur`, `Windrichtung` werden erkannt).
   Erforderlich sind nur ein Zeitstempel und eine GHI-Spalte. Eine gemessene
   PAR-Spalte ist optional und schaltet den Genauigkeitsbericht frei.
3. **Standort angeben** — wenn möglich aus dem Dateinamen vorgeschlagen.

**Was passiert** — dieselbe Bereinigung wie bei den Trainingsdaten:
Logger-Ersatzwerte entfernt, physikalisch unmögliche Messwerte verworfen,
Nachtzeilen gefiltert, doppelte Zeitstempel entfernt, kurze Lücken fortgeschrieben,
Wind vektoriell gemittelt, dann Minutenmittel, die {n_feat} Merkmale und ein
einziger Modellaufruf.

**Was zurückkommt** — Zeilenzahlen nach jedem Schritt, ein Bereinigungsbericht mit
Begründung, Modell-gegen-Referenz-Metriken (sofern gemessene PAR-Werte vorliegen),
ein Diagramm, eine Prüfung des Trainingsbereichs und die Downloads.
"""),

    ("weather", "Woher die Wetterdaten kommen", """
Alle Wetterdaten stammen von **Open-Meteo**, kostenlos und ohne API-Schlüssel.

| Ihr Datum | Quelle | Abgedeckt |
|---|---|---|
| Vor heute | ERA5-Reanalyse-Archiv | 01.01.1940 → gestern |
| Heute und später | Vorhersagemodell | heute → heute + 15 Tage |

Außerhalb dieses Zeitraums **verweigert** die Anwendung die Auskunft, statt
stillschweigend ein nahes Datum zu verwenden. Jedes Ergebnis nennt die Quelle und
die getroffene Stunde.

**Zwei ehrliche Einschränkungen.**

*Vergangene Daten* stammen aus ERA5, einer modellierten Reanalyse auf einem Raster
von rund 25 km — keine Messung auf Ihrem Feld. Gegen unsere beiden Pyranometer
gerechnet lag ERA5 beim GHI etwa **9–18 % zu hoch**, und PAR folgt GHI nahezu
eins zu eins. An einem vergangenen Datum ist diese Abweichung größer als der
Fehler des Modells selbst.

*Zukünftige Daten* sind eine Wettervorhersage. Ihre Unsicherheit wächst mit dem
Vorhersagehorizont.
"""),

    ("model", "Wie die Vorhersage entsteht", """
Das Modell ist **{model_name}**, trainiert auf **{n_train}** Minutenwerten von
{n_stations} Forschungsstationen in Brandenburg ({stations}), mit {n_val} Zeilen
zur Validierung und **{n_test}** zurückgehaltenen Zeilen für den Abschlusstest.

Es liest **{n_feat} Merkmale**:

- **8 gemessene** — GHI, Lufttemperatur, Luftfeuchte, Taupunkt,
  Windgeschwindigkeit, zwei Niederschlagskanäle und die Referenzzellentemperatur
- **5 aus der Sonnenphysik**, mit `pvlib` berechnet — Zenit, Höhe, Luftmasse,
  Klarheitsindex, Direktnormalstrahlung
- **2 für die Windrichtung**, als Sinus und Kosinus kodiert, damit 359° und 1°
  benachbart sind und nicht entgegengesetzt

Zwei Entscheidungen sind erwähnenswert:

**Die Aufteilung erfolgt nach ganzen Tagen, nicht nach Zeilen.** Messwerte im
Minutenabstand sind nahezu identisch; eine zufällige Aufteilung ließe das Modell
die Nachbarminute wiedererkennen, statt die Physik zu lernen. So gemessen wirkte
es 38 % besser als die Referenz — ehrlich gemessen sind es {gain} %.

**PAR muss mit GHI steigen.** Eine Monotoniebedingung sorgt dafür, dass mehr
Sonnenlicht nie zu weniger vorhergesagtem PAR führen kann.
"""),

    ("accuracy", "Wie genau es ist — und wann nicht", """
Auf **{n_test}** Zeilen von Tagen, die das Modell nie gesehen hat:

| | McCree-Formel | ParPredict |
|---|---|---|
| MAE (µmol/m²/s) | {base_mae} | **{mae}** |
| RMSE (µmol/m²/s) | {base_rmse} | **{rmse}** |
| R² | {base_r2} | **{r2}** |

Das Modell ist beim mittleren Fehler **{gain} % genauer** als die physikalische
Formel. Genau dieser Abstand rechtfertigt überhaupt ein Modell.

**Was „typischer Fehler ± {mae_round}" bedeutet.** Es ist der mittlere absolute
Fehler: Etwa **zwei von drei** Werten liegen innerhalb dieses Abstands zur
Wahrheit. Es ist keine 95-%-Garantie — dafür wären es etwa ±100. Der Fehler ist
außerdem nicht gleichmäßig: absolut wächst er mit der Helligkeit, relativ sinkt
er — von rund einem Viertel des Werts bei Dämmerung auf unter 2 % bei voller
Sonne.

**Wo Vorsicht geboten ist**

- **Weit weg von Brandenburg.** Das Modell kennt zwei deutsche Tieflandstandorte.
  Alles andere ist Extrapolation. Die Anwendung misst die Entfernung zur nächsten
  Trainingsstation und vermerkt sie unter *Model domain*.
- **Außerhalb des Trainingsbereichs.** Entscheidungsbäume extrapolieren nicht:
  Jenseits der gelernten Werte antworten sie wie am Rand. Betroffene Eingaben
  werden aufgeführt, samt Angabe, ob sie begrenzt wurden.
- **Bei vergangenen Daten** überwiegt die Abweichung von ERA5 (siehe oben).
"""),

    ("downloads", "Ergebnisse mitnehmen", """
Jede Seite exportiert, als **CSV, Excel, JSON oder Parquet**.

| Von | Sie erhalten |
|---|---|
| Normal / Expert Mode | eine Zusammenfassung in einer Zeile, den Tagesverlauf, alle 22 berechneten Merkmale und einen JSON-Bericht |
| Dataset Upload | Vorhersagen je Minute, Ihre bereinigte Datei, die Merkmalsmatrix, einen JSON-Bericht oder alles als ZIP |

Die Dateien entstehen erst beim Klick, nicht beim Aufbau der Seite; ein großes
Ergebnis kostet also nichts, solange Sie es nicht anfordern. Der JSON-Bericht
enthält Einstellungen, jeden Bereinigungsschritt mit Zahlen, die Metriken, die
Bereichsprüfung und die Modellangaben — genug, um einen Lauf später zu zitieren
oder zu prüfen.
"""),

    ("appearance", "Darstellung und Barrierefreiheit", """
**Hell oder dunkel** — der Schalter steht in der Seitenleiste unter
*Appearance*. Er ändert die gesamte Anwendung, und der Browser merkt sich die
Wahl.

Das Layout passt sich Tablets und Telefonen an. Die Farben wurden nach Kontrast
gewählt, nicht nach Geschmack: Das Marken-Grün ist auf Weiß nicht lesbar genug,
deshalb verwendet Text im hellen Modus ein dunkleres Grün, während Rahmen und
Balken das hellere behalten.
"""),

    ("faq", "Häufige Fragen", """
**Warum ist meine Vorhersage 0?**
Die Sonne steht zu diesem Zeitpunkt unter dem Horizont. PAR ist dann wirklich
null.

**Warum wirkt der Wind falsch?**
Open-Meteo gibt Wind in 10 m Höhe an; Stationsanemometer hängen meist tiefer,
deshalb liegen die Werte über den Trainingsdaten. Wind ist das 13.-wichtigste von
{n_feat} Merkmalen und verschiebt die Vorhersage um etwa 0,1 % — praktisch also
ohne Bedeutung.

**Warum dauert eine ganze Datei Sekunden, ein Notebook aber Minuten?**
Die Notebooks *bauen* das Modell — hundert Kandidaten werden trainiert, um den
besten zu finden. Die Anwendung *benutzt* nur das fertige Modell und hält alles
im Arbeitsspeicher, statt zwischen den Schritten Dateien zu schreiben.

**Kann ich für ein Gewächshaus oder unter Modulen vorhersagen?**
Nur, indem Sie die *dort* gemessene Globalstrahlung im Expert Mode eingeben. Das
Modell wandelt das ankommende Licht um; was Ihre Konstruktion damit macht, weiß
es nicht.

**Wird meine hochgeladene Datei gespeichert?**
Nein. Sie wird für Ihre Sitzung im Arbeitsspeicher verarbeitet und nie auf die
Festplatte geschrieben.
"""),

    ("credits", "Impressum", """
Entstanden an der **Hochschule Anhalt**, Masterstudiengang Data Science 2026.
Ersteller: **Mahdi Bey**.

Modell: XGBoost. Sonnengeometrie: `pvlib`. Wetter: Open-Meteo. Oberfläche:
Streamlit. Ortsnamen: die GeoNames-Datenbank über Open-Meteo, für die
Rückwärtssuche BigDataCloud.

Die McCree-Beziehung stammt aus K. J. McCree (1972); daher kommen die Faktoren
{par_frac} und {par_eff}.
"""),
]

SECTIONS = {"en": EN, "de": DE}

INTRO = {
    "en": "Everything the site does, in one place. Pick a section, or read it through.",
    "de": "Alles, was die Anwendung kann, an einem Ort. Wählen Sie einen Abschnitt oder lesen Sie durch.",
}
TITLE = {"en": "Documentation", "de": "Dokumentation"}
CONTENTS = {"en": "Contents", "de": "Inhalt"}


def sections(lang: str) -> list[tuple[str, str, str]]:
    """The manual in `lang`, with the model's own figures filled in."""
    facts = _facts(lang)
    facts["model_name"] = model_card()["model"]
    out = []
    for anchor, title, body in SECTIONS.get(lang, EN):
        out.append((anchor, title, body.format(**facts).strip()))
    return out
