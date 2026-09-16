# Explication complète de la page web webapp

Ce document explique simplement et en détail comment la page web de ce projet a été construite "from scratch", comment elle communique avec les APIs externes, et comment elle est reliée à votre modèle XGBoost.

---

## 1. Objectif global

La page web s’appelle PAR Predictor.

Son but est de prédire la PAR, c’est-à-dire la radiation photosynthétiquement active, en utilisant :

- des coordonnées GPS,
- une date / heure,
- des données météo,
- et un modèle d’intelligence artificielle.

En pratique, l’utilisateur peut :

- entrer une ville ou des coordonnées,
- récupérer automatiquement la météo depuis une API,
- calculer des variables techniques (angle solaire, airmass, clarté du ciel, etc.),
- envoyer ces informations au modèle XGBoost,
- obtenir une prédiction de PAR en quelques millisecondes.

Le projet est donc un mélange de :

- interface utilisateur,
- traitement de données,
- calcul de features,
- appel à des API externes,
- chargement et utilisation d’un modèle ML.

---

## 2. Structure du projet webapp

Le dossier principal de l’application est :

```text
webapp/
├── app.py
├── home.py
├── pages/
│   ├── 1_Normal_Mode.py
│   ├── 2_Expert_Mode.py
│   └── 3_Dataset_Upload.py
├── core/
│   ├── __init__.py
│   ├── weather.py
│   ├── features.py
│   ├── predict.py
│   └── dataset.py
├── assets/
│   └── logo.svg
├── .streamlit/
│   └── config.toml
├── requirements_web.txt
└── README.md
```

Chaque fichier a un rôle précis :

- app.py : point d’entrée principal de l’application,
- home.py : page d’accueil avec le menu et les choix de mode,
- pages/*.py : écrans fonctionnels de l’application,
- core/weather.py : appels API météo,
- core/features.py : calcul des variables d’entrée du modèle,
- core/predict.py : chargement du modèle XGBoost et prédiction,
- core/dataset.py : traitement des fichiers CSV uploadés,
- requirements_web.txt : dépendances Python de l’application web.

---

## 3. Comment la page web a été développée "from scratch"

### 3.1 Le framework principal : Streamlit

La page web est construite avec Streamlit, qui est un outil Python très simple pour créer des applications de data science et d’IA.

Avantages :

- très rapide à développer,
- facile à créer des formulaires,
- parfait pour des dashboards de machine learning,
- utile pour afficher des graphiques, cartes, tableaux, métriques.

Dans ce projet, Streamlit sert à :

- afficher la page d’accueil,
- demander les paramètres à l’utilisateur,
- afficher les résultats de prédiction,
- montrer la comparaison avec la méthode McCree,
- montrer les variables du modèle.

Le fichier principal est :

```python
webapp/app.py
```

Il contient la configuration globale de l’interface :

- titre de la page,
- icône,
- thème sombre,
- barre latérale,
- navigation entre les pages.

Exemple simplifié :

```python
st.set_page_config(
    page_title="PAR Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

Cela permet d’avoir une app moderne et propre, ressemblant à un dashboard.

---

### 3.2 La navigation entre les pages

Dans `app.py`, la navigation est faite avec :

```python
st.navigation([
    st.Page("home.py", title="Home", icon="🏠"),
    st.Page("pages/1_Normal_Mode.py", title="Normal Mode", icon="🌱"),
    st.Page("pages/2_Expert_Mode.py", title="Expert Mode", icon="⚙️"),
    st.Page("pages/3_Dataset_Upload.py", title="Dataset Upload", icon="📊"),
])
```

Cela crée plusieurs écrans dans la même application :

1. Home
2. Normal Mode
3. Expert Mode
4. Dataset Upload

Ainsi, on ne construit pas plusieurs apps, mais une seule app multi-pages.

---

### 3.3 La page d’accueil

Le fichier :

```python
webapp/home.py
```

est la page d’introduction.

Elle contient :

- logo du projet,
- titre “PAR Predictor”,
- description du projet,
- sélection du mode de travail,
- message d’avertissement si le modèle n’est pas chargé,
- accès rapide aux différentes vues.

La page n’a pas directement de logique ML avancée. Elle sert surtout comme portail d’entrée.

---

## 4. APIs used by the application

There are two different kinds of “API” in this project:

### 4.1 External API: Open-Meteo

The file:

```python
webapp/core/weather.py
```

contains the HTTP calls to Open-Meteo.

It is used to:

- convert a city name into GPS coordinates,
- fetch weather for a given location,
- retrieve variables such as:
  - temperature,
  - humidity,
  - dew point,
  - wind speed,
  - wind direction,
  - precipitation,
  - solar radiation (GHI).

This is a real API call to a public external service.

### 4.2 Internal project API: Python functions acting like services

The project does not have a real REST backend built with FastAPI, Flask, or Django.

This means that the application is not exposing a separate HTTP endpoint such as:

```text
POST /predict
GET /health
```

Instead, the app uses Python functions directly inside Streamlit:

- `fetch_weather(...)`
- `geocode_city(...)`
- `compute_features(...)`
- `load_model()`
- `predict_par(...)`

These functions are called directly from the frontend pages and behave like an internal service layer.

So the architecture is:

```text
Streamlit UI
   ↓
Python functions
   ↓
Open-Meteo API / model inference
```

This is simpler than a full backend API, but it is also less reusable outside the app itself.

### 4.3 How to turn the XGBoost model into a real API

If you want to expose the model as a real web API, you can wrap it in a backend such as FastAPI.

The idea would be:

1. receive JSON from the client,
2. validate the inputs,
3. build the same feature table,
4. call the XGBoost model,
5. return the prediction as JSON.

Example concept:

```python
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    lat: float
    lon: float
    alt: float
    dt: str
    ghi: float
    temp: float
    rh: float
    dwp: float
    ws: float
    wd: float
    prec: float

@app.post("/predict")
def predict(data: InputData):
    # 1. build weather dict
    weather = {
        "GHI_RC_01": data.ghi,
        "Temp_WS": data.temp,
        "RH_WS": data.rh,
        "DWP_WS": data.dwp,
        "WS_WS": data.ws,
        "WD_WS": data.wd,
        "PREC_INT_WS": data.prec,
    }

    # 2. compute features
    features_df, _ = compute_features(data.lat, data.lon, data.alt, data.dt, weather, "Europe/Berlin")

    # 3. load model and predict
    model, feature_names = load_model()
    X = pd.DataFrame(index=features_df.index)
    for col in feature_names:
        X[col] = features_df[col] if col in features_df.columns else 0.0

    pred = float(model.predict(X)[0])
    return {"par_prediction": pred}
```

In that version, the web app and the model are separated:

- Streamlit is the frontend,
- FastAPI is the backend,
- the model is loaded by the API server,
- the app sends requests to the API,
- the API returns the prediction.

This is the typical ML deployment pattern in production.

---

## 4. Les APIs utilisées par l’application

Il y a deux types de “API” dans ce projet :

### 4.1 API externe : Open-Meteo

Le fichier :

```python
webapp/core/weather.py
```

contient les appels HTTP vers Open-Meteo.

Il sert à :

- convertir une ville en coordonnées GPS,
- récupérer la météo de la localisation demandée,
- obtenir des variables comme :
  - température,
  - humidité,
  - point de rosée,
  - vitesse du vent,
  - direction du vent,
  - précipitations,
  - rayonnement solaire GHI.

#### API de géocodage

Open-Meteo géocoding :

```python
_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
```

Cette API permet de faire :

- "Paris" → latitude / longitude,
- "Berlin" → latitude / longitude.

Le code retourne une liste de résultats avec :

- nom,
- pays,
- région,
- latitude,
- longitude,
- altitude.

#### API météo

Open-Meteo forecast :

```python
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
```

Elle reçoit un couple latitude/longitude et retourne la météo pour une date/heure donnée.

Exemple d’info récupérée :

```python
{
  "GHI_RC_01": 650,
  "Temp_WS": 19.5,
  "RH_WS": 58,
  "DWP_WS": 10.2,
  "WS_WS": 3.8,
  "WD_WS": 180,
  "PREC_INT_WS": 0.1,
  "_timezone": "Europe/Berlin"
}
```

Le code dans `fetch_weather()` fait un appel HTTP, parse la réponse JSON, puis sélectionne la bonne heure la plus proche.

C’est important car la prédiction dépend du moment précis de la journée.

---

### 4.2 API interne du projet : modules Python comme services

Le projet n’a pas un vrai backend REST API avec FastAPI/Flask. En revanche, il a des fonctions Python qui jouent le rôle d’API internes.

Par exemple :

- `fetch_weather(...)`
- `geocode_city(...)`
- `compute_features(...)`
- `load_model()`
- `predict_par(...)`

Ces fonctions sont appelées par les pages Streamlit.

On peut donc les voir comme un “mini service backend”, même s’il n’y a pas de serveur HTTP séparé.

---

## 5. Comment le modèle XGBoost est lié à la page web

C’est la partie la plus importante.

La relation est la suivante :

1. L’utilisateur donne des informations.
2. L’application récupère la météo via Open-Meteo.
3. La page appelle le calcul de features.
4. Les features sont générées selon le même schéma que pendant l’entraînement.
5. Ces features sont passées au modèle XGBoost.
6. Le modèle retourne une valeur de PAR prédite.

### Schéma logique

```text
Utilisateur
   ↓
webapp/pages/...
   ↓
core/weather.py
   ↓
core/features.py
   ↓
core/predict.py
   ↓
XGBoost model
   ↓
PRÉDICTION PAR
```

---

## 6. Les features calculées avant d’envoyer les données au modèle

Le fichier :

```python
webapp/core/features.py
```

est le cœur du calcul des variables du modèle.

Il reproduit la même logique que dans les notebooks d’entraînement.

### 6.1 Pourquoi le calcul des features est crucial ?

Le modèle XGBoost ne reçoit pas directement des “raw values” comme :

- ville,
- date,
- pluie,
- température.

Il reçoit des variables transformées qui capturent la physique du problème.

C’est pour cela qu’on calcule :

- position solaire,
- angle solaire,
- airmass,
- indice de clarté,
- DNI,
- encodage de la direction du vent,
- variables dérivées comme la dépression de rosée.

---

### 6.2 Les 22 features du modèle

Le modèle attend un ensemble de colonnes. Les variables sont structurées en 4 groupes :

#### a) Variables météo de base

- GHI_RC_01
- Temp_WS
- RH_WS
- DWP_WS
- WS_WS
- WD_WS
- PREC_INT_WS
- PREC_DIFF_WS
- PREC_WS
- Temp_RC_merged
- Temp_RC_01

#### b) Variables solaires calculées avec pvlib

- zenith
- elevation
- airmass
- clearness_kt
- dni

#### c) Encodage de la direction du vent

- wind_sin
- wind_cos

#### d) Variables dérivées / “engineered”

- is_raining
- GHI_rolling_5min
- temp_diff
- dew_depression

L’ensemble est construit dans le DataFrame retourné par `compute_features()`.

---

### 6.3 Le rôle de pvlib

La librairie `pvlib` est utilisée pour calculer :

- la position du soleil,
- l’élévation du soleil,
- le zénith,
- l’air mass,
- le DNI,
- l’indice de clarté,
- l’irradiation extra-terrestre.

C’est la partie qui rend l’application globale, car cela fonctionne pour n’importe quelle latitude / longitude.

---

## 7. Le chargement du modèle XGBoost

Le fichier clé est :

```python
webapp/core/predict.py
```

C’est ici que le modèle est chargé et utilisé.

### 7.1 Fichiers du modèle

Les fichiers sont stockés dans le projet, souvent via Git LFS :

```text
data/results/xgboost_model_all_locations.pkl
data/processed/pkl_features_GradientBoosting/feature_names_all_locations.pkl
```

Le code fait :

```python
MODEL_PATH = _ROOT / "data" / "results" / "xgboost_model_all_locations.pkl"
FEATURE_NAMES_PATH = (
    _ROOT / "data" / "processed"
    / "pkl_features_GradientBoosting"
    / "feature_names_all_locations.pkl"
)
```

Cela signifie que la webapp ne réentraîne pas le modèle à chaque appel. Elle charge le modèle déjà entraîné qui a été sauvegardé auparavant.

---

### 7.2 Pourquoi un chargement “robuste” est nécessaire ?

Le modèle a été entraîné dans un notebook Jupyter, et il a été sérialisé avec `joblib`.

Le problème : le notebook a défini des classes et fonctions dans `__main__`.

Quand le modèle est chargé plus tard dans l’application, Python essaie de retrouver ces objets. S’ils ne sont pas présents, le chargement échoue.

C’est pour cela qu’on trouve dans `predict.py` :

```python
class XGBEnsemble:
    ...
```

puis :

```python
import __main__ as _main_module
if not hasattr(_main_module, "XGBEnsemble"):
    setattr(_main_module, "XGBEnsemble", XGBEnsemble)
```

et on ajoute aussi des stubs pour des fonctions de métriques personnalisées si nécessaire.

Cela évite les erreurs du type :

> AttributeError: module '__main__' has no attribute ...

Autrement dit, le code est fait pour “rétablir le contexte du notebook” avant le chargement du modèle.

---

### 7.3 La fonction de prédiction

La fonction clé est :

```python
def predict_par(features_df: pd.DataFrame) -> float:
```

Elle fait exactement ceci :

1. Charge le modèle avec `load_model()`.
2. Charge la liste des features attendues.
3. Construit un DataFrame avec les bonnes colonnes dans le bon ordre.
4. Appelle :

```python
pred = model.predict(X)[0]
```

5. retourne la valeur prédite.

Le résultat est ensuite converti en float, et if nécessaire, limité à zéro :

```python
return float(max(0.0, pred))
```

Cela évite des valeurs négatives impossibles pour une radiation.

---

## 8. Comment le modèle est lié aux pages Streamlit

La page “Normal Mode” demande des données utilisateur :

- ville ou coordonnées,
- date et heure,
- éventuellement une météo manuelle.

Ensuite elle fait :

```python
weather = fetch_weather(lat, lon, dt)
features_df, is_daytime = compute_features(lat, lon, alt, dt, weather, tz)
par_pred = predict_par(features_df)
```

Autrement dit :

- API météo -> donne des valeurs climatiques
- `compute_features` -> transforme en 22 features
- `predict_par` -> appelle le modèle XGBoost
- résultat affiché en interface utilisateur

La logique est donc totalement linéaire et très lisible.

---

## 9. La comparaison avec la méthode McCree

Le projet compare son modèle ML à la formule classique de McCree.

La formule de référence est :

```text
PAR = GHI × 0.45 × 4.57 ≈ GHI × 2.06
```

Le fichier :

```python
webapp/core/predict.py
```

contient :

```python
def mccree_estimate(ghi: float) -> float:
    return max(0.0, ghi * 2.06)
```

Donc :

- la formula McCree est très simple,
- mais elle est peu précise sous nuages, pluie, angle du soleil faible,
- le XGBoost est mieux adapté aux variables non linéaires.

C’est exactement ce que la webapp montre dans les pages d’analyse.

---

## 10. Le rôle de Dataset Upload

Le fichier :

```python
webapp/core/dataset.py
```

traite les fichiers CSV uploadés par l’utilisateur.

Il permet de :

- détecter les colonnes de date,
- détecter les colonnes de latitude/longitude,
- détecter les colonnes météo,
- faire des remplacements de noms de colonnes variés,
- nettoyer les valeurs,
- convertir les types,
- vérifier les données manquantes,
- calculer les features,
- prédire PAR sur plusieurs lignes de données.

### Exemple de logique

Le code essaie automatiquement de reconnaître plusieurs variantes de colonnes :

- `ghi`, `GHI`, `GHI_RC_01`, `irradiance`
- `temp`, `temperature`, `Temp_WS`
- `rh`, `humidity`, `RH_WS`
- etc.

C’est très utile car les datasets utilisateurs n’ont pas toujours les mêmes noms de colonnes.

---

## 11. Flux complet de données dans l’application

Voici le flux réel, de manière simple :

### Cas 1 : utilisateur entre une ville

```text
Page Streamlit
   ↓
Ville saisi
   ↓
Open-Meteo geocoding
   ↓
latitude / longitude récupérées
   ↓
Open-Meteo forecast
   ↓
Météo à l’heure demandée
   ↓
compute_features()
   ↓
22 features calculées
   ↓
XGBoost predict()
   ↓
Valeur PAR affichée
```

### Cas 2 : utilisateur upload un CSV

```text
CSV uploadé
   ↓
dataset.py
   ↓
colonnes détectées + nettoyage
   ↓
feature engineering pour chaque ligne
   ↓
modèle XGBoost appliqué batch
   ↓
résultats comparés/affichés
```

---

## 12. Comment l’application sait quelle feature est attendue par le modèle

Le modèle a été entraîné avec un ordre précis de colonnes.

Pour éviter toute erreur, le code charge :

```python
feature_names_all_locations.pkl
```

et ensuite :

```python
X = pd.DataFrame(index=features_df.index)
for col in feature_names:
    X[col] = features_df[col] if col in features_df.columns else 0.0
```

Cela signifie :

- on recalcule les features dans le bon ordre,
- on crée une matrice compatible avec le modèle,
- si une colonne manque, on met une valeur par défaut 0,
- le modèle reçoit toujours un tableau propre.

Sans cette étape, le modèle pourrait recevoir des colonnes dans le mauvais ordre et produire un mauvais résultat ou casser.

---

## 13. Pourquoi le projet est dit “global” et “portable”

Le système n’est pas limité à une ville précise.

Il fonctionne mondialement car :

- Open-Meteo couvre presque tout le monde,
- pvlib calcule la position du soleil avec latitude/longitude,
- le code de features est plus ou moins agnostique en fonction du lieu,
- le modèle a été entraîné sur plusieurs sites allemands, mais la logique reste utilisable ailleurs.

Le projet garde cependant une limite importante :

- le modèle a été entraîné sur des données de climat tempéré,
- donc les performances peuvent être moins fiables dans des climats très chauds, trop froids, ou très humides.

---

## 14. Comment le modèle est sauvegardé et relancé

Le modèle XGBoost est stocké en `.pkl` via `joblib`.

Cela veut dire que :

- on entraîne le modèle dans le notebook,
- on le sauvegarde dans un fichier `.pkl`,
- on le charge ensuite dans la webapp.

Ainsi, l’application n’a pas besoin de réapprendre le modèle à chaque ouverture.

C’est une architecture très classique en ML web :

- entrainement offline,
- sauvegarde du modèle,
- déploiement du modèle chargé dans l’application.

---

## 15. Ce que l’application “fait” concrètement pour l’utilisateur

Quand l’utilisateur clique sur un bouton de prédiction, la suite est généralement :

1. choisir la ville ou les coordonnées,
2. choisir ou confirmer la date/heure,
3. récupérer la météo,
4. calculer la position du soleil,
5. construire les features,
6. passer ces features au modèle,
7. afficher :
   - PAR prévue,
   - valeur McCree,
   - comparaison,
   - parfois les features importantes,
   - parfois les séries météo.

L’utilisateur voit un résultat final, mais il ne voit pas forcément toute la chaîne qui a conduit au résultat. La webapp est un wrapper simple autour de la logique ML.

---

## 16. Ce qui a été fait “from scratch” dans le projet

En résumé, le développement a été construit étape par étape :

1. créer une interface Streamlit,
2. ajouter la navigation des pages,
3. intégrer des formulaires utilisateur,
4. créer le module météo avec Open-Meteo,
5. créer le module de features avec pvlib,
6. sauvegarder le modèle XGBoost entraîné,
7. écrire la logique de chargement du modèle,
8. relier les features au modèle,
9. ajouter comparatif avec McCree,
10. créer les pages “expert” et “dataset upload”,
11. tester l’application localement.

C’est donc bien une architecture faite maison, sans framework front lourd ni backend externe complexe.

---

## 17. Dépendances principales

Le fichier :

```text
webapp/requirements_web.txt
```

contient les bibliothèques nécessaires, notamment :

- streamlit
- pandas
- numpy
- pvlib
- requests
- xgboost
- scikit-learn
- joblib

Ces dépendances permettent :

- d’afficher l’interface,
- de calculer les features,
- de faire les appels API,
- d’utiliser le modèle XGBoost.

---

## 18. Comment lancer la webapp localement

À partir du projet, on exécute :

```bash
cd webapp
pip install -r requirements_web.txt
streamlit run app.py
```

Ensuite l’application fonctionne sur :

```text
http://localhost:8501
```

Si le modèle n’est pas disponible, le site affiche un message de warning, car le fichier `.pkl` est probablement stocké via Git LFS.

Dans ce cas, le bon ordre est souvent :

```bash
git lfs pull
```

puis relancer la page.

---

## 19. Résumé ultra-simple

Si on résume tout en une phrase :

> La webapp est une application Streamlit qui récupère des données météo via Open-Meteo, transforme ces données en features solaires et météorologiques avec pvlib, puis envoie ces features à un modèle XGBoost déjà entraîné pour prédire la PAR.

Et le lien au modèle est très concret :

- le modèle est chargé dans `core/predict.py`,
- les features calculées sont mises dans le bon format,
- le modèle retourne une prédiction,
- le résultat est affiché dans les pages web.

---

## 20. Conclusion

La page web n’est pas seulement une interface jolie : elle est la couche qui connecte :

- les données utilisateur,
- les APIs externes,
- la physique solaire,
- le modèle d’apprentissage automatique,
- et le résultat final visible par l’utilisateur.

Donc, si on parle “d’architecture du projet”, la logique est :

- UI Streamlit,
- service météo,
- feature engineering,
- chargement du modèle,
- prédiction PAR,
- comparaison vs baseline.

C’est une architecture simple, claire, et très bien adaptée à un projet de data science/ML.

---

Si vous voulez, je peux maintenant vous faire une deuxième version encore plus “projet de mémoire” avec :

- un schéma architectural détaillé,
- une description “technicienne” du flux de données,
- et une version plus orientée “rapport de projet / documentation GitHub”.
