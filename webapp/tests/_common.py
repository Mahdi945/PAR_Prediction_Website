"""Constants shared by the web application's test modules."""
from pathlib import Path

# The repository root: webapp/tests/ -> webapp/ -> root. A few tests read
# pipeline artefacts from data/ and skip when they are absent.
ROOT = Path(__file__).resolve().parents[2]

# The 15 features the model is trained on, in training order.
# Mirrors FEATURE_COLS in notebooks/Mahdi/5_stratification_All_Files.ipynb.
MODEL_FEATURES = [
    "GHI_RC_01", "Temp_WS", "RH_WS", "DWP_WS", "WS_WS",
    "PREC_INT_WS", "PREC_DIFF_WS", "Temp_RC_merged",
    "zenith", "elevation", "airmass", "clearness_kt", "dni",
    "wind_sin", "wind_cos",
]
