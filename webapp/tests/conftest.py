"""Pytest bootstrap for the web application's own suite.

Run from the repository root:   python -m pytest webapp/tests
or from inside webapp/:         python -m pytest tests

`webapp/` goes on the path so the modules import as `core.*`, exactly as they do
when Streamlit runs `app.py`. The repository root goes on too, because a few
tests read pipeline artefacts (the trained model, the notebook's recorded test
predictions) and skip cleanly when those are not present — on Streamlit Cloud,
or in CI without Git LFS.
"""
import sys
from pathlib import Path

WEBAPP = Path(__file__).resolve().parents[1]
ROOT = WEBAPP.parent
for p in (Path(__file__).resolve().parent, WEBAPP, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
