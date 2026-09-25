"""Every page renders, offline, without an exception — and shows what it claims to.

Runs the page scripts through Streamlit's AppTest with the network cut, so a
NameError, a bad import or a widget misuse is caught before a deploy. Nothing
here touches Open-Meteo.
"""
import re
from datetime import date, time as dtime

import pytest
import requests
import streamlit.delta_generator as _dg
from streamlit.testing.v1 import AppTest

from _common import ROOT
from core import predict as P
from core import weather as W

WEBAPP = ROOT / "webapp"
needs_model = pytest.mark.skipif(not P.is_model_available(), reason="model file not available (git lfs pull)")


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No page here needs the network; any attempt fails immediately.

    Two deliberate details:

    * ``_maybe_print_use_warning`` is stubbed out. It only prints the "run me
      with streamlit run" developer hint, but it calls ``inspect.stack()``,
      which resolves the real path of every frame. Under pytest the stack is
      ~50 frames deep and on Windows that costs minutes per page — the
      difference between a 2-second render and a 180-second AppTest timeout.
    * ``W.time.sleep`` is deliberately NOT stubbed. ``W.time`` is the global
      time module, and AppTest's own wait loop is ``while ...: sleep(0.001)``;
      replacing it turns that loop into a busy-spin that starves the script
      thread.
    """
    def _boom(*_a, **_k):
        raise requests.exceptions.ConnectionError("offline in tests")

    monkeypatch.setattr(W.requests, "get", _boom)
    monkeypatch.setattr(_dg, "_maybe_print_use_warning", lambda *_a, **_k: None)


def _run(page: str) -> AppTest:
    at = AppTest.from_file(str(WEBAPP / page), default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


# An inlined image is a wall of base64, and base64 contains every short string
# you might search for - "43M" really does appear inside the logo's payload.
_INLINED = re.compile(r"data:[^;\s\"']+;base64,[A-Za-z0-9+/=]+")


def _text(at: AppTest) -> str:
    return _INLINED.sub("<inlined image>", " ".join(m.value for m in at.markdown))


def test_home_shows_the_true_project_numbers():
    """The landing page used to advertise 4 stations and ~43M training rows.
    The project has 2 stations and 189,660 training rows."""
    text = _text(_run("home.py"))
    assert "Monitoring Sites" in text and "Training Rows" in text
    assert ">2</div>" in text and ">4</div>" not in text
    assert "190 K" in text and "43M" not in text


@needs_model
def test_normal_mode_renders_its_input_panel():
    at = _run("pages/1_Normal_Mode.py")
    assert any("Predict PAR" in b.label for b in at.button)
    assert "Ready to predict PAR" in _text(at)


@needs_model
def test_expert_mode_predicts_offline_with_error_band_and_export():
    """Pin a summer noon. The page defaults to *now*, so this test used to pass
    only in daylight: after sunset PAR is 0 and the error band is correctly
    hidden, which looked like a regression every evening."""
    at = _run("pages/2_Expert_Mode.py")
    at.date_input(key="e_date").set_value(date(2025, 6, 15))
    at.time_input(key="e_time").set_value(dtime(12, 0))
    at.run()
    next(b for b in at.button if "Predict PAR" in b.label).click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    text = _text(at)
    # Scoped wording: the MAE covers the model, not the readings it is given.
    assert "model error" in text                    # held-out MAE under the ML card
    assert "for the readings shown" in text
    assert "McCree Baseline" in text
    assert any("Export" in e.label for e in at.expander)


@needs_model
def test_expert_mode_flags_a_far_away_location():
    at = _run("pages/2_Expert_Mode.py")
    at.number_input(key="e_lat").set_value(-1.29)
    at.number_input(key="e_lon").set_value(36.82)          # Nairobi
    at.run()
    # The distance check is pure geometry, so it works with the network down;
    # naming the point needs the reverse lookup, which is covered separately.
    next(b for b in at.button if "Predict PAR" in b.label).click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    # Quietly: in a collapsed expander and in the export, not as a banner on
    # every prediction outside Germany.
    assert not any("extrapolation" in w.value for w in at.warning)
    assert any("Model domain" in e.label for e in at.expander)
    assert any("extrapolation" in c.value for c in at.caption)


def test_dataset_upload_renders_the_welcome_card():
    text = _text(_run("pages/3_Dataset_Upload.py"))
    assert "Ready to score your own data" in text
    assert "CSV, Excel, JSON or Parquet" in text


def test_the_timezone_follows_the_coordinates_when_the_lookup_answers(monkeypatch):
    """Expert Mode passes the zone straight to compute_features, so coordinates
    in Kenya with the field still on Europe/Berlin would place the sun wrongly."""
    from core import places as PL

    monkeypatch.setattr(PL, "reverse", lambda lat, lon: {
        "name": "Nairobi", "admin1": "Nairobi", "admin2": "", "country": "Kenya",
        "country_code": "KE", "latitude": lat, "longitude": lon, "elevation": None,
        "timezone": "Africa/Nairobi", "population": 0, "display": "Nairobi, Kenya",
    })
    at = _run("pages/2_Expert_Mode.py")
    assert at.session_state.e_tz == "Europe/Berlin"
    at.number_input(key="e_lat").set_value(-1.29)
    at.number_input(key="e_lon").set_value(36.82)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state.e_tz == "Africa/Nairobi"


def test_the_timezone_is_a_dropdown_of_real_zones():
    at = _run("pages/2_Expert_Mode.py")
    tz = at.selectbox(key="e_tz")
    assert tz.value == "Europe/Berlin"
    assert len(tz.options) > 400
    assert "Asia/Tokyo" in tz.options


# ── Appearance ───────────────────────────────────────────────────────────────

def test_the_palette_is_published_as_css_variables():
    """Every colour the app draws itself comes from these, so a page that fell
    back to a hard-coded hex would be the one that looks wrong in light mode."""
    from core import theme

    css = " ".join(m.value for m in _run("home.py").markdown if "--pp-" in m.value)
    for token in ("bg", "surface", "border", "text", "muted", "green", "orange"):
        assert f"--pp-{token}:" in css, token


def test_both_appearances_define_the_same_tokens():
    from core import theme

    assert set(theme.DARK) == set(theme.LIGHT)
    assert theme.DARK["bg"] != theme.LIGHT["bg"]
    assert theme.tokens("light")["bg"] == theme.LIGHT["bg"]
    assert theme.tokens("dark")["bg"] == theme.DARK["bg"]


def test_appearance_falls_back_to_dark_outside_a_browser():
    """st.context.theme is None under AppTest; the app must still render."""
    from core import theme

    assert theme.active() in ("dark", "light")
    assert theme.tokens()


def test_no_page_still_hard_codes_a_colour():
    import re

    hexes = re.compile(r"#[0-9a-fA-F]{6}\b")
    for name in ("home.py", "pages/1_Normal_Mode.py",
                 "pages/2_Expert_Mode.py", "pages/3_Dataset_Upload.py"):
        found = hexes.findall((WEBAPP / name).read_text(encoding="utf-8"))
        assert not found, f"{name} still hard-codes {sorted(set(found))}"


# ── Documentation ────────────────────────────────────────────────────────────

def test_documentation_renders_in_both_languages():
    from core.docs import LANGUAGES

    at = _run("pages/4_Documentation.py")
    english = _text(at)
    assert "What ParPredict does" in english
    assert english.count('class="doc-section"') == 12

    at.radio(key="doc_lang").set_value("de")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    german = _text(at)
    assert "Was ParPredict macht" in german
    assert "Globalstrahlung" in german
    assert set(LANGUAGES) == {"en", "de"}


def test_both_languages_cover_the_same_ground():
    """A section added to one language and forgotten in the other is the way
    bilingual docs rot."""
    from core.docs import sections

    en = [anchor for anchor, _, _ in sections("en")]
    de = [anchor for anchor, _, _ in sections("de")]
    assert en == de, f"sections differ: {set(en) ^ set(de)}"


def test_the_manual_quotes_the_model_not_a_remembered_number():
    """Every figure is filled in from model_card(), so retraining updates the
    prose instead of leaving it subtly wrong."""
    from core.docs import sections
    from core.predict import model_card

    card = model_card()
    body = " ".join(b for _, _, b in sections("en"))
    assert f"{card['test_mae']:.1f}" in body
    assert f"{card['baseline_mae']:.1f}" in body
    assert f"{card['n_test']:,}" in body
    assert "{" not in body and "}" not in body          # every placeholder filled

    german = " ".join(b for _, _, b in sections("de"))
    assert f"{card['test_mae']:.1f}".replace(".", ",") in german    # German decimals


# ── The documentation page follows the interface language ─────────────────────
# Two language controls on one screen is a trap: before this, a visitor could
# switch the interface to German and still be handed an English manual.


def _docs() -> AppTest:
    return AppTest.from_file(
        str(WEBAPP / "pages" / "4_Documentation.py"), default_timeout=120
    )


def test_documentation_opens_in_the_interface_language():
    at = _docs()
    at.session_state["pp_lang"] = "de"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["doc_lang"] == "de"


def test_documentation_page_switch_survives_a_rerun():
    """A choice made on the page is not undone by the next interaction."""
    at = _docs()
    at.session_state["pp_lang"] = "en"
    at.run()
    assert at.session_state["doc_lang"] == "en"
    at.radio(key="doc_lang").set_value("de").run()
    assert at.session_state["doc_lang"] == "de"
    at.run()                                  # nothing touched; must stay German
    assert at.session_state["doc_lang"] == "de"


def test_documentation_follows_a_later_sidebar_change():
    at = _docs()
    at.session_state["pp_lang"] = "en"
    at.run()
    assert at.session_state["doc_lang"] == "en"
    at.session_state["pp_lang"] = "de"        # as the sidebar switcher would
    at.run()
    assert at.session_state["doc_lang"] == "de"
