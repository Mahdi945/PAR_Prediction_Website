"""German for the interface, and the two ways it silently goes wrong.

The catalogue is keyed by English source text, so anything missing falls back
to readable English. What must never happen is a translation reaching something
structural — a file path, a column name — or a wrapper translating the wrong
argument.
"""
import pytest
import streamlit as st

from core import i18n


@pytest.fixture(autouse=True)
def _german():
    st.session_state[i18n._KEY] = "de"
    yield
    st.session_state.pop(i18n._KEY, None)


# ── Translating ──────────────────────────────────────────────────────────────

def test_a_known_label_becomes_german():
    assert i18n.translate("Latitude (°N)") == "Breitengrad (°N)"
    assert i18n.translate("Dataset Upload") == "Datei-Upload"


def test_unknown_text_stays_readable_english():
    """A missing entry must never surface as a key or an empty string."""
    odd = "Some sentence that was never catalogued at all"
    assert i18n.translate(odd) == odd


def test_english_is_returned_untouched():
    st.session_state[i18n._KEY] = "en"
    assert i18n.translate("Latitude (°N)") == "Latitude (°N)"


def test_prose_inside_markup_is_translated():
    out = i18n.translate('<div class="x">Ready to predict PAR</div>')
    assert "Bereit für die PAR-Vorhersage" in out
    assert '<div class="x">' in out          # the markup is left alone


def test_a_phrase_broken_across_lines_still_matches():
    """Prose in the HTML blocks is wrapped and indented, so the literal string
    never appears contiguously."""
    wrapped = "Search for any city or enter\n            coordinates."
    assert "Nach einer Stadt suchen" in i18n.translate(wrapped)


# ── What must never be translated ────────────────────────────────────────────

def test_a_page_path_is_never_rewritten():
    """"Documentation" is in the catalogue; replacing it inside this path would
    break st.page_link and the page would stop resolving."""
    assert i18n.translate("pages/4_Documentation.py") == "pages/4_Documentation.py"


def test_identifiers_and_column_names_survive():
    for name in ("GHI_RC_01", "model_prediction", "Temp_RC_merged", "e_tz"):
        assert i18n.translate(name) == name


def test_non_strings_pass_through():
    for value in (None, 42, 3.5, ["a"], {"k": "v"}):
        assert i18n.translate(value) is value or i18n.translate(value) == value


# ── The wrapper ──────────────────────────────────────────────────────────────

def test_class_methods_skip_self():
    """Patching a class hands the wrapper `self` first. Getting the offset wrong
    is silent: st.markdown() translates while col.markdown() quietly does not."""
    seen = {}

    def fake_method(self, label, **kw):
        seen["label"] = label
        return label

    wrapped = i18n._wrap(fake_method, offset=1)
    wrapped(object(), "Dataset Upload")
    assert seen["label"] == "Datei-Upload"


def test_module_functions_translate_the_first_argument():
    seen = {}

    def fake(label, **kw):
        seen["label"] = label
        return label

    i18n._wrap(fake)("Dataset Upload")
    assert seen["label"] == "Datei-Upload"


def test_help_and_label_keywords_are_translated():
    seen = {}

    def fake(*a, **kw):
        seen.update(kw)

    i18n._wrap(fake)(label="Appearance", help="Changes the whole app, and is remembered in this browser.")
    assert seen["label"] == "Darstellung"
    assert seen["help"].startswith("Ändert die gesamte Anwendung")


def test_install_is_idempotent():
    from streamlit.delta_generator import DeltaGenerator

    i18n.install()
    once = DeltaGenerator.markdown
    i18n.install()
    assert DeltaGenerator.markdown is once, "install() wrapped an already wrapped function"


# ── The catalogue itself ─────────────────────────────────────────────────────

def test_catalogue_is_sane():
    for src, dst in i18n.CATALOG.items():
        assert src and dst, f"empty entry for {src!r}"
        assert src != dst, f"{src!r} is not translated"


def test_language_follows_the_browser_when_nothing_is_chosen(monkeypatch):
    st.session_state.pop(i18n._KEY, None)
    monkeypatch.setattr(i18n, "_browser_language", lambda: "de")
    assert i18n.language() == "de"
    monkeypatch.setattr(i18n, "_browser_language", lambda: "en")
    assert i18n.language() == "en"

