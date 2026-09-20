"""
pages/4_Documentation.py  –  ParPredict · Documentation / Dokumentation
────────────────────────────────────────────────────────
The user manual, in English and German. The text lives in core/docs.py; this
page only chooses a language and lays it out.

Deliberately one page rather than a site-wide translation: the interface stays
in English, and everything a visitor might need explaining is here in both.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from core import theme
from core.docs import LANGUAGES, TITLE, INTRO, CONTENTS, sections

T = theme.tokens()

st.set_page_config(
    page_title="Documentation · ParPredict",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject()
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
.doc-head {
    font-size: .78rem; font-weight: 700; color: var(--pp-green-text);
    text-transform: uppercase; letter-spacing: 1.5px;
    border-left: 3px solid var(--pp-green); padding-left: .55rem;
    margin: 0 0 .6rem 0;
}
/* The contents list: quiet, and sticky so it stays with you on a long read. */
.doc-toc {
    background: var(--pp-surface); border: 1px solid var(--pp-border);
    border-radius: 14px; padding: 1rem 1.1rem;
    position: sticky; top: 3.2rem;
}
.doc-toc a {
    display: block; padding: .28rem 0; font-size: .89rem;
    color: var(--pp-muted); text-decoration: none; line-height: 1.45;
}
.doc-toc a:hover { color: var(--pp-green-text); }
.doc-toc .n { color: var(--pp-faint); font-size: .78rem; margin-right: .45rem; }

.doc-section { scroll-margin-top: 4rem; margin-bottom: 2.2rem; }
.doc-section h2 {
    font-size: 1.35rem; font-weight: 800; color: var(--pp-text-strong);
    margin: 0 0 .1rem 0;
}
.doc-rule { height: 2px; width: 46px; background: var(--pp-green);
            border-radius: 2px; margin: .45rem 0 .9rem 0; }

@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .doc-toc { position: static; margin-bottom: 1.2rem; }
    div[data-testid="column"] > div:first-child { min-width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Language ──────────────────────────────────────────────────────────────────
if "doc_lang" not in st.session_state:
    st.session_state.doc_lang = "en"

_head, _pick = st.columns([3, 1.1], gap="medium")
with _head:
    st.markdown(f"# 📖 {TITLE[st.session_state.doc_lang]}")
with _pick:
    lang = st.radio(
        "Language / Sprache",
        list(LANGUAGES),
        format_func=lambda c: LANGUAGES[c],
        key="doc_lang",
        horizontal=True,
    )

st.caption(INTRO[lang])
st.divider()

# ── Contents + body ───────────────────────────────────────────────────────────
blocks = sections(lang)

toc, body = st.columns([1, 3], gap="large")

with toc:
    links = "".join(
        f'<a href="#{anchor}"><span class="n">{i:02d}</span>{title}</a>'
        for i, (anchor, title, _) in enumerate(blocks, 1)
    )
    st.markdown(
        f'<div class="doc-head">{CONTENTS[lang]}</div>'
        f'<div class="doc-toc">{links}</div>',
        unsafe_allow_html=True,
    )

with body:
    for i, (anchor, title, text) in enumerate(blocks, 1):
        st.markdown(
            f'<div class="doc-section" id="{anchor}">'
            f'<h2>{i:02d} · {title}</h2><div class="doc-rule"></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(text)
        st.markdown("<br>", unsafe_allow_html=True)
