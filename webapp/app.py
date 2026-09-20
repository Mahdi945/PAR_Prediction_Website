"""
app.py  -  ParPredict - Navigation Router
Entry point. Run with:  python -m streamlit run app.py
"""

import streamlit as st

from core import theme

# ── Global page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ParPredict",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The palette goes in before anything is drawn, so every page's CSS can use it.
theme.inject()

# ── Sidebar: logo + title pinned above nav links via CSS ::before ─────────────
st.markdown("""
<style>
/* Pin ParPredict branding at the very top of the sidebar */
[data-testid="stSidebarNav"]::before {
    content: "🌱  ParPredict";
    display: block;
    font-size: 1.25rem;
    font-weight: 900;
    color: var(--pp-text-strong);
    padding: 1.4rem 1rem 1rem 1.4rem;
    letter-spacing: .4px;
    border-bottom: 1px solid var(--pp-border);
    margin-bottom: .4rem;
}

/* Hide Streamlit's own toolbar actions. config.toml already sets
   toolbarMode = "minimal"; this covers the case where the host renders the
   row anyway. The running/stopped status indicator is left alone. */
[data-testid="stToolbarActions"] { display: none !important; }
[data-testid="stMainMenu"]       { display: none !important; }

/* Remove default top padding so logo sits flush at top */
[data-testid="stSidebarNav"] {
    padding-top: 0 !important;
}

/* Documentation is reference material, not a fourth thing to do, so it sits
   below its own rule rather than in the run of working pages. */
[data-testid="stSidebarNavItems"] li:last-child {
    margin-top: .5rem;
    padding-top: .5rem;
    border-top: 1px solid var(--pp-border);
}

/* Close the gap between the nav and the Appearance control under it. */
[data-testid="stSidebarNavSeparator"] { margin-bottom: 0 !important; }
[data-testid="stSidebarUserContent"] { padding-top: .4rem !important; }

/* Style each nav link cleanly */
[data-testid="stSidebarNavLink"] {
    border-radius: 8px !important;
    margin: 2px 6px !important;
}
[data-testid="stSidebarNavLink"]:hover {
    background-color: var(--pp-green-soft) !important;
}
[data-testid="stSidebarNavLink"][aria-selected="true"] {
    background-color: var(--pp-green-soft) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────────────────────
pg = st.navigation([
    st.Page("home.py",                  title="Home",            icon="🏠"),
    st.Page("pages/1_Normal_Mode.py",   title="Normal Mode",    icon="🌱"),
    st.Page("pages/2_Expert_Mode.py",   title="Expert Mode",    icon="⚙️"),
    st.Page("pages/3_Dataset_Upload.py", title="Dataset Upload", icon="📊"),
    st.Page("pages/4_Documentation.py", title="Documentation",  icon="📖"),
])

# ── Appearance ────────────────────────────────────────────────────────────────
# Straight under the navigation. Streamlit already draws a rule below the nav
# (stSidebarNavSeparator), so adding st.divider() here put two lines and a gap
# between the pages and this control.
with st.sidebar:
    theme.switcher()

pg.run()
