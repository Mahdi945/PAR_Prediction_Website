"""
app.py  -  ParPredict - Navigation Router
Entry point. Run with:  python -m streamlit run app.py
"""

import streamlit as st

# ── Global page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ParPredict",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: logo + title pinned above nav links via CSS ::before ─────────────
st.markdown("""
<style>
/* Pin ParPredict branding at the very top of the sidebar */
[data-testid="stSidebarNav"]::before {
    content: "🌱  ParPredict";
    display: block;
    font-size: 1.25rem;
    font-weight: 900;
    color: #ffffff;
    padding: 1.4rem 1rem 1rem 1.4rem;
    letter-spacing: .4px;
    border-bottom: 1px solid #2a2d3e;
    margin-bottom: .4rem;
}

/* Remove default top padding so logo sits flush at top */
[data-testid="stSidebarNav"] {
    padding-top: 0 !important;
}

/* Style each nav link cleanly */
[data-testid="stSidebarNavLink"] {
    border-radius: 8px !important;
    margin: 2px 6px !important;
}
[data-testid="stSidebarNavLink"]:hover {
    background-color: rgba(46,204,113,.12) !important;
}
[data-testid="stSidebarNavLink"][aria-selected="true"] {
    background-color: rgba(46,204,113,.18) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────────────────────
pg = st.navigation([
    st.Page("home.py",                  title="Home",            icon="🏠"),
    st.Page("pages/1_Normal_Mode.py",   title="Normal Mode",    icon="🌱"),
    st.Page("pages/2_Expert_Mode.py",   title="Expert Mode",    icon="⚙️"),
    st.Page("pages/3_Dataset_Upload.py", title="Dataset Upload", icon="📊"),
])
pg.run()
