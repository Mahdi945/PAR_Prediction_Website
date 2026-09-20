"""
core/theme.py
─────────────
One palette, two appearances.

Streamlit themes its own chrome from .streamlit/config.toml, which now carries
a [theme.light] and a [theme.dark] block. Everything the app draws itself - the
mode cards, the PAR card, the weather pills, the charts - is styled by hand,
so it has to follow the same switch. That is what this module is for.

    active()      → "dark" | "light"      what the visitor is actually seeing
    tokens()      → dict                  the palette for that appearance
    inject()      → writes :root { --pp-* } once per page
    switcher()    → the sidebar control that changes appearance
    plotly_layout() → chart colours that match

Use the variables, not literals: `color: var(--pp-muted)` rather than
`color: #8892b0`. A hex written into a page is a colour that cannot follow the
theme, and it will be the one that looks wrong in light mode.

On contrast: the brand green #2ECC71 is 2.0:1 on white, which fails WCAG AA for
text. So each accent has two tokens - a *fill* for borders, bars and gauges
(where contrast rules do not apply) and a *text* variant dark enough to read.
In dark mode the two are usually the same value; in light mode they are not.
"""

from __future__ import annotations

import streamlit as st

__all__ = ["active", "tokens", "inject", "switcher", "plotly_layout", "is_dark"]

# ── The two palettes ─────────────────────────────────────────────────────────

DARK: dict[str, str] = {
    "bg":            "#0F1117",
    "surface":       "#1A1D2E",
    "surface-2":     "#141726",
    "border":        "#2A2D3E",
    "border-soft":   "#22253a",
    "text":          "#E8EAF6",
    "text-strong":   "#FFFFFF",
    "muted":         "#8892B0",
    "faint":         "#5A6070",
    # accents: fill first, then a text-safe variant
    "green":         "#2ECC71",
    "green-text":    "#2ECC71",
    "green-soft":    "rgba(46,204,113,.12)",
    "orange":        "#F39C12",
    "orange-text":   "#F39C12",
    "orange-soft":   "rgba(243,156,18,.12)",
    "blue":          "#3498DB",
    "blue-text":     "#3498DB",
    "red":           "#E74C3C",
    "red-text":      "#E74C3C",
    "grey":          "#6C757D",
    "yellow":        "#F7C948",
    "warn-bg":       "#2D1A00",
    "warn-text":     "#F7C948",
    # card washes
    "card-green":    "linear-gradient(135deg,#0D2B1A 0%,#0F1117 100%)",
    "card-orange":   "linear-gradient(135deg,#2D1A00 0%,#0F1117 100%)",
    "card-plain":    "#1A1D2E",
    # charts
    "chart-plot":    "rgba(26,29,46,0.7)",
    "chart-grid":    "rgba(255,255,255,0.06)",
    "chart-line":    "#FFFFFF",
    "chart-step":    "rgba(52,73,94,.30)",
}

LIGHT: dict[str, str] = {
    "bg":            "#F6F8F5",
    "surface":       "#FFFFFF",
    "surface-2":     "#FAFBF9",
    "border":        "#DFE4DC",
    "border-soft":   "#EAEEE8",
    "text":          "#16181D",
    "text-strong":   "#0B0D12",
    "muted":         "#5B6470",
    "faint":         "#8A93A0",
    "green":         "#22A65B",
    "green-text":    "#15803D",
    "green-soft":    "rgba(34,166,91,.10)",
    "orange":        "#E08A00",
    "orange-text":   "#B45309",
    "orange-soft":   "rgba(224,138,0,.10)",
    "blue":          "#2B7FC0",
    "blue-text":     "#1D5F91",
    "red":           "#D64535",
    "red-text":      "#B32D1F",
    "grey":          "#6C757D",
    "yellow":        "#C08A00",
    "warn-bg":       "#FEF3E2",
    "warn-text":     "#92400E",
    "card-green":    "linear-gradient(135deg,#EAF7EF 0%,#FFFFFF 100%)",
    "card-orange":   "linear-gradient(135deg,#FDF3E3 0%,#FFFFFF 100%)",
    "card-plain":    "#FFFFFF",
    "chart-plot":    "rgba(255,255,255,0.85)",
    "chart-grid":    "rgba(20,24,30,0.10)",
    "chart-line":    "#16181D",
    "chart-step":    "rgba(20,24,30,0.06)",
}

_DEFAULT = "dark"


# ── Which appearance is showing ──────────────────────────────────────────────

def active() -> str:
    """"dark" or "light", as the browser currently has it.

    Streamlit infers this from the rendered background colour. It is None
    outside a browser (AppTest, a plain script), so the default stands in.
    """
    try:
        kind = st.context.theme.type
    except Exception:
        kind = None
    return kind if kind in ("dark", "light") else _DEFAULT


def is_dark() -> bool:
    return active() == "dark"


def tokens(appearance: str | None = None) -> dict[str, str]:
    return dict(DARK if (appearance or active()) == "dark" else LIGHT)


# ── Injecting the variables ──────────────────────────────────────────────────

def inject() -> None:
    """Publish the palette as CSS variables. Call once, before any page CSS."""
    t = tokens()
    body = "\n".join(f"    --pp-{k}: {v};" for k, v in t.items())
    st.markdown(
        f"<style>\n:root {{\n{body}\n}}\n</style>",
        unsafe_allow_html=True,
    )


# ── The switch ───────────────────────────────────────────────────────────────

# Streamlit keeps the visitor's choice in localStorage under
# `stActiveTheme-<pathname>-v2`, holding a JSON string: "Light", "Dark" or
# "System". Writing it and reloading is what its own Settings menu does. The
# key is per path, so every page of the app gets it or the choice would be
# forgotten on navigation.
_PATHS = ["/", "/Normal_Mode", "/Expert_Mode", "/Dataset_Upload"]

_SET_THEME_JS = """
<script>
(function () {
  const store = window.parent.localStorage;
  const want = %s;
  const paths = %s;
  const here = window.parent.location.pathname;
  const keyFor = p => "stActiveTheme-" + p + "-v2";

  // Streamlit reads this entry with JSON.parse on boot, so it has to hold a
  // JSON document: the six characters "Dark", quotes included. Writing the
  // bare word threw an uncaught SyntaxError inside Streamlit's own bundle,
  // the app never rendered, and because the value persisted every reload
  // failed the same way - a spinner that could only be cleared by hand.
  const encoded = JSON.stringify(want);

  let changed = false;
  for (const p of paths.concat([here])) {
    const k = keyFor(p);
    const current = store.getItem(k);
    // Also repair anything unparseable left behind by an earlier version.
    let broken = false;
    if (current !== null) {
      try { JSON.parse(current); } catch (e) { broken = true; }
    }
    if (current !== encoded || broken) { store.setItem(k, encoded); changed = true; }
  }
  if (changed) { window.parent.location.reload(); }
})();
</script>
"""


def switcher(*, key: str = "pp_theme", location=None) -> None:
    """A light/dark control that drives Streamlit's own theme.

    Rendered in the sidebar by default. Selecting an appearance writes the same
    localStorage entry Streamlit's Settings menu writes, then reloads, so the
    whole app - its widgets as well as this app's cards - changes together.
    """
    import json

    import streamlit.components.v1 as components

    target = location if location is not None else st.sidebar
    now = active()
    labels = {"light": "☀️  Light", "dark": "🌙  Dark"}
    options = ["light", "dark"]

    choice = target.radio(
        "Appearance",
        options,
        index=options.index(now),
        format_func=lambda v: labels[v],
        key=key,
        horizontal=True,
        help="Changes the whole app, and is remembered in this browser.",
    )

    if choice != now:
        # components.html runs in an iframe; reach out to the parent document.
        components.html(
            _SET_THEME_JS % (json.dumps(choice.capitalize()), json.dumps(_PATHS)),
            height=0,
        )


# ── Charts ───────────────────────────────────────────────────────────────────

def plotly_layout(**overrides) -> dict:
    """Layout defaults so a chart sits on the page instead of on top of it."""
    t = tokens()
    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": t["chart-plot"],
        "font": {"color": t["text"]},
        "xaxis": {"gridcolor": t["chart-grid"]},
        "yaxis": {"gridcolor": t["chart-grid"]},
        "legend": {"bgcolor": "rgba(0,0,0,0)"},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(layout.get(k), dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout
