"""Raw HTML must survive an interpolation that comes out empty.

`unsafe_allow_html=True` does not turn Markdown off. CommonMark ends an HTML
block at the first blank line, and Streamlit dedents the body before parsing it,
so a line holding nothing but an optional interpolation becomes a blank line the
moment that value is "". Everything after it is Markdown again — and at four
spaces of indentation, that is a code block.

This is what put `<div class="par-cat">…` on screen as text in the prediction
card at night, when the "typical error" line is deliberately dropped.
"""
import ast
import textwrap

import pytest

from _common import ROOT
from core.html import block

WEBAPP = ROOT / "webapp"
PAGES = [p for p in WEBAPP.rglob("*.py")
         if "__pycache__" not in p.parts and "tests" not in p.parts]


# ── The helper ────────────────────────────────────────────────────────────────

def test_block_leaves_no_blank_line_for_an_empty_fragment():
    out = block(f"""
        <div class="card">
            <div class="value">0.0</div>
            {""}
            <div class="cat">Night</div>
        </div>
        """)
    assert "\n\n" not in out
    assert not any(line.startswith(" ") for line in out.splitlines())


def test_block_keeps_the_text_and_the_order():
    out = block("""
        <div>
            first
            second
        </div>
        """)
    assert out == "<div>\nfirst\nsecond\n</div>"


def test_block_output_is_one_html_block_in_commonmark():
    """The real check: a CommonMark parser must not produce a code block."""
    md = pytest.importorskip("markdown_it").MarkdownIt("commonmark")
    card = """
        <div class="par-card">
            <div class="par-big">0.0</div>
            {frag}
            <div class="par-cat">Night</div>
            <hr>
        </div>
        """
    raw = md.render(textwrap.dedent(card.format(frag="")))
    assert "<code>" in raw, "expected the unguarded version to break — if not, the hazard changed"

    safe = md.render(block(card.format(frag="")))
    assert "<code>" not in safe and "<pre>" not in safe


# ── Every page, not just the two that were caught ─────────────────────────────

def _lone_interpolation_lines(source: str) -> list[str]:
    return [
        ln.strip() for ln in source.splitlines()
        if ln.strip().startswith("{") and ln.strip().endswith("}") and ln.startswith(" ")
    ]


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_optional_html_fragments_go_through_block(path):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    offenders = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not any(k.arg == "unsafe_allow_html" for k in node.keywords):
            continue
        if not node.args:
            continue
        body = node.args[0]

        # Already guarded: st.markdown(block(f"..."), unsafe_allow_html=True)
        if isinstance(body, ast.Call) and getattr(body.func, "id", "") == "block":
            continue

        segment = ast.get_source_segment(text, body) or ""
        if "\n" not in segment:
            continue
        lone = _lone_interpolation_lines(segment)
        if lone:
            offenders.append(f"line {node.lineno}: {lone}")

    assert not offenders, (
        f"{path.name} interpolates a whole line inside raw HTML without block(): "
        + "; ".join(offenders)
        + ". If that value is ever empty the rest of the block renders as source."
    )
