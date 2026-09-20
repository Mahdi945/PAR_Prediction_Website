"""
core/html.py  –  one safe way to hand raw HTML to Streamlit.

`st.markdown(..., unsafe_allow_html=True)` does not switch Markdown off. The
body is still parsed as CommonMark, and CommonMark ends an HTML block at the
first blank line. Whatever follows is ordinary Markdown again — and if it is
indented four spaces or more, it becomes a code block and the tags are shown to
the visitor as text.

That is easy to write by accident, because Streamlit dedents the body first.
A multi-line f-string like

    st.markdown(f'''
        <div class="card">
            <div class="value">{par:.1f}</div>
            {optional_note}
            <div class="label">{label}</div>
        </div>
        ''', unsafe_allow_html=True)

renders correctly every time `optional_note` has content. The day it is empty,
that line holds only the indentation, dedent turns it into a blank line, the
HTML block ends, and every tag after it appears on screen as source. It happened
here at night, when the "typical error" line is deliberately dropped: the
prediction card rendered as raw markup from `<div class="par-cat">` onwards.

`block()` removes the conditions for it: every line is flattened to column zero
and empty lines are dropped, so there is no blank line to end the block and no
indentation to start a code block. Interpolations become free to be empty, or
to span several lines, without the caller having to think about it.
"""

from __future__ import annotations


def block(markup: str) -> str:
    """Flatten raw HTML so no interpolated value can break it out of its block.

    Newlines between the remaining lines are kept: HTML collapses whitespace, so
    they change nothing visually, and they keep words apart where an element's
    text was wrapped across lines in the source.
    """
    return "\n".join(
        stripped for stripped in (line.strip() for line in markup.splitlines())
        if stripped
    )
