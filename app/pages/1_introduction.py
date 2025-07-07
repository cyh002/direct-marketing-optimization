"""Introduction page."""
from __future__ import annotations

from pathlib import Path
import streamlit as st


def read_summary(lines: int = 20) -> str:
    """Return the first lines of the project README."""
    readme = Path(__file__).resolve().parents[2] / "README.md"
    content = readme.read_text(encoding="utf-8").splitlines()
    return "\n".join(content[:lines])


st.markdown(read_summary())
