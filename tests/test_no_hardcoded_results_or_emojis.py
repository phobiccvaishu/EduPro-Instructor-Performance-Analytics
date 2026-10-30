"""
tests.test_no_hardcoded_results_or_emojis
=========================================
Static analysis and AST inspection test suite:
- Asserts zero Unicode emojis exist in app.py, edupro/ui, edupro/insights.py, or edupro/ui_filters.py.
- Asserts no hardcoded analytical result numbers (e.g. '3.125', '0.598', '73.1', '4.234', '3.098')
  appear inside string literals in app.py, edupro/ui, or edupro/insights.py.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT_DIR = Path(__file__).resolve().parent.parent

EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]"
)

# Known analytical output strings that must never be hard-coded inside UI or insight strings
FORBIDDEN_RESULT_STRINGS = [
    "3.125",
    "0.598",
    "73.1",
    "4.234",
    "3.0978",
    "0.635",
    "3.72",
    "3.6667",
    "3.16",
    "3.253",
    "0.816",
]


def get_target_files() -> list[Path]:
    """Retrieve all Python files in app.py, edupro/ui/, edupro/insights.py, and edupro/ui_filters.py."""
    files = [
        ROOT_DIR / "app.py",
        ROOT_DIR / "edupro" / "insights.py",
        ROOT_DIR / "edupro" / "ui_filters.py",
    ]
    ui_dir = ROOT_DIR / "edupro" / "ui"
    if ui_dir.exists():
        files.extend(ui_dir.rglob("*.py"))
    return [f for f in files if f.exists()]


def test_zero_emojis_in_source_code():
    """Verify that no Unicode emojis exist anywhere in source code files."""
    for p in get_target_files():
        content = p.read_text(encoding="utf-8")
        matches = EMOJI_PATTERN.findall(content)
        assert len(matches) == 0, f"Found emoji(s) {matches} in {p.relative_to(ROOT_DIR)}"


def test_zero_hardcoded_result_numbers_in_string_literals():
    """Parse AST of UI files and verify no string literal contains hardcoded analytical results."""
    for p in get_target_files():
        # Skip insights.py docstrings if any, but parse all string constants
        content = p.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(p))

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                str_val = node.value
                for forbidden in FORBIDDEN_RESULT_STRINGS:
                    # Allow slider bounds or axis ranges if explicit, but forbid result text
                    assert forbidden not in str_val, (
                        f"Hardcoded analytical result '{forbidden}' found in string literal inside "
                        f"{p.relative_to(ROOT_DIR)} at line {node.lineno}: '{str_val}'"
                    )


def test_app_rendered_text_contains_zero_emojis():
    """Verify that AppTest rendered text contains zero Unicode emojis."""
    app_path = str(ROOT_DIR / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception

    all_rendered_text = " ".join(
        [str(m.value) for m in at.markdown]
        + [str(c.value) for c in at.caption]
        + [str(t.value) for t in at.text]
        + [str(m.label) for m in at.metric]
    )
    emoji_matches = EMOJI_PATTERN.findall(all_rendered_text)
    assert len(emoji_matches) == 0, f"Rendered UI text contains emojis: {emoji_matches}"
