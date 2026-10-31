"""
tests.test_no_hardcoded_results_or_emojis
=========================================
Static analysis and AST inspection test suite:
- Asserts zero Unicode emojis exist in source code or rendered UI.
- Asserts no hardcoded analytical result numbers (e.g. '3.125', '0.598', '73.1', '4.234', '3.098')
  appear inside string literals.
- Enforces extended banned-word scan across all UI and insight source code.
- Enforces numeric literal AST scanner with documented allowlist.
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

# Extended banned vocabulary (causal, superlative, editorial, promotional)
EXTENDED_BANNED_WORDS = [
    r"\bcauses?\b",
    r"\bcaused\b",
    r"\bcausing\b",
    r"\bdrives?\b",
    r"\bdriven\b",
    r"\bdriving\b",
    r"\bdue to\b",
    r"\bleads? to\b",
    r"\bleading to\b",
    r"\bresults? in\b",
    r"\bresulting in\b",
    r"\bproves?\b",
    r"\bproven\b",
    r"\bproving\b",
    r"\bentirely\b",
    r"\battract\w*\b",
    r"\bcapture\w*\b",
    r"\bdemonstrat\w*\b",
    r"\bconfirms?\b",
    r"\bestablishes?\b",
    r"\breflects?\b",
    r"\bsuffer\w*\b",
    r"\bboost\w*\b",
    r"\bbecause\b",
    r"\btherefore\b",
]

# Documented allowlist for numeric literals appearing in UI code
# (e.g. CSS layout measurements, standard statistical alphas, format strings)
NUMERIC_LITERAL_ALLOWLIST = {
    # Alphas and significance cutoffs
    "0.05",
    "0.01",
    "0.001",
    "0.0001",
    # CSS dimensions and font sizes
    "0.15rem",
    "0.4rem",
    "0.5rem",
    "0.6rem",
    "0.75rem",
    "0.80rem",
    "0.85rem",
    "1.15rem",
    "1.25rem",
    "1.45rem",
    "1.55rem",
    "0.01em",
    "0.02em",
    "100%",
    "1.45",
}


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
        content = p.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(p))

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                str_val = node.value
                for forbidden in FORBIDDEN_RESULT_STRINGS:
                    assert forbidden not in str_val, (
                        f"Hardcoded analytical result '{forbidden}' found in string literal inside "
                        f"{p.relative_to(ROOT_DIR)} at line {node.lineno}: '{str_val}'"
                    )


def get_docstring_nodes(tree: ast.AST) -> set[ast.AST]:
    """Identify AST Constant nodes that serve as module, class, or function docstrings."""
    doc_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                val_node = node.body[0].value
                if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
                    doc_nodes.add(val_node)
    return doc_nodes


def test_extended_banned_words_in_source_code():
    """Verify that extended banned vocabulary does not appear in UI or insight code outside docstrings."""
    for p in get_target_files():
        content = p.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(p))
        doc_nodes = get_docstring_nodes(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node in doc_nodes:
                    continue
                str_val = node.value
                for pattern in EXTENDED_BANNED_WORDS:
                    match = re.search(pattern, str_val, re.IGNORECASE)
                    assert not match, (
                        f"Banned word matching '{pattern}' found in {p.relative_to(ROOT_DIR)} "
                        f"at line {node.lineno}: '{str_val}'"
                    )


def test_numeric_literal_ast_scanner():
    """Verify that any numeric literal in string constants belongs to the documented allowlist."""
    num_re = re.compile(r"\b\d+\.\d{2,}\b|\b\d+(\.\d+)?%\b")
    for p in get_target_files():
        content = p.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(p))
        doc_nodes = get_docstring_nodes(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node in doc_nodes:
                    continue
                str_val = node.value
                matches = num_re.findall(str_val)
                for m in matches:
                    matched_token = m[0] if isinstance(m, tuple) else m
                    if not matched_token:
                        continue
                    assert (
                        matched_token in NUMERIC_LITERAL_ALLOWLIST
                        or str_val in NUMERIC_LITERAL_ALLOWLIST
                    ), (
                        f"Non-allowlisted numeric literal '{matched_token}' found in "
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
