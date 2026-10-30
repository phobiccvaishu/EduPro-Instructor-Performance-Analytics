"""
edupro.ui.theme
===============
Design system tokens, typography scales, neutral colour palette,
accessible Plotly template, and documented CSS rules for executive reporting.
"""

from __future__ import annotations

import streamlit as st

# ── 1. Neutral Color Palette ──────────────────────────────────────────────────
# Primary and text hierarchy (contrast ratio > 4.5:1 against light surfaces)
COLOR_TEXT_PRIMARY = "#0F172A"  # Slate 900 (Dark Navy)
COLOR_TEXT_SECONDARY = "#475569"  # Slate 600 (Muted Body)
COLOR_TEXT_MUTED = "#64748B"  # Slate 500 (Captions)

# Surfaces and dividers
COLOR_BG_WHITE = "#FFFFFF"  # Base canvas
COLOR_BG_PANEL = "#F8FAFC"  # Slate 50 (Card backgrounds)
COLOR_BORDER = "#E2E8F0"  # Slate 200 (Dividers, borders)

# Data series colors (restrained, accessible)
COLOR_SERIES_PRIMARY = "#1E40AF"  # Slate/Cobalt Blue 800 (Primary data)
COLOR_SERIES_SECONDARY = "#94A3B8"  # Slate 400 (Reference/secondary data)
COLOR_SERIES_ACCENT = "#2563EB"  # Blue 600

# Color-blind-safe sequential scale for heatmaps and density plots
BLUES_SEQUENTIAL = [
    "#F0F7FF",  # 0%
    "#C7E2FE",  # 25%
    "#73B4FD",  # 50%
    "#2B7FFF",  # 75%
    "#1E40AF",  # 100%
]

# Semantic status colors (used strictly for evidence/verdict status badges)
STATUS_COLORS = {
    "Supported": {"text": "#166534", "bg": "#F0FDF4", "border": "#BBF7D0"},  # Green
    "Inconclusive": {"text": "#92400E", "bg": "#FFFBEB", "border": "#FDE68A"},  # Amber
    "Not supported": {"text": "#334155", "bg": "#F1F5F9", "border": "#CBD5E1"},  # Slate
    "Descriptive": {"text": "#1E3A8A", "bg": "#EFF6FF", "border": "#BFDBFE"},  # Blue
    "Insufficient data": {"text": "#B45309", "bg": "#FEF3C7", "border": "#FCD34D"},  # Amber
}

TIER_COLORS = {
    "High": "#166534",
    "Mid": "#475569",
    "Low": "#991B1B",
}

FONT_FAMILY = "Segoe UI, -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif"

# ── 2. Unified Plotly Base Template ───────────────────────────────────────────
CHART_FONT = dict(family=FONT_FAMILY, size=12, color=COLOR_TEXT_PRIMARY)

PLOTLY_LAYOUT = dict(
    plot_bgcolor=COLOR_BG_WHITE,
    paper_bgcolor=COLOR_BG_WHITE,
    font=CHART_FONT,
    margin=dict(t=40, b=30, l=40, r=20),
    title=dict(
        font=dict(family=FONT_FAMILY, size=14, color=COLOR_TEXT_PRIMARY),
        x=0.0,
        xanchor="left",
    ),
    xaxis=dict(
        gridcolor="#F1F5F9",
        linecolor=COLOR_BORDER,
        tickfont=dict(size=11, color=COLOR_TEXT_SECONDARY),
        title_font=dict(size=12, color=COLOR_TEXT_PRIMARY),
        zerolinecolor=COLOR_BORDER,
    ),
    yaxis=dict(
        gridcolor="#F1F5F9",
        linecolor=COLOR_BORDER,
        tickfont=dict(size=11, color=COLOR_TEXT_SECONDARY),
        title_font=dict(size=12, color=COLOR_TEXT_PRIMARY),
        zerolinecolor=COLOR_BORDER,
    ),
    legend=dict(
        font=dict(size=11, color=COLOR_TEXT_SECONDARY),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor=COLOR_BORDER,
        borderwidth=1,
    ),
    hoverlabel=dict(
        bgcolor=COLOR_BG_WHITE,
        font_size=12,
        font_family=FONT_FAMILY,
        font_color=COLOR_TEXT_PRIMARY,
        bordercolor=COLOR_BORDER,
    ),
)


# ── 3. Minimal Executive CSS Block ────────────────────────────────────────────
def inject_theme_css() -> None:
    """
    Inject clean, neutral styling into the Streamlit app.
    Enforces strict executive typography, compact spacing, readable metrics,
    and removes all CSS animations, transitions, box shadows, and hover effects.
    """
    st.markdown(
        f"""
<style>
/* Main content container clearance */
.block-container {{
    padding-top: 3.2rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 98% !important;
}}

/* Typography scale */
body, [data-testid="stAppViewContainer"] {{
    font-family: {FONT_FAMILY} !important;
    color: {COLOR_TEXT_PRIMARY} !important;
}}

/* Metric cards: dense, accessible, non-decorative */
[data-testid="stMetric"] {{
    background-color: {COLOR_BG_PANEL} !important;
    border: 1px solid {COLOR_BORDER} !important;
    border-radius: 4px !important;
    padding: 0.75rem 1rem !important;
    box-shadow: none !important;
}}

[data-testid="stMetricLabel"] {{
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    color: {COLOR_TEXT_SECONDARY} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
}}

[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {{
    font-size: clamp(0.95rem, 1.25vw, 1.45rem) !important;
    font-weight: 700 !important;
    color: {COLOR_TEXT_PRIMARY} !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
}}

/* Disables all transitions and animations */
*, *::before, *::after {{
    transition: none !important;
    animation: none !important;
}}

/* Expander headers */
.streamlit-expanderHeader {{
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: {COLOR_TEXT_PRIMARY} !important;
    border-radius: 4px !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )
