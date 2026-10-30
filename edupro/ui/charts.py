"""
edupro.ui.charts
================
Standardized, accessible Plotly chart builders for the EduPro executive dashboard.
Applies the unified design system template, explicit axis units, color-blind safe
sequential palettes, and transparent sample size annotations.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from edupro.ui.theme import (
    BLUES_SEQUENTIAL,
    COLOR_BORDER,
    COLOR_SERIES_PRIMARY,
    COLOR_TEXT_PRIMARY,
    FONT_FAMILY,
    PLOTLY_LAYOUT,
)


def apply_base_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    """Apply standard executive styling, left-aligned title, and margins."""
    layout_copy = dict(PLOTLY_LAYOUT)
    layout_copy["title"] = dict(
        text=title,
        font=dict(family=FONT_FAMILY, size=13, color=COLOR_TEXT_PRIMARY),
        x=0.0,
        xanchor="left",
    )
    layout_copy["height"] = height
    fig.update_layout(**layout_copy)
    return fig


def build_horizontal_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_title: str,
    y_title: str = "",
    color_col: str | None = None,
    color_map: dict[str, str] | None = None,
    height: int = 380,
    text_format: str = ".2f",
    x_range: list[float] | None = None,
) -> go.Figure:
    """Build a clean horizontal bar chart with outside text formatting and explicit units."""
    if df.empty:
        fig = go.Figure()
        return apply_base_layout(fig, title, height)

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        orientation="h",
        color=color_col,
        color_discrete_map=color_map,
        text=x_col,
    )

    fig.update_traces(
        texttemplate=f"<b>%{{text:{text_format}}}</b>",
        textposition="outside",
        textfont=dict(size=11, color=COLOR_TEXT_PRIMARY),
        cliponaxis=False,
    )

    if color_col and color_map:
        fig.update_layout(legend_title_text=color_col)
    else:
        fig.update_traces(marker_color=COLOR_SERIES_PRIMARY)

    fig.update_xaxes(title_text=x_title, range=x_range)
    fig.update_yaxes(title_text=y_title, categoryorder="total ascending")
    return apply_base_layout(fig, title, height)


def build_scatter_with_trendline(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_title: str,
    y_title: str,
    slope: float | None = None,
    slope_ci: list[float] | tuple[float, float] | None = None,
    hover_name: str | None = None,
    color_col: str | None = None,
    height: int = 400,
    y_range: list[float] | None = None,
) -> go.Figure:
    """
    Build a scatter plot annotated with sample size and an overlay regression line
    whose slope is supplied by statistical findings.
    """
    fig = go.Figure()
    n_pts = len(df)

    if df.empty or n_pts == 0:
        return apply_base_layout(fig, f"{title} (n = 0)", height)

    # Primary scatter points
    hover_text = []
    for _, row in df.iterrows():
        name_str = f"<b>{row[hover_name]}</b><br>" if hover_name and hover_name in row else ""
        hover_text.append(f"{name_str}{x_title}: {row[x_col]}<br>{y_title}: {row[y_col]:.2f}")

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            marker=dict(
                size=8,
                color=COLOR_SERIES_PRIMARY,
                opacity=0.75,
                line=dict(width=1, color=COLOR_BORDER),
            ),
            text=hover_text,
            hoverinfo="text",
            name="Records",
            showlegend=False,
        )
    )

    # Add regression trendline if slope is valid and n >= 3
    if slope is not None and not math.isnan(slope) and n_pts >= 3 and df[x_col].nunique() > 1:
        x_min, x_max = float(df[x_col].min()), float(df[x_col].max())
        y_mean = float(df[y_col].mean())
        x_mean = float(df[x_col].mean())
        intercept = y_mean - slope * x_mean

        x_line = np.linspace(x_min, x_max, 50)
        y_line = intercept + slope * x_line

        ci_label = ""
        if slope_ci and len(slope_ci) == 2 and not any(math.isnan(c) for c in slope_ci):
            ci_label = f" [95% CI: {slope_ci[0]:+.3f}, {slope_ci[1]:+.3f}]"

        line_name = f"Trend: {slope:+.3f}/yr{ci_label}"
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color=COLOR_TEXT_PRIMARY, width=2, dash="dash"),
                name=line_name,
                hoverinfo="name",
            )
        )

    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title, range=y_range)
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=-0.22,
            x=0.0,
            xanchor="left",
            font=dict(size=11),
        )
    )
    return apply_base_layout(fig, f"{title} (n = {n_pts})", height)


def build_heatmap_matrix(
    pivot_df: pd.DataFrame,
    title: str,
    x_title: str,
    y_title: str,
    count_df: pd.DataFrame | None = None,
    min_group_size: int = 3,
    color_scale: list[str] | None = None,
    text_format: str = ".2f",
    z_min: float | None = None,
    z_max: float | None = None,
    height: int = 400,
) -> go.Figure:
    """
    Build an accessible 2D matrix heatmap annotated with cell counts,
    explicitly flagging cells where n < min_group_size.
    """
    fig = go.Figure()
    if pivot_df.empty:
        return apply_base_layout(fig, title, height)

    scale = color_scale or BLUES_SEQUENTIAL

    # Construct text annotations combining mean value and sample count
    text_matrix = []
    for y_idx in pivot_df.index:
        row_text = []
        for x_col in pivot_df.columns:
            val = pivot_df.loc[y_idx, x_col]
            if pd.isna(val) or math.isnan(float(val)):
                row_text.append("No data")
                continue

            n_cell = None
            if count_df is not None and y_idx in count_df.index and x_col in count_df.columns:
                n_cell = int(count_df.loc[y_idx, x_col])

            val_str = f"{float(val):{text_format}}"
            if n_cell is not None:
                if n_cell < min_group_size:
                    row_text.append(
                        f"{val_str}<br><span style='font-size: 10px; color: #DC2626;'>n={n_cell}*</span>"
                    )
                else:
                    row_text.append(
                        f"{val_str}<br><span style='font-size: 10px; color: #475569;'>n={n_cell}</span>"
                    )
            else:
                row_text.append(val_str)
        text_matrix.append(row_text)

    fig.add_trace(
        go.Heatmap(
            z=pivot_df.values,
            x=list(pivot_df.columns),
            y=list(pivot_df.index),
            colorscale=scale,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=11, family=FONT_FAMILY),
            zmin=z_min,
            zmax=z_max,
            colorbar=dict(
                title=dict(text="", font=dict(size=11)),
                thickness=12,
                len=0.85,
                tickfont=dict(size=10),
            ),
            hoverongaps=False,
        )
    )

    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    return apply_base_layout(fig, title, height)


def build_distribution_histogram(
    df: pd.DataFrame,
    col_name: str,
    title: str,
    x_title: str,
    nbins: int = 15,
    height: int = 340,
) -> go.Figure:
    """Build a distribution histogram with density boxplot overlay."""
    fig = px.histogram(
        df,
        x=col_name,
        nbins=nbins,
        marginal="box",
        color_discrete_sequence=[COLOR_SERIES_PRIMARY],
    )
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text="Count")
    n_pts = len(df[col_name].dropna())
    return apply_base_layout(fig, f"{title} (n = {n_pts})", height)
