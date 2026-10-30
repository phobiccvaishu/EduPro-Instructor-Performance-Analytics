"""
edupro.ui.tabs.tab3_course_quality
==================================
Tab 3 — Course Content Quality, Category Variance, and Difficulty Distribution.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from edupro.insights import generate_f6_insight, generate_f7_insight
from edupro.ui.charts import build_heatmap_matrix, build_horizontal_bar
from edupro.ui.components import (
    empty_state,
    insight_box,
    method_note,
    scope_caption,
    section_header,
)


def render(
    view: Any, findings: dict[str, Any], kpis: dict[str, Any], is_fast_mode: bool = False
) -> None:
    """Render Tab 3 view."""
    section_header(
        "Course Content Quality & Distribution Maps",
        "Assessing course quality ratings across subject categories and curriculum difficulty tiers.",
    )
    scope_caption(view, unit="catalog course & enrollment transaction", is_fast_mode=is_fast_mode)

    if view.is_empty or len(view.course_table) == 0:
        empty_state("No course records match the selected filter combination.")
        return

    courses_df = view.course_table.copy()
    master_df = view.master.copy()

    # 1. Generated Dynamic Insights
    if "F6" in findings:
        f6_ins = generate_f6_insight(findings["F6"], is_fast_mode=is_fast_mode)
        insight_box(
            headline=f6_ins.headline,
            evidence_text=f6_ins.evidence_sentence,
            caveat_text=f6_ins.caveat,
            verdict=f6_ins.verdict,
            evidence_level=f6_ins.evidence_level,
            stat_meta=f6_ins.stat_summary,
        )

    if "F7" in findings:
        f7_ins = generate_f7_insight(findings["F7"], is_fast_mode=is_fast_mode)
        insight_box(
            headline=f7_ins.headline,
            evidence_text=f7_ins.evidence_sentence,
            caveat_text=f7_ins.caveat,
            verdict=f7_ins.verdict,
            evidence_level=f7_ins.evidence_level,
            stat_meta=f7_ins.stat_summary,
        )

    # 2. Methodology & Governance Note
    method_note(
        "Course Quality Assessment & Matrix Construction",
        [
            ("Unit of Analysis", "Catalog course level (n = 60 baseline)."),
            (
                "Statistical Tests",
                "Welch one-way ANOVA for category variance; Kruskal-Wallis test for difficulty levels.",
            ),
            (
                "Cell Suppression",
                "Heatmap cells with sample count n < 3 are annotated with an asterisk to denote small sample fragility.",
            ),
            (
                "Catalog Independence",
                "Course ratings evaluate curriculum content independently of instructor reputation.",
            ),
        ],
    )

    # 3. Heatmap Matrices: Category x Level Rating and Category x Level Enrollments
    st.markdown("##### Category by Difficulty Level Matrix")
    h1, h2 = st.columns(2, gap="large")

    with h1:
        # Matrix 1: Average Course Rating (Category x Level)
        cat_level_rating = master_df.pivot_table(
            index="CourseCategory",
            columns="CourseLevel",
            values="CourseRating",
            aggfunc="mean",
            observed=False,
        ).round(2)

        cat_level_counts = master_df.pivot_table(
            index="CourseCategory",
            columns="CourseLevel",
            values="CourseRating",
            aggfunc="count",
            observed=False,
        ).fillna(0)

        fig_hm1 = build_heatmap_matrix(
            cat_level_rating,
            title="Course Rating — Category by Difficulty Level",
            x_title="Difficulty Level",
            y_title="Course Category",
            count_df=cat_level_counts,
            min_group_size=3,
            text_format=".2f",
            z_min=1.0,
            z_max=5.0,
            height=430,
        )
        st.plotly_chart(fig_hm1, width="stretch")
        st.caption(
            "Note: Red asterisk (n=*) indicates fewer than 3 observations in that combination."
        )

    with h2:
        # Matrix 2: Total Enrollments (Category x Level)
        cat_level_enr = master_df.pivot_table(
            index="CourseCategory",
            columns="CourseLevel",
            values="EnrollmentCount",
            aggfunc="sum",
            observed=False,
        ).fillna(0)

        fig_hm2 = build_heatmap_matrix(
            cat_level_enr,
            title="Total Enrollments — Category by Difficulty Level",
            x_title="Difficulty Level",
            y_title="Course Category",
            count_df=cat_level_counts,
            text_format=".0f",
            height=430,
        )
        st.plotly_chart(fig_hm2, width="stretch")

    st.markdown("---")

    # 4. Bar Charts: Category and Difficulty Level Rankings
    b1, b2 = st.columns(2, gap="large")

    with b1:
        cat_summary = (
            courses_df.groupby("CourseCategory", observed=False)
            .agg(
                avg_rating=("CourseRating", "mean"),
                course_count=("CourseID", "count"),
            )
            .reset_index()
            .sort_values("avg_rating", ascending=True)
        )

        fig_cat = build_horizontal_bar(
            cat_summary,
            x_col="avg_rating",
            y_col="CourseCategory",
            title="Average Course Rating by Category",
            x_title="Average Course Rating (1–5 scale)",
            y_title="Category",
            height=380,
            x_range=[0.0, 5.0],
        )
        st.plotly_chart(fig_cat, width="stretch")

    with b2:
        lvl_summary = (
            courses_df.groupby("CourseLevel", observed=False)
            .agg(
                avg_rating=("CourseRating", "mean"),
                course_count=("CourseID", "count"),
            )
            .reset_index()
            .sort_values("avg_rating", ascending=True)
        )

        fig_lvl = build_horizontal_bar(
            lvl_summary,
            x_col="avg_rating",
            y_col="CourseLevel",
            title="Average Course Rating by Difficulty Level",
            x_title="Average Course Rating (1–5 scale)",
            y_title="Difficulty Level",
            height=380,
            x_range=[0.0, 5.0],
        )
        st.plotly_chart(fig_lvl, width="stretch")

    st.markdown("---")

    # 5. Tabular Category Quality Summary
    st.markdown("##### Subject Category Quality Breakdown")
    cat_table = (
        courses_df.groupby("CourseCategory", observed=False)
        .agg(
            Courses=("CourseID", "count"),
            Avg_Rating=("CourseRating", "mean"),
            Min_Rating=("CourseRating", "min"),
            Max_Rating=("CourseRating", "max"),
            Total_Enrollments=("total_enrollments", "sum"),
        )
        .reset_index()
        .sort_values("Avg_Rating", ascending=False)
    )

    st.dataframe(
        cat_table,
        width="stretch",
        hide_index=True,
        column_config={
            "CourseCategory": st.column_config.TextColumn("Course Category"),
            "Courses": st.column_config.NumberColumn("Courses", format="%d"),
            "Avg_Rating": st.column_config.NumberColumn("Average Quality Score", format="%.2f"),
            "Min_Rating": st.column_config.NumberColumn("Minimum Score", format="%.2f"),
            "Max_Rating": st.column_config.NumberColumn("Maximum Score", format="%.2f"),
            "Total_Enrollments": st.column_config.NumberColumn(
                "Total Student Enrollments", format="%d"
            ),
        },
    )
