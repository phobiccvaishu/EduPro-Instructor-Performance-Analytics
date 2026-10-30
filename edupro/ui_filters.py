"""
edupro.ui_filters
=================
Streamlit UI sidebar filter component for EduPro Analytics Platform.
Renders unified widgets, handles reset state, and reports active scope counts.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import streamlit as st

from edupro.filters import FilterBounds, FilterState, apply_filters


def render_sidebar_filters(
    dataset: Any,
    session_key: str = "filter_state",
) -> FilterState:
    """
    Render central sidebar filter widgets, maintain state in st.session_state,
    support one-click reset, and display active scope metrics.
    """
    b = FilterBounds.from_dataset(dataset)

    if session_key not in st.session_state:
        init_state = FilterState.default(dataset)
        preset = st.query_params.get("preset")
        if preset == "narrow":
            init_state = dataclasses.replace(
                init_state,
                course_category=("Data Science",),
                experience_range=(5.0, 15.0),
            )
        st.session_state[session_key] = init_state

    curr: FilterState = st.session_state[session_key]

    with st.sidebar:
        st.title("EduPro Analytics")
        st.caption("Instructor & Course Quality Dashboard")
        st.divider()

        # Reset all filters button
        if st.button("Reset all filters", width="stretch"):
            st.session_state[session_key] = FilterState.default(dataset)
            for k in [
                "ui_sel_expertise",
                "ui_sel_category",
                "ui_sel_level",
                "ui_sel_gender",
                "ui_sel_tr",
                "ui_sel_cr",
                "ui_sel_exp",
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

        st.subheader("Filters")

        # Multiselects
        def_exp = list(curr.expertise) if curr.expertise else list(b.all_expertises)
        sel_exp = st.multiselect(
            "Instructor Expertise",
            options=list(b.all_expertises),
            default=def_exp,
            key="ui_sel_expertise",
        )

        def_cat = list(curr.course_category) if curr.course_category else list(b.all_categories)
        sel_cat = st.multiselect(
            "Course Category",
            options=list(b.all_categories),
            default=def_cat,
            key="ui_sel_category",
        )

        def_lvl = list(curr.course_level) if curr.course_level else list(b.all_levels)
        sel_lvl = st.multiselect(
            "Course Level",
            options=list(b.all_levels),
            default=def_lvl,
            key="ui_sel_level",
        )

        def_gen = list(curr.instructor_gender) if curr.instructor_gender else list(b.all_genders)
        sel_gen = st.multiselect(
            "Instructor Gender",
            options=list(b.all_genders),
            default=def_gen,
            key="ui_sel_gender",
        )

        st.divider()
        st.subheader("Rating Range")

        sel_tr = st.slider(
            "Teacher Rating",
            min_value=b.teacher_rating_min,
            max_value=b.teacher_rating_max,
            value=(
                max(b.teacher_rating_min, curr.teacher_rating_range[0]),
                min(b.teacher_rating_max, curr.teacher_rating_range[1]),
            ),
            step=0.01,
            key="ui_sel_tr",
        )

        sel_cr = st.slider(
            "Course Rating",
            min_value=b.course_rating_min,
            max_value=b.course_rating_max,
            value=(
                max(b.course_rating_min, curr.course_rating_range[0]),
                min(b.course_rating_max, curr.course_rating_range[1]),
            ),
            step=0.01,
            key="ui_sel_cr",
        )

        st.divider()
        st.subheader("Experience")
        sel_exp_rng = st.slider(
            "Years of Experience",
            min_value=float(b.experience_min),
            max_value=float(b.experience_max),
            value=(
                max(float(b.experience_min), curr.experience_range[0]),
                min(float(b.experience_max), curr.experience_range[1]),
            ),
            step=1.0,
            key="ui_sel_exp",
        )

        # Normalize selections: all selected or none selected = empty tuple for unconstrained
        tup_exp = (
            () if (len(sel_exp) == len(b.all_expertises) or len(sel_exp) == 0) else tuple(sel_exp)
        )
        tup_cat = (
            () if (len(sel_cat) == len(b.all_categories) or len(sel_cat) == 0) else tuple(sel_cat)
        )
        tup_lvl = () if (len(sel_lvl) == len(b.all_levels) or len(sel_lvl) == 0) else tuple(sel_lvl)
        tup_gen = (
            () if (len(sel_gen) == len(b.all_genders) or len(sel_gen) == 0) else tuple(sel_gen)
        )

        new_state = FilterState(
            expertise=tup_exp,
            course_category=tup_cat,
            course_level=tup_lvl,
            instructor_gender=tup_gen,
            teacher_rating_range=(float(sel_tr[0]), float(sel_tr[1])),
            course_rating_range=(float(sel_cr[0]), float(sel_cr[1])),
            experience_range=(float(sel_exp_rng[0]), float(sel_exp_rng[1])),
        )
        st.session_state[session_key] = new_state

        st.divider()
        view = apply_filters(dataset, new_state)
        n_teachers_total = len(
            dataset.teachers if hasattr(dataset, "teachers") else dataset["teachers"]
        )
        n_courses_total = len(
            dataset.courses if hasattr(dataset, "courses") else dataset["courses"]
        )
        st.caption(
            f"{view.counts['n_teachers']} of {n_teachers_total} instructors, "
            f"{view.counts['n_courses']} of {n_courses_total} courses in view"
        )
        st.caption("EduPro Analytics Suite v2.0")

    return new_state
