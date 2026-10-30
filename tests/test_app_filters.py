"""
tests.test_app_filters
======================
Integration and regression tests for Streamlit UI filtering, state persistence,
reset workflows, privacy rule enforcement, and randomized stress testing.
"""

from __future__ import annotations

import random
from pathlib import Path

from streamlit.testing.v1 import AppTest

from edupro.analytics.engine import AnalyticsEngine
from edupro.config import AnalysisConfig
from edupro.filters import FilterBounds, FilterState, apply_filters

APP_PATH = str((Path(__file__).resolve().parent.parent / "app.py").resolve())


def test_default_view_renders_cleanly_and_respects_privacy():
    """Verify that the dashboard initializes without errors and never exposes private user data."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert not at.exception, f"App raised unexpected exception: {at.exception}"

    # Verify KPI metrics exist
    metrics = at.metric
    assert len(metrics) >= 5, f"Expected at least 5 KPI metrics, got {len(metrics)}"

    # Check avg teacher rating metric is rendered
    labels = [m.label for m in metrics]
    assert "Avg Teacher Rating" in labels
    assert "Avg Course Rating" in labels
    assert "Quality Consistency" in labels

    # Privacy Rule Check:
    # Ensure no User sheet attributes (UserName, Email, or raw user demographic info) appear
    # in any dataframe, caption, markdown, or text element.
    private_tokens = ["username", "user_name", "email", "@example.com", "user age", "learner age"]

    # Check all dataframes rendered
    for df_elem in at.dataframe:
        df = df_elem.value
        cols_lower = [str(c).lower() for c in df.columns]
        for token in private_tokens:
            assert token not in cols_lower, (
                f"Privacy violation: Column '{token}' found in dataframe"
            )

    # Check all text / markdown / caption elements
    rendered_text = " ".join(
        [str(m.value) for m in at.markdown]
        + [str(c.value) for c in at.caption]
        + [str(t.value) for t in at.text]
    ).lower()

    for token in ["username", "user_name", "@example.com"]:
        assert token not in rendered_text, (
            f"Privacy violation: Token '{token}' found in rendered UI text"
        )


def test_multiselect_filter_and_reset_interaction():
    """Verify changing a multiselect filter updates the app and reset restores default state."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Select single expertise: Machine Learning
    exp_ms = at.sidebar.multiselect(key="ui_sel_expertise")
    assert exp_ms is not None
    exp_ms.set_value(["Machine Learning"])
    at.run()
    assert not at.exception

    # Verify caption reflects narrowed scope (Machine Learning has 5 instructors)
    captions = [c.value for c in at.caption]
    tab1_captions = [c for c in captions if "Unit of analysis: instructor" in c]
    assert len(tab1_captions) > 0
    assert "5 instructors" in tab1_captions[0]

    # Click Reset all filters
    reset_btn = at.sidebar.button[0]
    assert "Reset all filters" in reset_btn.label
    reset_btn.click()
    at.run()
    assert not at.exception

    # Verify reset restored 60 instructors
    captions_after = [c.value for c in at.caption]
    tab1_captions_after = [c for c in captions_after if "Unit of analysis: instructor" in c]
    assert len(tab1_captions_after) > 0
    assert "60 instructors" in tab1_captions_after[0]


def test_narrow_slider_filter():
    """Verify applying tight rating sliders updates without crashing."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Set teacher rating slider to high tier only (4.0 to 4.97)
    tr_slider = at.sidebar.slider(key="ui_sel_tr")
    tr_slider.set_value((4.0, 4.97))
    at.run()
    assert not at.exception

    captions = [c.value for c in at.caption]
    tab1_captions = [c for c in captions if "Unit of analysis: instructor" in c]
    assert len(tab1_captions) > 0
    # There are 11 instructors with rating >= 4.0
    assert "11 instructors" in tab1_captions[0]


def test_empty_filter_guard():
    """Verify that an empty filter combination renders the empty-view info guard safely."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Set teacher rating slider to range where 0 teachers exist (1.10 to 1.20)
    tr_slider = at.sidebar.slider(key="ui_sel_tr")
    tr_slider.set_value((1.10, 1.20))
    at.run()
    assert not at.exception

    # Verify st.info message was displayed
    infos = [i.value for i in at.info]
    assert any("No instructors, courses, or enrollments match" in i for i in infos)


def test_50_combination_randomized_stress_test(dataset):
    """
    Stress-test the central filtering engine and analytics pipeline with 50 randomized
    filter combinations to ensure complete mathematical stability and no uncaught exceptions.
    """
    b = FilterBounds.from_dataset(dataset)
    rng = random.Random(42)

    all_exp = list(b.all_expertises)
    all_cat = list(b.all_categories)
    all_lvl = list(b.all_levels)
    all_gen = list(b.all_genders)

    cfg = AnalysisConfig(fast_mode=True)

    for _ in range(50):
        # Sample subsets of varying sizes (including empty or full)
        n_exp = rng.randint(0, len(all_exp))
        exp_sample = tuple(sorted(rng.sample(all_exp, n_exp)))

        n_cat = rng.randint(0, len(all_cat))
        cat_sample = tuple(sorted(rng.sample(all_cat, n_cat)))

        n_lvl = rng.randint(0, len(all_lvl))
        lvl_sample = tuple(sorted(rng.sample(all_lvl, n_lvl)))

        n_gen = rng.randint(0, len(all_gen))
        gen_sample = tuple(sorted(rng.sample(all_gen, n_gen)))

        # Sample numeric ranges
        tr_low = round(rng.uniform(b.teacher_rating_min, b.teacher_rating_max - 0.5), 2)
        tr_high = round(rng.uniform(tr_low, b.teacher_rating_max), 2)

        cr_low = round(rng.uniform(b.course_rating_min, b.course_rating_max - 0.5), 2)
        cr_high = round(rng.uniform(cr_low, b.course_rating_max), 2)

        exp_low = round(rng.uniform(b.experience_min, b.experience_max - 2), 1)
        exp_high = round(rng.uniform(exp_low, b.experience_max), 1)

        state = FilterState(
            expertise=exp_sample,
            course_category=cat_sample,
            course_level=lvl_sample,
            instructor_gender=gen_sample,
            teacher_rating_range=(tr_low, tr_high),
            course_rating_range=(cr_low, cr_high),
            experience_range=(exp_low, exp_high),
        )

        view = apply_filters(dataset, state)
        assert isinstance(view.is_empty, bool)

        if not view.is_empty:
            assert view.counts["n_teachers_active"] > 0
            assert view.counts["n_courses_active"] > 0
            assert view.counts["n_pairs"] > 0
            assert view.counts["n_enrollments"] > 0

            # Total enrollments in re-aggregated teacher table must equal view total enrollments
            assert view.teacher_table["total_enrollments"].sum() == view.counts["n_enrollments"]
            assert view.course_table["total_enrollments"].sum() == view.counts["n_enrollments"]

            # Run analytics engine in fast mode
            engine = AnalyticsEngine(
                view.master,
                view.teachers,
                view.courses,
                view.transactions,
            )
            res = engine.run_full_analysis(cfg=cfg)

            assert "kpis" in res
            assert "findings" in res
            assert "expertise_performance" in res
            assert "instructor_impact" in res
            assert "course_quality" in res
            assert "experience_vs_performance" in res
            assert "teacher_course_corr" in res
