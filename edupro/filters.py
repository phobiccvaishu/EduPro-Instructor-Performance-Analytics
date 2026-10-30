"""
edupro.filters
==============
Centralised filtering architecture, semantic models, exact slider bounds,
and dataset slicing for the EduPro Analytics Platform.

Semantics
---------
1. Teacher-side filters:
   - expertise (multiselect, empty = all)
   - instructor_gender (multiselect, empty = all)
   - teacher_rating_range (lo, hi slider, floor/ceil bounds)
   - experience_range (lo, hi slider, floor/ceil bounds)

2. Course-side filters:
   - course_category (multiselect, empty = all)
   - course_level (multiselect, empty = all)
   - course_rating_range (lo, hi slider, floor/ceil bounds)

3. Interaction:
   The master dataset contains (TeacherID, CourseID) pairs resulting from enrollments.
   A pair is included if and only if its Teacher passes teacher-side filters AND
   its Course passes course-side filters.
   Entity tables (teacher_table and course_table) are recomputed strictly from the
   filtered master dataset. Instructors or courses with zero remaining enrollments in
   the filtered scope are excluded from entity aggregations and recorded in view notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any

import pandas as pd

from edupro.analytics.aggregation import course_table, teacher_table


def floor_2d(val: float) -> float:
    """Floor a float to 2 decimal places with decimal precision guarding."""
    d = Decimal(str(round(float(val), 6)))
    return float(d.quantize(Decimal("0.01"), rounding=ROUND_FLOOR))


def ceil_2d(val: float) -> float:
    """Ceil a float to 2 decimal places with decimal precision guarding."""
    d = Decimal(str(round(float(val), 6)))
    return float(d.quantize(Decimal("0.01"), rounding=ROUND_CEILING))


@dataclass(frozen=True)
class FilterBounds:
    """Data-derived exact slider boundary specifications."""

    teacher_rating_min: float
    teacher_rating_max: float
    course_rating_min: float
    course_rating_max: float
    experience_min: float
    experience_max: float
    all_expertises: tuple[str, ...]
    all_categories: tuple[str, ...]
    all_levels: tuple[str, ...]
    all_genders: tuple[str, ...]

    @classmethod
    def from_dataset(cls, dataset: Any) -> FilterBounds:
        """Compute exact bounds from raw or cleaned dataset."""
        teachers = dataset.teachers if hasattr(dataset, "teachers") else dataset["teachers"]
        courses = dataset.courses if hasattr(dataset, "courses") else dataset["courses"]

        tr_min = floor_2d(teachers["TeacherRating"].min())
        tr_max = ceil_2d(teachers["TeacherRating"].max())
        cr_min = floor_2d(courses["CourseRating"].min())
        cr_max = ceil_2d(courses["CourseRating"].max())
        exp_min = floor_2d(teachers["YearsOfExperience"].min())
        exp_max = ceil_2d(teachers["YearsOfExperience"].max())

        all_exp = tuple(sorted(teachers["Expertise"].dropna().unique().tolist()))
        all_cat = tuple(sorted(courses["CourseCategory"].dropna().unique().tolist()))
        all_lvl = tuple(sorted(courses["CourseLevel"].dropna().unique().tolist()))
        all_gen = tuple(sorted(teachers["Gender"].dropna().unique().tolist()))

        return cls(
            teacher_rating_min=tr_min,
            teacher_rating_max=tr_max,
            course_rating_min=cr_min,
            course_rating_max=cr_max,
            experience_min=exp_min,
            experience_max=exp_max,
            all_expertises=all_exp,
            all_categories=all_cat,
            all_levels=all_lvl,
            all_genders=all_gen,
        )


@dataclass(frozen=True)
class FilterState:
    """
    Frozen, hashable representation of user filter selections.
    Empty tuples in multiselect fields signify 'all selected'.
    """

    expertise: tuple[str, ...] = ()
    course_category: tuple[str, ...] = ()
    course_level: tuple[str, ...] = ()
    instructor_gender: tuple[str, ...] = ()
    teacher_rating_range: tuple[float, float] = (1.05, 4.97)
    course_rating_range: tuple[float, float] = (1.13, 4.94)
    experience_range: tuple[float, float] = (1.0, 24.0)

    def __post_init__(self) -> None:
        """Validate ranges are not inverted."""
        if self.teacher_rating_range[0] > self.teacher_rating_range[1]:
            raise ValueError(f"Inverted teacher_rating_range: {self.teacher_rating_range}")
        if self.course_rating_range[0] > self.course_rating_range[1]:
            raise ValueError(f"Inverted course_rating_range: {self.course_rating_range}")
        if self.experience_range[0] > self.experience_range[1]:
            raise ValueError(f"Inverted experience_range: {self.experience_range}")

    @classmethod
    def default(cls, dataset: Any) -> FilterState:
        """Construct the default filter state covering all data rows."""
        b = FilterBounds.from_dataset(dataset)
        return cls(
            expertise=(),
            course_category=(),
            course_level=(),
            instructor_gender=(),
            teacher_rating_range=(b.teacher_rating_min, b.teacher_rating_max),
            course_rating_range=(b.course_rating_min, b.course_rating_max),
            experience_range=(b.experience_min, b.experience_max),
        )

    def is_default(self, dataset: Any) -> bool:
        """Check whether the filter state matches the unconstrained default."""
        b = FilterBounds.from_dataset(dataset)
        exp_is_all = (len(self.expertise) == 0) or (set(self.expertise) == set(b.all_expertises))
        cat_is_all = (len(self.course_category) == 0) or (
            set(self.course_category) == set(b.all_categories)
        )
        lvl_is_all = (len(self.course_level) == 0) or (set(self.course_level) == set(b.all_levels))
        gen_is_all = (len(self.instructor_gender) == 0) or (
            set(self.instructor_gender) == set(b.all_genders)
        )

        tr_is_all = (self.teacher_rating_range[0] <= b.teacher_rating_min) and (
            self.teacher_rating_range[1] >= b.teacher_rating_max
        )
        cr_is_all = (self.course_rating_range[0] <= b.course_rating_min) and (
            self.course_rating_range[1] >= b.course_rating_max
        )
        exp_is_all_rng = (self.experience_range[0] <= b.experience_min) and (
            self.experience_range[1] >= b.experience_max
        )

        return bool(
            exp_is_all
            and cat_is_all
            and lvl_is_all
            and gen_is_all
            and tr_is_all
            and cr_is_all
            and exp_is_all_rng
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a serializable dictionary."""
        return {
            "expertise": list(self.expertise),
            "course_category": list(self.course_category),
            "course_level": list(self.course_level),
            "instructor_gender": list(self.instructor_gender),
            "teacher_rating_range": list(self.teacher_rating_range),
            "course_rating_range": list(self.course_rating_range),
            "experience_range": list(self.experience_range),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FilterState:
        """Reconstruct FilterState from a dictionary."""
        tr_raw = d.get("teacher_rating_range", (1.05, 4.97))
        cr_raw = d.get("course_rating_range", (1.12, 4.95))
        exp_raw = d.get("experience_range", (1.0, 24.0))

        tr_lo, tr_hi = float(tr_raw[0]), float(tr_raw[1])
        if tr_lo > tr_hi:
            tr_lo, tr_hi = tr_hi, tr_lo

        cr_lo, cr_hi = float(cr_raw[0]), float(cr_raw[1])
        if cr_lo > cr_hi:
            cr_lo, cr_hi = cr_hi, cr_lo

        exp_lo, exp_hi = float(exp_raw[0]), float(exp_raw[1])
        if exp_lo > exp_hi:
            exp_lo, exp_hi = exp_hi, exp_lo

        return cls(
            expertise=tuple(d.get("expertise", ())),
            course_category=tuple(d.get("course_category", ())),
            course_level=tuple(d.get("course_level", ())),
            instructor_gender=tuple(d.get("instructor_gender", ())),
            teacher_rating_range=(tr_lo, tr_hi),
            course_rating_range=(cr_lo, cr_hi),
            experience_range=(exp_lo, exp_hi),
        )

    def sanitize(self, dataset: Any) -> FilterState:
        """Drop nonexistent options and clamp ranges to dataset bounds."""
        b = FilterBounds.from_dataset(dataset)
        valid_exp = tuple(e for e in self.expertise if e in b.all_expertises)
        valid_cat = tuple(c for c in self.course_category if c in b.all_categories)
        valid_lvl = tuple(lvl for lvl in self.course_level if lvl in b.all_levels)
        valid_gen = tuple(g for g in self.instructor_gender if g in b.all_genders)

        tr_lo = max(b.teacher_rating_min, min(b.teacher_rating_max, self.teacher_rating_range[0]))
        tr_hi = max(tr_lo, min(b.teacher_rating_max, self.teacher_rating_range[1]))

        cr_lo = max(b.course_rating_min, min(b.course_rating_max, self.course_rating_range[0]))
        cr_hi = max(cr_lo, min(b.course_rating_max, self.course_rating_range[1]))

        exp_lo = max(b.experience_min, min(b.experience_max, self.experience_range[0]))
        exp_hi = max(exp_lo, min(b.experience_max, self.experience_range[1]))

        return FilterState(
            expertise=valid_exp,
            course_category=valid_cat,
            course_level=valid_lvl,
            instructor_gender=valid_gen,
            teacher_rating_range=(tr_lo, tr_hi),
            course_rating_range=(cr_lo, cr_hi),
            experience_range=(exp_lo, exp_hi),
        )


@dataclass(frozen=True)
class FilteredView:
    """
    Immutable representation of an analytically valid filtered dataset slice.
    """

    teachers: pd.DataFrame
    courses: pd.DataFrame
    master: pd.DataFrame
    transactions: pd.DataFrame
    teacher_table: pd.DataFrame
    course_table: pd.DataFrame
    counts: dict[str, int]
    active_filters: tuple[str, ...]
    state: FilterState
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        """Return True if no instructors, courses, or enrollments remain."""
        return self.counts.get("n_enrollments", 0) == 0 or len(self.master) == 0


def apply_filters(dataset: Any, state: FilterState | None = None) -> FilteredView:
    """
    Apply teacher-side and course-side filter criteria to dataset and recompute
    all entity tables and enrollment shares.
    """
    raw_teachers = dataset.teachers if hasattr(dataset, "teachers") else dataset["teachers"]
    raw_courses = dataset.courses if hasattr(dataset, "courses") else dataset["courses"]
    raw_master = dataset.master if hasattr(dataset, "master") else dataset["master"]
    raw_txns = (
        dataset.transactions
        if hasattr(dataset, "transactions")
        else dataset.get("transactions", pd.DataFrame())
    )

    if state is None:
        state = FilterState.default(dataset)

    b = FilterBounds.from_dataset(dataset)

    # 1. Teacher-side masking
    t_mask = pd.Series(True, index=raw_teachers.index)
    if state.expertise and len(state.expertise) < len(b.all_expertises):
        t_mask &= raw_teachers["Expertise"].isin(state.expertise)
    if state.instructor_gender and len(state.instructor_gender) < len(b.all_genders):
        t_mask &= raw_teachers["Gender"].isin(state.instructor_gender)

    t_mask &= raw_teachers["TeacherRating"].between(
        state.teacher_rating_range[0], state.teacher_rating_range[1], inclusive="both"
    )
    t_mask &= raw_teachers["YearsOfExperience"].between(
        state.experience_range[0], state.experience_range[1], inclusive="both"
    )
    filtered_teachers = raw_teachers[t_mask].copy()

    # 2. Course-side masking
    c_mask = pd.Series(True, index=raw_courses.index)
    if state.course_category and len(state.course_category) < len(b.all_categories):
        c_mask &= raw_courses["CourseCategory"].isin(state.course_category)
    if state.course_level and len(state.course_level) < len(b.all_levels):
        c_mask &= raw_courses["CourseLevel"].isin(state.course_level)

    c_mask &= raw_courses["CourseRating"].between(
        state.course_rating_range[0], state.course_rating_range[1], inclusive="both"
    )
    filtered_courses = raw_courses[c_mask].copy()

    # 3. Master pair intersection (teacher passes AND course passes)
    m_mask = raw_master["TeacherID"].isin(filtered_teachers["TeacherID"]) & raw_master[
        "CourseID"
    ].isin(filtered_courses["CourseID"])
    filtered_master = raw_master[m_mask].copy()

    # 4. Filter transactions
    if not raw_txns.empty:
        txn_mask = raw_txns["TeacherID"].isin(filtered_teachers["TeacherID"]) & raw_txns[
            "CourseID"
        ].isin(filtered_courses["CourseID"])
        filtered_txns = raw_txns[txn_mask].copy()
    else:
        filtered_txns = pd.DataFrame()

    # 5. Recompute entity tables
    if not filtered_master.empty:
        tt = teacher_table(filtered_teachers, filtered_master)
        ct = course_table(filtered_courses, filtered_master)

        # Drop instructors with 0 enrollments in scope
        no_enr_teachers = int((tt["total_enrollments"] == 0).sum())
        tt = tt[tt["total_enrollments"] > 0].copy()

        # Drop courses with 0 enrollments in scope
        no_enr_courses = int((ct["total_enrollments"] == 0).sum())
        ct = ct[ct["total_enrollments"] > 0].copy()

        # Recompute enrollment shares within the filtered view
        tot_enr = float(filtered_master["EnrollmentCount"].sum())
        if tot_enr > 0:
            if not tt.empty:
                tt["enrollment_share"] = tt["total_enrollments"] / tot_enr
            if not ct.empty:
                ct["enrollment_share"] = ct["total_enrollments"] / tot_enr
    else:
        tt = pd.DataFrame()
        ct = pd.DataFrame()
        no_enr_teachers = len(filtered_teachers)
        no_enr_courses = len(filtered_courses)
        tot_enr = 0.0

    # 6. Active filter descriptions
    active = []
    if state.expertise and len(state.expertise) < len(b.all_expertises):
        active.append(f"Expertise: {', '.join(sorted(state.expertise))}")
    if state.course_category and len(state.course_category) < len(b.all_categories):
        active.append(f"Course Category: {', '.join(sorted(state.course_category))}")
    if state.course_level and len(state.course_level) < len(b.all_levels):
        active.append(f"Course Level: {', '.join(sorted(state.course_level))}")
    if state.instructor_gender and len(state.instructor_gender) < len(b.all_genders):
        active.append(f"Instructor Gender: {', '.join(sorted(state.instructor_gender))}")

    if (
        state.teacher_rating_range[0] > b.teacher_rating_min
        or state.teacher_rating_range[1] < b.teacher_rating_max
    ):
        active.append(
            f"Teacher Rating: {state.teacher_rating_range[0]:.2f} – {state.teacher_rating_range[1]:.2f}"
        )
    if (
        state.course_rating_range[0] > b.course_rating_min
        or state.course_rating_range[1] < b.course_rating_max
    ):
        active.append(
            f"Course Rating: {state.course_rating_range[0]:.2f} – {state.course_rating_range[1]:.2f}"
        )
    if state.experience_range[0] > b.experience_min or state.experience_range[1] < b.experience_max:
        active.append(
            f"Experience: {state.experience_range[0]:.0f} – {state.experience_range[1]:.0f} yrs"
        )

    notes = []
    if no_enr_teachers > 0:
        notes.append(
            f"{no_enr_teachers} instructor(s) have no enrollments in the selected course scope and were excluded from instructor-level quality aggregations."
        )
    if no_enr_courses > 0:
        notes.append(
            f"{no_enr_courses} course(s) have no enrollments in the selected instructor scope and were excluded from course-level quality aggregations."
        )

    counts = {
        "n_teachers": len(filtered_teachers),
        "n_teachers_active": len(tt),
        "n_teachers_dropped": no_enr_teachers,
        "n_courses": len(filtered_courses),
        "n_courses_active": len(ct),
        "n_courses_dropped": no_enr_courses,
        "n_pairs": len(filtered_master),
        "n_enrollments": int(tot_enr),
    }

    return FilteredView(
        teachers=filtered_teachers,
        courses=filtered_courses,
        master=filtered_master,
        transactions=filtered_txns,
        teacher_table=tt,
        course_table=ct,
        counts=counts,
        active_filters=tuple(active),
        state=state,
        notes=tuple(notes),
    )
