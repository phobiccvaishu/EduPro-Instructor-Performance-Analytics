"""
edupro.analytics.aggregation
============================
Statistically defensible entity-level aggregations for EduPro Analytics.

Because the underlying data model has NO direct TeacherID in Courses,
all instructor-course pairs are joined through Transactions (yielding 887 pairs
from only 60 teachers and 60 courses).

To prevent pseudo-replication in statistical testing, analysis must be conducted
at the genuine observational unit:
- Teacher-level table (n=60)
- Course-level table (n=60)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edupro.config import settings


def teacher_table(
    teachers: pd.DataFrame,
    master: pd.DataFrame,
    attribution: str = "enrollment_weighted",
    min_pair_enrollment: int | None = None,
) -> pd.DataFrame:
    """
    Aggregate master pairs to the teacher level (n=60).

    Columns produced:
    - Teacher profile: TeacherID, TeacherName, Age, Gender, Expertise, YearsOfExperience, TeacherRating
    - total_enrollments: total students enrolled across all courses taught by instructor
    - enrollment_share: total_enrollments / total_platform_enrollments
    - n_courses: number of distinct courses taught
    - weighted_course_rating: enrollment-weighted (or unweighted) mean of course ratings
    - own_domain_share: proportion of teacher's enrollments in courses matching their Expertise
    - InstructorTier: tier category ('High (>=4.0)', 'Mid (3.0-3.9)', 'Low (<3.0)')
    """
    total_platform_enrollments = float(master["EnrollmentCount"].sum()) if not master.empty else 1.0

    m = master.copy()
    if min_pair_enrollment is not None and min_pair_enrollment > 0:
        m = m[m["EnrollmentCount"] >= min_pair_enrollment]

    if m.empty:
        df = teachers.copy()
        df["total_enrollments"] = 0
        df["enrollment_share"] = 0.0
        df["n_courses"] = 0
        df["weighted_course_rating"] = np.nan
        df["own_domain_share"] = 0.0
        df["own_domain_course_share"] = 0.0
        df["InstructorTier"] = df["TeacherRating"].apply(settings.get_tier)
        return df

    # Aggregations per TeacherID
    records = []
    for tid, grp in m.groupby("TeacherID"):
        tot_enr = int(grp["EnrollmentCount"].sum())
        n_c = int(grp["CourseID"].nunique())

        if attribution == "enrollment_weighted":
            if tot_enr > 0:
                wt_course_rating = float(
                    (grp["EnrollmentCount"] * grp["CourseRating"]).sum() / tot_enr
                )
            else:
                wt_course_rating = np.nan
        else:  # unweighted arithmetic mean across courses
            wt_course_rating = float(grp["CourseRating"].mean())

        # Own domain share: CourseCategory == Expertise
        own_domain_mask = grp["CourseCategory"] == grp["Expertise"]
        own_enr = float(grp.loc[own_domain_mask, "EnrollmentCount"].sum())
        own_domain_sh = float(own_enr / tot_enr) if tot_enr > 0 else 0.0
        own_c = int(grp.loc[own_domain_mask, "CourseID"].nunique())
        own_domain_course_sh = float(own_c / n_c) if n_c > 0 else 0.0

        records.append(
            {
                "TeacherID": tid,
                "total_enrollments": tot_enr,
                "enrollment_share": float(tot_enr / total_platform_enrollments),
                "n_courses": n_c,
                "weighted_course_rating": wt_course_rating,
                "own_domain_share": own_domain_sh,
                "own_domain_course_share": own_domain_course_sh,
            }
        )

    agg_df = pd.DataFrame(records)
    merged = teachers.merge(agg_df, on="TeacherID", how="left")

    # Fill defaults for teachers with zero remaining pairs
    merged["total_enrollments"] = merged["total_enrollments"].fillna(0).astype(int)
    merged["enrollment_share"] = merged["enrollment_share"].fillna(0.0).astype(float)
    merged["n_courses"] = merged["n_courses"].fillna(0).astype(int)
    merged["own_domain_share"] = merged["own_domain_share"].fillna(0.0).astype(float)
    merged["own_domain_course_share"] = merged["own_domain_course_share"].fillna(0.0).astype(float)
    # weighted_course_rating stays np.nan if no courses

    # InstructorTier
    merged["InstructorTier"] = merged["TeacherRating"].apply(settings.get_tier)

    return merged


def course_table(
    courses: pd.DataFrame,
    master: pd.DataFrame,
    attribution: str = "enrollment_weighted",
    min_pair_enrollment: int | None = None,
) -> pd.DataFrame:
    """
    Aggregate master pairs to the course level (n=60).

    Columns produced:
    - Course catalog: CourseID, CourseName, CourseCategory, CourseType, CourseLevel,
                      CoursePrice, CourseDuration, CourseRating
    - total_enrollments: total students enrolled across all instructors of this course
    - n_teachers: number of distinct instructors assigned to this course
    - weighted_teacher_rating: enrollment-weighted (or unweighted) mean of instructor ratings
    """
    m = master.copy()
    if min_pair_enrollment is not None and min_pair_enrollment > 0:
        m = m[m["EnrollmentCount"] >= min_pair_enrollment]

    if m.empty:
        df = courses.copy()
        df["total_enrollments"] = 0
        df["n_teachers"] = 0
        df["weighted_teacher_rating"] = np.nan
        return df

    records = []
    for cid, grp in m.groupby("CourseID"):
        tot_enr = int(grp["EnrollmentCount"].sum())
        n_t = int(grp["TeacherID"].nunique())

        if attribution == "enrollment_weighted":
            if tot_enr > 0:
                wt_teacher_rating = float(
                    (grp["EnrollmentCount"] * grp["TeacherRating"]).sum() / tot_enr
                )
            else:
                wt_teacher_rating = np.nan
        else:
            wt_teacher_rating = float(grp["TeacherRating"].mean())

        records.append(
            {
                "CourseID": cid,
                "total_enrollments": tot_enr,
                "n_teachers": n_t,
                "weighted_teacher_rating": wt_teacher_rating,
            }
        )

    agg_df = pd.DataFrame(records)
    merged = courses.merge(agg_df, on="CourseID", how="left")

    merged["total_enrollments"] = merged["total_enrollments"].fillna(0).astype(int)
    merged["n_teachers"] = merged["n_teachers"].fillna(0).astype(int)

    return merged
