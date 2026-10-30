"""
edupro.data.integration
=======================
Integration layer: combines cleaned Teachers, Courses, and Transactions
into the canonical master dataset with exact characterization preservation.
"""

from __future__ import annotations

import pandas as pd

from edupro.logging_setup import get_logger

logger = get_logger("data.integration")


def build_master(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge cleaned Transactions, Courses, and Teachers into the master dataset.

    Join Logic
    ----------
    1. Group Transactions by (TeacherID, CourseID) to compute EnrollmentCount.
    2. Left-join Courses on CourseID.
    3. Left-join Teachers on TeacherID.
    4. Compute validation flags `_has_teacher_rating` and `_has_course_rating`.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Dictionary with cleaned DataFrames keyed by 'teachers', 'courses', 'transactions'.

    Returns
    -------
    pd.DataFrame
        Master DataFrame containing 887 instructor-course pairs.
    """
    teachers = tables["teachers"]
    courses = tables["courses"]
    transactions = tables["transactions"]

    # Step 1: aggregate transactions into (TeacherID, CourseID) enrollment counts
    enrollments = (
        transactions.groupby(["TeacherID", "CourseID"], as_index=False)
        .agg(EnrollmentCount=("TransactionID", "count"))
    )

    # Step 2: merge with courses
    courses_enr = pd.merge(enrollments, courses, on="CourseID", how="left")

    # Step 3: merge with teachers
    master = pd.merge(courses_enr, teachers, on="TeacherID", how="left")

    # Add boolean validation flags matching legacy behavior
    master["_has_teacher_rating"] = master["TeacherRating"].notna()
    master["_has_course_rating"] = master["CourseRating"].notna()

    master.reset_index(drop=True, inplace=True)
    logger.info(
        "Master dataset built successfully: %d rows, %d total enrollments, %d unique teachers, %d unique courses",
        len(master),
        int(master["EnrollmentCount"].sum()),
        master["TeacherID"].nunique(),
        master["CourseID"].nunique(),
    )
    return master
