"""
edupro.data.schema
==================
Schema definitions, required columns, and dtypes for EduPro datasets.
"""

from __future__ import annotations

from typing import Final

REQUIRED_SHEETS: Final[tuple[str, ...]] = (
    "Users",
    "Teachers",
    "Courses",
    "Transactions",
)

USERS_COLS: Final[list[str]] = [
    "UserID",
    "UserName",
    "Age",
    "Gender",
    "Email",
]

TEACHERS_COLS: Final[list[str]] = [
    "TeacherID",
    "TeacherName",
    "Age",
    "Gender",
    "Expertise",
    "YearsOfExperience",
    "TeacherRating",
]

COURSES_COLS: Final[list[str]] = [
    "CourseID",
    "CourseName",
    "CourseCategory",
    "CourseType",
    "CourseLevel",
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
]

TRANSACTIONS_COLS: Final[list[str]] = [
    "TransactionID",
    "UserID",
    "CourseID",
    "TransactionDate",
    "Amount",
    "PaymentMethod",
    "TeacherID",
]

MASTER_COLS: Final[list[str]] = [
    "TeacherID",
    "CourseID",
    "EnrollmentCount",
    "CourseName",
    "CourseCategory",
    "CourseType",
    "CourseLevel",
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "TeacherName",
    "Age",
    "Gender",
    "Expertise",
    "YearsOfExperience",
    "TeacherRating",
    "_has_teacher_rating",
    "_has_course_rating",
]

# Primary keys per sheet
PRIMARY_KEYS: Final[dict[str, str]] = {
    "Users": "UserID",
    "Teachers": "TeacherID",
    "Courses": "CourseID",
    "Transactions": "TransactionID",
}

# Required non-null columns per sheet for cleaning
REQUIRED_KEYS: Final[dict[str, list[str]]] = {
    "Users": ["UserID"],
    "Teachers": ["TeacherID", "TeacherRating"],
    "Courses": ["CourseID", "CourseRating"],
    "Transactions": ["TransactionID", "CourseID", "TeacherID"],
}

SHEET_COLUMNS: Final[dict[str, list[str]]] = {
    "Users": USERS_COLS,
    "Teachers": TEACHERS_COLS,
    "Courses": COURSES_COLS,
    "Transactions": TRANSACTIONS_COLS,
}
