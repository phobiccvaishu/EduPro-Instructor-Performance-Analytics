"""
data_loader.py
==============
EduPro Online Platform — Data Integration Pipeline
Loads, validates, and merges the Teachers, Courses, and Transactions sheets
from 'EduPro Online Platform.xlsx'.

Author  : Senior Data Scientist
Project : Instructor Performance and Course Quality Evaluation on EduPro
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "EduPro Online Platform.xlsx",
)

TEACHERS_COLS   = ["TeacherID", "TeacherName", "Age", "Gender",
                    "Expertise", "YearsOfExperience", "TeacherRating"]
COURSES_COLS    = ["CourseID", "CourseName", "CourseCategory", "CourseType",
                    "CourseLevel", "CoursePrice", "CourseDuration", "CourseRating"]
TRANSACTIONS_COLS = ["TransactionID", "UserID", "CourseID",
                      "TransactionDate", "Amount", "PaymentMethod", "TeacherID"]


# ---------------------------------------------------------------------------
# Raw sheet loaders
# ---------------------------------------------------------------------------

def load_raw_sheets(filepath: str = DEFAULT_DATA_PATH) -> dict[str, pd.DataFrame]:
    """
    Load all sheets from the Excel workbook and return a dict keyed by
    sheet name.  Each DataFrame retains all original columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Please place 'EduPro Online Platform.xlsx' in the project root."
        )

    xl = pd.ExcelFile(filepath, engine="openpyxl")
    sheets = {}
    for name in xl.sheet_names:
        df = xl.parse(name)
        # Normalise column names: strip whitespace
        df.columns = [c.strip() for c in df.columns]
        sheets[name] = df

    return sheets


def load_teachers(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and clean the Teachers sheet."""
    sheets = load_raw_sheets(filepath)
    df = sheets["Teachers"][TEACHERS_COLS].copy()

    # Type enforcement
    df["TeacherID"]         = df["TeacherID"].astype(str).str.strip()
    df["TeacherName"]       = df["TeacherName"].astype(str).str.strip()
    df["Gender"]            = df["Gender"].astype(str).str.strip()
    df["Expertise"]         = df["Expertise"].astype(str).str.strip()
    df["Age"]               = pd.to_numeric(df["Age"], errors="coerce")
    df["YearsOfExperience"] = pd.to_numeric(df["YearsOfExperience"], errors="coerce")
    df["TeacherRating"]     = pd.to_numeric(df["TeacherRating"], errors="coerce")

    df.dropna(subset=["TeacherID", "TeacherRating"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_courses(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and clean the Courses sheet."""
    sheets = load_raw_sheets(filepath)
    df = sheets["Courses"][COURSES_COLS].copy()

    df["CourseID"]       = df["CourseID"].astype(str).str.strip()
    df["CourseName"]     = df["CourseName"].astype(str).str.strip()
    df["CourseCategory"] = df["CourseCategory"].astype(str).str.strip()
    df["CourseType"]     = df["CourseType"].astype(str).str.strip()
    df["CourseLevel"]    = df["CourseLevel"].astype(str).str.strip()
    df["CoursePrice"]    = pd.to_numeric(df["CoursePrice"], errors="coerce").fillna(0.0)
    df["CourseDuration"] = pd.to_numeric(df["CourseDuration"], errors="coerce")
    df["CourseRating"]   = pd.to_numeric(df["CourseRating"], errors="coerce")

    df.dropna(subset=["CourseID", "CourseRating"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_transactions(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and clean the Transactions sheet."""
    sheets = load_raw_sheets(filepath)
    df = sheets["Transactions"][TRANSACTIONS_COLS].copy()

    df["TransactionID"]   = df["TransactionID"].astype(str).str.strip()
    df["UserID"]          = df["UserID"].astype(str).str.strip()
    df["CourseID"]        = df["CourseID"].astype(str).str.strip()
    df["TeacherID"]       = df["TeacherID"].astype(str).str.strip()
    df["PaymentMethod"]   = df["PaymentMethod"].astype(str).str.strip()
    df["Amount"]          = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")

    df.dropna(subset=["TransactionID", "CourseID", "TeacherID"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Integration / Merge
# ---------------------------------------------------------------------------

def build_master_dataset(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Merge Teachers ↔ Courses ↔ Transactions into one master DataFrame.

    Join logic
    ----------
    1. Aggregate Transactions → per (TeacherID, CourseID) enrollment count.
    2. Left-join Courses onto enrollment aggregation on CourseID.
    3. Left-join Teachers onto result on TeacherID.

    Returns
    -------
    pd.DataFrame with all instructor, course, and enrollment columns.
    """
    teachers     = load_teachers(filepath)
    courses      = load_courses(filepath)
    transactions = load_transactions(filepath)

    # --- Step 1: enrollment count per (TeacherID, CourseID) -----------------
    enrollments = (
        transactions
        .groupby(["TeacherID", "CourseID"], as_index=False)
        .agg(EnrollmentCount=("TransactionID", "count"))
    )

    # --- Step 2: merge enrollments with courses on CourseID -----------------
    courses_enr = pd.merge(enrollments, courses, on="CourseID", how="left")

    # --- Step 3: merge with teachers on TeacherID ---------------------------
    master = pd.merge(courses_enr, teachers, on="TeacherID", how="left")

    # Validation flags
    master["_has_teacher_rating"] = master["TeacherRating"].notna()
    master["_has_course_rating"]  = master["CourseRating"].notna()

    master.reset_index(drop=True, inplace=True)
    return master


def validate_mapping(master: pd.DataFrame) -> dict:
    """
    Run data-quality checks and return a validation report dict.
    """
    report = {}
    report["total_records"]            = len(master)
    report["unique_teachers"]          = master["TeacherID"].nunique()
    report["unique_courses"]           = master["CourseID"].nunique()
    report["missing_teacher_rating"]   = master["TeacherRating"].isna().sum()
    report["missing_course_rating"]    = master["CourseRating"].isna().sum()
    report["courses_without_teacher"]  = master[master["TeacherName"].isna()].shape[0]
    report["courses_without_rating"]   = master[master["CourseRating"].isna()].shape[0]
    report["mapping_accuracy_pct"]     = round(
        100 * master["_has_teacher_rating"].mean(), 2
    )
    return report


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def get_all_data(filepath: str = DEFAULT_DATA_PATH) -> dict[str, pd.DataFrame]:
    """
    Return a dict with keys: 'teachers', 'courses', 'transactions', 'master'.
    Single entry point for the Streamlit app.
    """
    teachers     = load_teachers(filepath)
    courses      = load_courses(filepath)
    transactions = load_transactions(filepath)
    master       = build_master_dataset(filepath)

    return {
        "teachers":     teachers,
        "courses":      courses,
        "transactions": transactions,
        "master":       master,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = get_all_data()
    print(f"Teachers     : {len(data['teachers'])} rows")
    print(f"Courses      : {len(data['courses'])} rows")
    print(f"Transactions : {len(data['transactions'])} rows")
    print(f"Master       : {len(data['master'])} rows")
    report = validate_mapping(data["master"])
    print("\nValidation Report:")
    for k, v in report.items():
        print(f"  {k:<35} {v}")
