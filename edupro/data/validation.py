"""
edupro.data.validation
======================
Comprehensive validation engine with PASS/WARN/FAIL/INFO checks, referential integrity,
domain bounds, relationship cardinality auditing, and structured reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from edupro.config import settings
from edupro.data.schema import (
    COURSES_COLS,
    PRIMARY_KEYS,
    REQUIRED_KEYS,
    TEACHERS_COLS,
    TRANSACTIONS_COLS,
    USERS_COLS,
)


class CheckStatus(str, Enum):
    """Validation result statuses."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    INFO = "INFO"


@dataclass(frozen=True)
class CheckResult:
    """Individual validation check outcome."""

    name: str
    category: str
    status: CheckStatus
    value: Any
    detail: str


class ValidationReport:
    """Aggregated validation report across all validation categories."""

    def __init__(self, results: list[CheckResult]) -> None:
        self.results = list(results)

    @property
    def has_failures(self) -> bool:
        """True if any check returned FAIL."""
        return any(r.status == CheckStatus.FAIL for r in self.results)

    @property
    def has_warnings(self) -> bool:
        """True if any check returned WARN."""
        return any(r.status == CheckStatus.WARN for r in self.results)

    def failures(self) -> list[CheckResult]:
        """Return all failed checks."""
        return [r for r in self.results if r.status == CheckStatus.FAIL]

    def warnings(self) -> list[CheckResult]:
        """Return all warning checks."""
        return [r for r in self.results if r.status == CheckStatus.WARN]

    def to_frame(self) -> pd.DataFrame:
        """Export checks as a pandas DataFrame."""
        return pd.DataFrame([
            {
                "category": r.category,
                "name": r.name,
                "status": r.status.value,
                "value": str(r.value),
                "detail": r.detail,
            }
            for r in self.results
        ])

    def summary(self) -> str:
        """Human-readable formatted summary table."""
        lines = [
            f"{'Category':<22} | {'Check Name':<38} | {'Status':<6} | Detail",
            "-" * 95,
        ]
        for r in self.results:
            lines.append(
                f"{r.category:<22} | {r.name:<38} | {r.status.value:<6} | {r.detail}"
            )
        status_counts = pd.Series([r.status.value for r in self.results]).value_counts().to_dict()
        lines.append("-" * 95)
        lines.append(f"Total Checks: {len(self.results)} | Counts: {status_counts}")
        return "\n".join(lines)


def validate_mapping(master: pd.DataFrame) -> dict[str, Any]:
    """
    Backward-compatible summary for Stage 1 data integration.
    """
    has_teacher_rating = master["TeacherRating"].notna() if "TeacherRating" in master.columns else pd.Series(False, index=master.index)
    mapping_accuracy = round(100.0 * float(has_teacher_rating.mean()), 2) if len(master) > 0 else 0.0

    return {
        "total_records": len(master),
        "unique_teachers": master["TeacherID"].nunique() if "TeacherID" in master.columns else 0,
        "unique_courses": master["CourseID"].nunique() if "CourseID" in master.columns else 0,
        "missing_teacher_rating": int(master["TeacherRating"].isna().sum()) if "TeacherRating" in master.columns else 0,
        "missing_course_rating": int(master["CourseRating"].isna().sum()) if "CourseRating" in master.columns else 0,
        "courses_without_teacher": int(master[master["TeacherName"].isna()].shape[0]) if "TeacherName" in master.columns else 0,
        "courses_without_rating": int(master[master["CourseRating"].isna()].shape[0]) if "CourseRating" in master.columns else 0,
        "mapping_accuracy_pct": mapping_accuracy,
    }


def validate_all(tables: dict[str, pd.DataFrame]) -> ValidationReport:
    """
    Run comprehensive data validation checks across Teachers, Courses, Transactions, and Users.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Cleaned DataFrames keyed by 'teachers', 'courses', 'transactions', 'users'.

    Returns
    -------
    ValidationReport
        Validation report containing all check results.
    """
    results: list[CheckResult] = []

    teachers = tables.get("teachers", pd.DataFrame())
    courses = tables.get("courses", pd.DataFrame())
    transactions = tables.get("transactions", pd.DataFrame())
    users = tables.get("users", pd.DataFrame())

    # -------------------------------------------------------------------------
    # 1. Schema & Column Presence
    # -------------------------------------------------------------------------
    sheets_specs = [
        ("Teachers", teachers, TEACHERS_COLS),
        ("Courses", courses, COURSES_COLS),
        ("Transactions", transactions, TRANSACTIONS_COLS),
        ("Users", users, USERS_COLS),
    ]
    for sheet_name, df, expected_cols in sheets_specs:
        missing = [c for c in expected_cols if c not in df.columns]
        status = CheckStatus.FAIL if missing else CheckStatus.PASS
        results.append(
            CheckResult(
                name=f"Schema columns: {sheet_name}",
                category="Schema",
                status=status,
                value=f"{len(df.columns)} columns",
                detail="All required columns present" if not missing else f"Missing: {missing}",
            )
        )

    # -------------------------------------------------------------------------
    # 2. Duplicate Primary Keys
    # -------------------------------------------------------------------------
    pk_specs = [
        ("Teachers", teachers, PRIMARY_KEYS["Teachers"]),
        ("Courses", courses, PRIMARY_KEYS["Courses"]),
        ("Transactions", transactions, PRIMARY_KEYS["Transactions"]),
        ("Users", users, PRIMARY_KEYS["Users"]),
    ]
    for sheet_name, df, pk in pk_specs:
        if pk in df.columns:
            dups = int(df.duplicated(subset=[pk]).sum())
            status = CheckStatus.FAIL if dups > 0 else CheckStatus.PASS
            results.append(
                CheckResult(
                    name=f"Unique Primary Key: {sheet_name}.{pk}",
                    category="Completeness",
                    status=status,
                    value=dups,
                    detail="No duplicate primary keys" if dups == 0 else f"{dups} duplicate primary keys found",
                )
            )

    # -------------------------------------------------------------------------
    # 3. Null Rates on Required Keys
    # -------------------------------------------------------------------------
    req_specs = [
        ("Teachers", teachers, REQUIRED_KEYS["Teachers"]),
        ("Courses", courses, REQUIRED_KEYS["Courses"]),
        ("Transactions", transactions, REQUIRED_KEYS["Transactions"]),
        ("Users", users, REQUIRED_KEYS["Users"]),
    ]
    for sheet_name, df, req_cols in req_specs:
        for col in req_cols:
            if col in df.columns:
                nulls = int(df[col].isna().sum())
                status = CheckStatus.FAIL if nulls > 0 else CheckStatus.PASS
                results.append(
                    CheckResult(
                        name=f"Null check: {sheet_name}.{col}",
                        category="Completeness",
                        status=status,
                        value=nulls,
                        detail="Zero null values" if nulls == 0 else f"{nulls} missing values in required key",
                    )
                )

    # Check CoursePrice and Amount nulls
    if "CoursePrice" in courses.columns:
        null_prices = int(courses["CoursePrice"].isna().sum())
        status = CheckStatus.WARN if null_prices > 0 else CheckStatus.PASS
        results.append(
            CheckResult(
                name="Null check: Courses.CoursePrice",
                category="Completeness",
                status=status,
                value=null_prices,
                detail="Zero null prices" if null_prices == 0 else f"{null_prices} courses have missing CoursePrice",
            )
        )

    if "Amount" in transactions.columns:
        null_amounts = int(transactions["Amount"].isna().sum())
        status = CheckStatus.WARN if null_amounts > 0 else CheckStatus.PASS
        results.append(
            CheckResult(
                name="Null check: Transactions.Amount",
                category="Completeness",
                status=status,
                value=null_amounts,
                detail="Zero null amounts" if null_amounts == 0 else f"{null_amounts} transactions have missing Amount",
            )
        )

    # -------------------------------------------------------------------------
    # 4. Referential Integrity
    # -------------------------------------------------------------------------
    if not transactions.empty:
        total_tx = len(transactions)

        # TeacherID in Teachers
        orphan_teachers = int((~transactions["TeacherID"].isin(teachers["TeacherID"])).sum()) if "TeacherID" in teachers.columns else total_tx
        results.append(
            CheckResult(
                name="Referential Integrity: Transactions.TeacherID",
                category="Referential Integrity",
                status=CheckStatus.FAIL if orphan_teachers > 0 else CheckStatus.PASS,
                value=orphan_teachers,
                detail=f"{orphan_teachers} orphan TeacherID references" if orphan_teachers > 0 else "All TeacherIDs valid",
            )
        )

        # CourseID in Courses
        orphan_courses = int((~transactions["CourseID"].isin(courses["CourseID"])).sum()) if "CourseID" in courses.columns else total_tx
        results.append(
            CheckResult(
                name="Referential Integrity: Transactions.CourseID",
                category="Referential Integrity",
                status=CheckStatus.FAIL if orphan_courses > 0 else CheckStatus.PASS,
                value=orphan_courses,
                detail=f"{orphan_courses} orphan CourseID references" if orphan_courses > 0 else "All CourseIDs valid",
            )
        )

        # UserID in Users
        orphan_users = int((~transactions["UserID"].isin(users["UserID"])).sum()) if "UserID" in users.columns else total_tx
        results.append(
            CheckResult(
                name="Referential Integrity: Transactions.UserID",
                category="Referential Integrity",
                status=CheckStatus.FAIL if orphan_users > 0 else CheckStatus.PASS,
                value=orphan_users,
                detail=f"{orphan_users} orphan UserID references" if orphan_users > 0 else "All UserIDs valid",
            )
        )

        valid_tx = int(
            (
                transactions["TeacherID"].isin(teachers["TeacherID"])
                & transactions["CourseID"].isin(courses["CourseID"])
                & transactions["UserID"].isin(users["UserID"])
            ).sum()
        )
        ref_integrity_pct = round(100.0 * valid_tx / total_tx, 2) if total_tx > 0 else 0.0
        results.append(
            CheckResult(
                name="Referential Integrity %",
                category="Referential Integrity",
                status=CheckStatus.PASS if ref_integrity_pct == 100.0 else CheckStatus.FAIL,
                value=f"{ref_integrity_pct}%",
                detail=f"{valid_tx}/{total_tx} transactions fully resolve across all parent entities",
            )
        )

    # -------------------------------------------------------------------------
    # 5. Ranges & Domain Constraints
    # -------------------------------------------------------------------------
    # Rating range: 1.0 to 5.0
    if "TeacherRating" in teachers.columns:
        invalid_tr = int(((teachers["TeacherRating"] < settings.min_rating) | (teachers["TeacherRating"] > settings.max_rating)).sum())
        results.append(
            CheckResult(
                name="Range: TeacherRating in [1.0, 5.0]",
                category="Range",
                status=CheckStatus.FAIL if invalid_tr > 0 else CheckStatus.PASS,
                value=invalid_tr,
                detail=f"{invalid_tr} out-of-range TeacherRatings" if invalid_tr > 0 else "All TeacherRatings within [1.0, 5.0]",
            )
        )

    if "CourseRating" in courses.columns:
        invalid_cr = int(((courses["CourseRating"] < settings.min_rating) | (courses["CourseRating"] > settings.max_rating)).sum())
        results.append(
            CheckResult(
                name="Range: CourseRating in [1.0, 5.0]",
                category="Range",
                status=CheckStatus.FAIL if invalid_cr > 0 else CheckStatus.PASS,
                value=invalid_cr,
                detail=f"{invalid_cr} out-of-range CourseRatings" if invalid_cr > 0 else "All CourseRatings within [1.0, 5.0]",
            )
        )

    # Age plausible bounds: Teachers [min_teacher_age, max_age], Users [min_user_age, max_age]
    age_specs = [
        ("Teachers", teachers, settings.min_teacher_age, settings.max_age),
        ("Users", users, settings.min_user_age, settings.max_age),
    ]
    for sheet_name, df, min_a, max_a in age_specs:
        if "Age" in df.columns:
            age_series = df["Age"].dropna()
            invalid_age = int(((age_series < min_a) | (age_series > max_a)).sum())
            results.append(
                CheckResult(
                    name=f"Range: {sheet_name}.Age in [{min_a}, {max_a}]",
                    category="Range",
                    status=CheckStatus.FAIL if invalid_age > 0 else CheckStatus.PASS,
                    value=invalid_age,
                    detail=f"{invalid_age} ages outside [{min_a}, {max_a}]" if invalid_age > 0 else f"All ages within [{min_a}, {max_a}]",
                )
            )

    # Experience >= 0 and Age - Experience >= 18
    if "YearsOfExperience" in teachers.columns:
        neg_exp = int((teachers["YearsOfExperience"] < 0).sum())
        results.append(
            CheckResult(
                name="Range: Teachers.YearsOfExperience >= 0",
                category="Range",
                status=CheckStatus.FAIL if neg_exp > 0 else CheckStatus.PASS,
                value=neg_exp,
                detail=f"{neg_exp} negative experience values" if neg_exp > 0 else "All experience values non-negative",
            )
        )

        if "Age" in teachers.columns:
            exp_check = teachers.dropna(subset=["Age", "YearsOfExperience"])
            invalid_career_start = int((exp_check["Age"] - exp_check["YearsOfExperience"] < 18).sum())
            results.append(
                CheckResult(
                    name="Constraint: Age - YearsOfExperience >= 18",
                    category="Range",
                    status=CheckStatus.WARN if invalid_career_start > 0 else CheckStatus.PASS,
                    value=invalid_career_start,
                    detail=f"{invalid_career_start} instructors with implied career start age < 18" if invalid_career_start > 0 else "All instructors career start age >= 18",
                )
            )

    # Non-negative prices, amounts, duration
    if "CoursePrice" in courses.columns:
        neg_prices = int((courses["CoursePrice"].dropna() < 0).sum())
        results.append(
            CheckResult(
                name="Range: Courses.CoursePrice >= 0",
                category="Range",
                status=CheckStatus.FAIL if neg_prices > 0 else CheckStatus.PASS,
                value=neg_prices,
                detail=f"{neg_prices} negative course prices" if neg_prices > 0 else "All course prices non-negative",
            )
        )

    if "CourseDuration" in courses.columns:
        neg_dur = int((courses["CourseDuration"].dropna() <= 0).sum())
        results.append(
            CheckResult(
                name="Range: Courses.CourseDuration > 0",
                category="Range",
                status=CheckStatus.FAIL if neg_dur > 0 else CheckStatus.PASS,
                value=neg_dur,
                detail=f"{neg_dur} non-positive durations" if neg_dur > 0 else "All course durations positive",
            )
        )

    if "Amount" in transactions.columns:
        neg_amounts = int((transactions["Amount"].dropna() < 0).sum())
        results.append(
            CheckResult(
                name="Range: Transactions.Amount >= 0",
                category="Range",
                status=CheckStatus.FAIL if neg_amounts > 0 else CheckStatus.PASS,
                value=neg_amounts,
                detail=f"{neg_amounts} negative transaction amounts" if neg_amounts > 0 else "All amounts non-negative",
            )
        )

    # Transaction date parse & future check
    if "TransactionDate" in transactions.columns:
        invalid_dates = int(transactions["TransactionDate"].isna().sum())
        now_ts = pd.Timestamp.now() + pd.Timedelta(days=1)
        future_dates = int((transactions["TransactionDate"].dropna() > now_ts).sum())
        date_status = CheckStatus.FAIL if (invalid_dates > 0 or future_dates > 0) else CheckStatus.PASS
        results.append(
            CheckResult(
                name="Range: Transactions.TransactionDate valid & not in future",
                category="Range",
                status=date_status,
                value=f"unparsed: {invalid_dates}, future: {future_dates}",
                detail="All transaction dates parse and are not in the future" if date_status == CheckStatus.PASS else f"Found {invalid_dates} unparsed and {future_dates} future dates",
            )
        )

    # -------------------------------------------------------------------------
    # 6. Consistency Checks
    # -------------------------------------------------------------------------
    # Amount equals CoursePrice
    if "Amount" in transactions.columns and "CoursePrice" in courses.columns and "CourseID" in transactions.columns:
        merged_prices = transactions[["CourseID", "Amount"]].merge(
            courses[["CourseID", "CoursePrice"]],
            on="CourseID",
            how="left",
        )
        price_mismatches = int((merged_prices["Amount"] != merged_prices["CoursePrice"]).sum())
        results.append(
            CheckResult(
                name="Consistency: Transactions.Amount == CoursePrice",
                category="Consistency",
                status=CheckStatus.FAIL if price_mismatches > 0 else CheckStatus.PASS,
                value=price_mismatches,
                detail=f"{price_mismatches} price mismatches between transactions and course catalog" if price_mismatches > 0 else "Zero price mismatches (100% exact match)",
            )
        )

    # Duplicate (UserID, CourseID) enrollments
    if "UserID" in transactions.columns and "CourseID" in transactions.columns:
        dup_enrollments = int(transactions.duplicated(subset=["UserID", "CourseID"]).sum())
        results.append(
            CheckResult(
                name="Consistency: Duplicate (UserID, CourseID) Enrollments",
                category="Consistency",
                status=CheckStatus.WARN if dup_enrollments > 0 else CheckStatus.PASS,
                value=dup_enrollments,
                detail=f"{dup_enrollments} duplicate user-course enrollments" if dup_enrollments > 0 else "Zero duplicate enrollments per user-course pair",
            )
        )

    # -------------------------------------------------------------------------
    # 7. Relationship Cardinality (INFO)
    # -------------------------------------------------------------------------
    if not transactions.empty and "TeacherID" in transactions.columns and "CourseID" in transactions.columns:
        tpc = transactions.groupby("CourseID")["TeacherID"].nunique()
        cpt = transactions.groupby("TeacherID")["CourseID"].nunique()
        pairs = transactions.groupby(["TeacherID", "CourseID"])["TransactionID"].count()

        observed_pairs = len(pairs)
        possible_pairs = 3600  # 60 * 60
        pair_density_pct = round(100.0 * observed_pairs / possible_pairs, 2)
        share_le2_pct = round(100.0 * float((pairs <= 2).mean()), 2)

        cardinality_detail = (
            f"Teachers/course: min={tpc.min()}, med={tpc.median():.0f}, max={tpc.max()}; "
            f"Courses/teacher: min={cpt.min()}, med={cpt.median():.0f}, max={cpt.max()}; "
            f"Observed pairs={observed_pairs}/{possible_pairs} ({pair_density_pct}% density); "
            f"Pairs with <=2 enrollments={share_le2_pct}%. "
            "Course-instructor relationship is many-to-many and derived from Transactions; "
            "a course rating cannot be attributed to a single instructor."
        )

        results.append(
            CheckResult(
                name="Relationship Cardinality & Derivation",
                category="Relationship",
                status=CheckStatus.INFO,
                value=f"{observed_pairs} pairs ({pair_density_pct}%)",
                detail=cardinality_detail,
            )
        )

    # -------------------------------------------------------------------------
    # 8. Expertise Alignment (INFO)
    # -------------------------------------------------------------------------
    if (
        not transactions.empty
        and "TeacherID" in teachers.columns
        and "Expertise" in teachers.columns
        and "CourseID" in courses.columns
        and "CourseCategory" in courses.columns
    ):
        enr_align = transactions.merge(
            teachers[["TeacherID", "Expertise"]], on="TeacherID", how="left"
        ).merge(
            courses[["CourseID", "CourseCategory"]], on="CourseID", how="left"
        )
        alignment_rate = float((enr_align["Expertise"] == enr_align["CourseCategory"]).mean())
        align_pct = round(100.0 * alignment_rate, 2)
        results.append(
            CheckResult(
                name="Expertise Alignment: Teacher.Expertise == Course.CourseCategory",
                category="Expertise Alignment",
                status=CheckStatus.INFO,
                value=f"{align_pct}%",
                detail=f"{align_pct}% of enrollments align with instructor expertise (vs ~8.3% random baseline across 12 categories)",
            )
        )

    # -------------------------------------------------------------------------
    # 9. Small Groups (WARN)
    # -------------------------------------------------------------------------
    if not teachers.empty and "Expertise" in teachers.columns:
        exp_counts = teachers["Expertise"].value_counts()
        small_groups = exp_counts[exp_counts < settings.min_group_size].to_dict()
        if small_groups:
            small_desc = ", ".join(f"{k} ({v})" for k, v in small_groups.items())
            results.append(
                CheckResult(
                    name=f"Small Groups: Expertise < {settings.min_group_size} instructors",
                    category="Group Size",
                    status=CheckStatus.WARN,
                    value=len(small_groups),
                    detail=f"Expertise categories with < {settings.min_group_size} instructors: {small_desc}",
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"Small Groups: Expertise < {settings.min_group_size} instructors",
                    category="Group Size",
                    status=CheckStatus.PASS,
                    value=0,
                    detail="All expertise categories have sufficient sample sizes",
                )
            )

    return ValidationReport(results)
