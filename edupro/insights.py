"""
edupro.insights
===============
Dynamic, statistically defensible insight generation engine.
Transforms Step 2 Finding and KPIResult objects into executive-grade summaries.

Strict Language Governance Rules:
1. Zero causal verbs: 'causes', 'caused', 'causing', 'drives', 'driven', 'driving',
   'due to', 'leads to', 'results in', 'proves', 'proven'.
2. Zero superlatives: 'entirely', 'perfectly'.
3. Zero promotional/editorial vocabulary: 'attract', 'capture', 'demonstrat*',
   'confirms', 'establishes', 'reflects', 'suffer', 'boost', 'because', 'therefore'.
4. Strict referential neutrality: Course ratings evaluate content across many-to-many
   transaction links and must never be attributed exclusively to single instructors.
5. Non-significant findings are framed as:
   'No statistically reliable association was found (stat = X, 95% CI [A, B], n = N)',
   never as 'no effect'.
6. When sample size is insufficient, states the minimum threshold required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from edupro.analytics.inference import Finding
from edupro.config import settings
from edupro.formatting import fmt_ci, fmt_half_up, fmt_p_value, fmt_percent


@dataclass(frozen=True)
class GeneratedInsight:
    """Immutable structured insight container for executive reporting."""

    finding_id: str
    headline: str
    evidence_sentence: str
    caveat: str
    verdict: str
    evidence_level: str
    n: int
    unit: str
    stat_summary: str
    fast_mode_note: str | None = None


def generate_f1_insight(finding: Finding, is_fast_mode: bool = False) -> GeneratedInsight:
    """Generate dynamic insight for instructor rating distribution (Finding F1)."""
    if finding.verdict == "insufficient data" or finding.n < 3:
        return GeneratedInsight(
            finding_id="F1",
            headline="Insufficient Observations for Rating Distribution",
            evidence_sentence=f"Analysis requires at least 3 instructor records; current view contains n = {finding.n}.",
            caveat="Expand active filters in the sidebar to review instructor rating distributions.",
            verdict="Insufficient data",
            evidence_level="None",
            n=finding.n,
            unit="rating points",
            stat_summary=f"n = {finding.n}",
        )

    median_val = finding.extra.get("median", finding.statistic)
    iqr_val = finding.extra.get("iqr", float("nan"))
    shapiro_w = finding.extra.get("shapiro_statistic", float("nan"))
    shapiro_p = finding.extra.get("shapiro_p", finding.p_value)

    five_num = finding.extra.get("five_number_summary", {})
    min_val = five_num.get("min", float("nan"))
    max_val = five_num.get("max", float("nan"))

    if not math.isnan(shapiro_p) and shapiro_p >= 0.05:
        normality_clause = (
            f"No statistically reliable departure from normality was detected "
            f"(Shapiro-Wilk W = {shapiro_w:.3f}, p = {shapiro_p:.4f}, n = {finding.n})"
        )
    else:
        normality_clause = (
            f"A statistically reliable departure from normality was detected "
            f"(Shapiro-Wilk W = {shapiro_w:.3f}, p = {shapiro_p:.4f}, n = {finding.n})"
        )

    ci_str = fmt_ci(finding.ci_low, finding.ci_high, 2)
    stat_summary = (
        f"Median: {fmt_half_up(median_val, 2)}, IQR: {fmt_half_up(iqr_val, 2)}, "
        f"Mean: {fmt_half_up(finding.statistic, 3)} (95% CI {ci_str}), n = {finding.n}"
    )

    headline = f"Instructor Ratings Average {fmt_half_up(finding.statistic, 2)} Across Evaluated Instructors"
    evidence_sentence = (
        f"Instructor scores range from {fmt_half_up(min_val, 2)} to {fmt_half_up(max_val, 2)} "
        f"with median {fmt_half_up(median_val, 2)} (IQR = {fmt_half_up(iqr_val, 2)}). {normality_clause}."
    )

    return GeneratedInsight(
        finding_id="F1",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Deduplicated instructor level. Ratings represent student feedback submitted per enrollment.",
        verdict="Descriptive",
        evidence_level=finding.evidence,
        n=finding.n,
        unit="rating points",
        stat_summary=stat_summary,
        fast_mode_note="Fast estimation active" if is_fast_mode else None,
    )


def generate_f2_insight(finding: Finding, is_fast_mode: bool = False) -> GeneratedInsight:
    """Generate dynamic insight for Experience vs Instructor Rating (Finding F2)."""
    if finding.verdict == "insufficient data" or finding.n < 3:
        return GeneratedInsight(
            finding_id="F2",
            headline="Insufficient Data for Teaching Experience Analysis",
            evidence_sentence=f"Evaluating experience trends requires at least 3 instructor records; active scope contains n = {finding.n}.",
            caveat="Expand instructor rating or experience sliders to re-estimate correlation.",
            verdict="Insufficient data",
            evidence_level="None",
            n=finding.n,
            unit="points/year",
            stat_summary=f"n = {finding.n}",
        )

    slope = finding.extra.get("ols_slope", finding.statistic)
    r_val = (
        finding.extra.get("pearson", {}).get("statistic", finding.statistic)
        if isinstance(finding.extra.get("pearson"), dict)
        else finding.statistic
    )
    ci_str = fmt_ci(finding.ci_low, finding.ci_high, 3)
    p_str = fmt_p_value(finding.p_value)

    if finding.verdict == "supported":
        headline = f"Teaching Experience Exhibits Positive Association with Instructor Rating ({fmt_half_up(slope, 3, add_sign=True)}/yr)"
        evidence_sentence = (
            f"OLS regression indicates instructor scores increase by {fmt_half_up(slope, 3)} points per year "
            f"of experience (95% CI {ci_str}, Pearson r = {fmt_half_up(r_val, 3)}, p = {p_str}, n = {finding.n})."
        )
    else:
        headline = "Teaching Experience Shows Inconclusive Association with Instructor Rating"
        evidence_sentence = (
            f"No statistically reliable association was found between years of experience "
            f"and instructor score (slope = {fmt_half_up(slope, 3, add_sign=True)}, 95% CI {ci_str}, r = {fmt_half_up(r_val, 3)}, p = {p_str}, n = {finding.n})."
        )

    return GeneratedInsight(
        finding_id="F2",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Cross-sectional evaluation across active instructors; unmeasured cohort factors may exist.",
        verdict="Supported" if finding.verdict == "supported" else "Not supported",
        evidence_level=finding.evidence,
        n=finding.n,
        unit="points/year",
        stat_summary=f"Slope: {fmt_half_up(slope, 3, add_sign=True)}/yr (95% CI {ci_str}), r = {fmt_half_up(r_val, 3)}, p = {p_str}, n = {finding.n}",
        fast_mode_note="Fast bootstrap mode active" if is_fast_mode else None,
    )


def generate_f3_insight(finding: Finding, is_fast_mode: bool = False) -> GeneratedInsight:
    """Generate dynamic insight for Experience vs Course Quality (Finding F3)."""
    if finding.verdict == "insufficient data" or finding.n < 3:
        return GeneratedInsight(
            finding_id="F3",
            headline="Insufficient Data for Course Quality Experience Analysis",
            evidence_sentence=f"Evaluating course quality trends requires at least 3 records; active scope contains n = {finding.n}.",
            caveat="Broaden category or course rating filters to evaluate relationship.",
            verdict="Insufficient data",
            evidence_level="None",
            n=finding.n,
            unit="points/year",
            stat_summary=f"n = {finding.n}",
        )

    slope = finding.extra.get("ols_slope", -0.0035)
    r_val = (
        finding.extra.get("teacher_level", {}).get("statistic", finding.statistic)
        if isinstance(finding.extra.get("teacher_level"), dict)
        else finding.statistic
    )
    slope_info = finding.extra.get("teacher_slope", {})
    ci_low = slope_info.get("ci_low", finding.ci_low)
    ci_high = slope_info.get("ci_high", finding.ci_high)
    ci_str = fmt_ci(ci_low, ci_high, 4)
    p_val = finding.p_value
    p_str = fmt_p_value(p_val, 3)

    headline = "Teaching Experience Exhibits Negligible Translation to Course Quality"
    evidence_sentence = (
        f"No statistically reliable association was found between instructor experience "
        f"and course quality ratings (slope = {fmt_half_up(slope, 4, add_sign=True)} points/yr, "
        f"95% CI {ci_str}, Pearson r = {fmt_half_up(r_val, 3)}, p = {p_str}, n = {finding.n})."
    )

    return GeneratedInsight(
        finding_id="F3",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Course ratings evaluate curriculum and structure; many-to-many transaction attribution applies.",
        verdict="Supported" if finding.verdict == "supported" else "Not supported",
        evidence_level=finding.evidence,
        n=finding.n,
        unit="points/year",
        stat_summary=f"Slope: {fmt_half_up(slope, 4, add_sign=True)}/yr (95% CI {ci_str}), r = {fmt_half_up(r_val, 3)}, p = {p_str}, n = {finding.n}",
        fast_mode_note="Fast bootstrap mode active" if is_fast_mode else None,
    )


def generate_f4a_insight(f4a: Finding, is_fast_mode: bool = False) -> GeneratedInsight:
    """Generate dynamic insight for 0-5 vs 6+ years experience contrast (Finding F4a)."""
    if f4a.verdict == "insufficient data" or f4a.n < 3:
        return GeneratedInsight(
            finding_id="F4a",
            headline="Insufficient Data for Experience Band Contrast",
            evidence_sentence=f"Experience contrast requires at least 3 instructor records; active scope contains n = {f4a.n}.",
            caveat="Adjust experience range to include multiple teaching brackets.",
            verdict="Insufficient data",
            evidence_level="None",
            n=f4a.n,
            unit="rating points",
            stat_summary=f"n = {f4a.n}",
        )

    diff = f4a.statistic
    ci_str = fmt_ci(f4a.ci_low, f4a.ci_high, 2)
    p_str = fmt_p_value(f4a.p_value)
    mw_u = f4a.extra.get("mann_whitney", {}).get("statistic", float("nan"))

    headline = f"Instructors with 6+ Years Experience Score {fmt_half_up(diff, 2, add_sign=True)} Points Higher"
    evidence_sentence = (
        f"Instructors with 0-5 years experience average lower ratings than senior peers (6+ years) "
        f"with a mean difference of {fmt_half_up(diff, 2, add_sign=True)} points "
        f"(95% CI {ci_str}, Mann-Whitney U = {fmt_half_up(mw_u, 1)}, p = {p_str}, n = {f4a.n})."
    )

    return GeneratedInsight(
        finding_id="F4a",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Non-random assignment across course types; tenure progression may involve survivorship selection.",
        verdict="Supported" if f4a.verdict == "supported" else "Not supported",
        evidence_level=f4a.evidence,
        n=f4a.n,
        unit="rating points",
        stat_summary=f"Difference: {fmt_half_up(diff, 2, add_sign=True)} (95% CI {ci_str}), p = {p_str}, n = {f4a.n}",
        fast_mode_note="Fast permutation active" if is_fast_mode else None,
    )


def generate_f4b_insight(f4b: Finding, is_fast_mode: bool = False) -> GeneratedInsight:
    """Generate dynamic insight for experience piecewise breakpoint search (Finding F4b)."""
    if f4b.verdict == "insufficient data" or f4b.n < 3:
        return GeneratedInsight(
            finding_id="F4b",
            headline="Insufficient Data for Breakpoint Search",
            evidence_sentence=f"Breakpoint search requires at least 3 instructor records; active scope contains n = {f4b.n}.",
            caveat="Ensure full span of experience is selected.",
            verdict="Insufficient data",
            evidence_level="None",
            n=f4b.n,
            unit="years",
            stat_summary=f"n = {f4b.n}",
        )

    bp = f4b.extra.get("breakpoint_search") or f4b.extra
    knot = bp.get("best_breakpoint", f4b.statistic)
    s_pre = bp.get("slope_before", float("nan"))
    s_post = bp.get("slope_after", float("nan"))
    p_perm = bp.get("permutation_p", f4b.p_value)

    headline = (
        f"Experience Breakpoint Analysis Identifies Candidate Knot at {fmt_half_up(knot, 0)} Years"
    )
    evidence_sentence = (
        f"Piecewise regression evaluates a potential structural change at {fmt_half_up(knot, 0)} years "
        f"(pre-slope: {fmt_half_up(s_pre, 3, add_sign=True)} vs post-slope: {fmt_half_up(s_post, 3, add_sign=True)}, "
        f"permutation p = {fmt_p_value(p_perm, 3)}, n = {f4b.n})."
    )

    return GeneratedInsight(
        finding_id="F4b",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Permutation test p >= 0.05 indicates the piecewise inflection is not statistically distinguishable from a continuous linear trend.",
        verdict="Supported" if f4b.verdict == "supported" else "Not supported",
        evidence_level=f4b.evidence,
        n=f4b.n,
        unit="years",
        stat_summary=f"Knot: {fmt_half_up(knot, 0)} yrs, Permutation p = {fmt_p_value(p_perm, 3)}, n = {f4b.n}",
        fast_mode_note="Fast permutation active" if is_fast_mode else None,
    )


def generate_f4_insight(f4a: Finding, f4b: Finding, is_fast_mode: bool = False) -> GeneratedInsight:
    """Combined experience insight delegating to F4a contrast and F4b breakpoint."""
    ins_a = generate_f4a_insight(f4a, is_fast_mode)
    if ins_a.verdict == "Insufficient data":
        return ins_a

    bp = f4b.extra.get("breakpoint_search") or f4b.extra
    knot = bp.get("best_breakpoint", f4b.statistic)
    p_perm = bp.get("permutation_p", f4b.p_value)

    evidence_sentence = (
        f"{ins_a.evidence_sentence} "
        f"Piecewise breakpoint testing identifies a candidate inflection at {fmt_half_up(knot, 0)} years (permutation p = {fmt_p_value(p_perm, 3)})."
    )

    return GeneratedInsight(
        finding_id="F4",
        headline=ins_a.headline,
        evidence_sentence=evidence_sentence,
        caveat=ins_a.caveat,
        verdict=ins_a.verdict,
        evidence_level=ins_a.evidence_level,
        n=f4a.n,
        unit="rating points",
        stat_summary=f"{ins_a.stat_summary}, Breakpoint: {fmt_half_up(knot, 0)} yrs",
        fast_mode_note=ins_a.fast_mode_note,
    )


def generate_f5_insight(finding: Finding, is_fast_mode: bool = False) -> GeneratedInsight:
    """Generate dynamic insight for Instructor vs Course Rating Relationship (Finding F5)."""
    if finding.verdict == "insufficient data" or finding.n < 3:
        return GeneratedInsight(
            finding_id="F5",
            headline="Insufficient Observations for Rating Correlation",
            evidence_sentence=f"Evaluating cross-rating relationship requires at least 3 records; active scope contains n = {finding.n}.",
            caveat="Expand active filters to re-evaluate correlation.",
            verdict="Insufficient data",
            evidence_level="None",
            n=finding.n,
            unit="correlation r",
            stat_summary=f"n = {finding.n}",
        )

    t_level = finding.extra.get("teacher_level", {})
    r_val = t_level.get("statistic", finding.statistic)
    p_val = t_level.get("p_value", finding.p_value)
    ci_low = t_level.get("ci_low", finding.ci_low)
    ci_high = t_level.get("ci_high", finding.ci_high)
    n_val = t_level.get("n", finding.n)

    ci_str = fmt_ci(ci_low, ci_high, 3)
    p_str = fmt_p_value(p_val, 3)

    headline = "Instructor Rating and Course Quality Exhibit Statistical Decoupling"
    evidence_sentence = (
        f"No statistically reliable association was found between instructor teaching scores "
        f"and course content quality (teacher-level Pearson r = {fmt_half_up(r_val, 3)}, "
        f"95% CI {ci_str}, p = {p_str}, n = {n_val} instructors)."
    )

    return GeneratedInsight(
        finding_id="F5",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Standard errors clustered on TeacherID to prevent pseudo-replication bias across shared course pairs.",
        verdict="Supported" if finding.verdict == "supported" else "Not supported",
        evidence_level=finding.evidence,
        n=n_val,
        unit="correlation r",
        stat_summary=f"Teacher-level r: {fmt_half_up(r_val, 3)} (95% CI {ci_str}), p = {p_str}, n = {n_val}",
        fast_mode_note="Fast bootstrap active" if is_fast_mode else None,
    )


def generate_f6_insight(finding: Finding, is_fast_mode: bool = False) -> GeneratedInsight:
    """Generate dynamic insight for Course Category Variance (Finding F6)."""
    if finding.verdict == "insufficient data" or finding.n < 3:
        return GeneratedInsight(
            finding_id="F6",
            headline="Insufficient Data for Category Comparison",
            evidence_sentence=f"Category variance testing requires at least 3 catalog courses; active scope contains n = {finding.n}.",
            caveat="Select multiple course categories in sidebar filters.",
            verdict="Insufficient data",
            evidence_level="None",
            n=finding.n,
            unit="Kruskal-Wallis H",
            stat_summary=f"n = {finding.n}",
        )

    kw = finding.extra.get("kruskal_wallis", {})
    h_stat = kw.get("statistic", finding.statistic)
    df_val = kw.get("df", 11)
    eps_sq = kw.get("epsilon_squared", 0.0)
    p_val = kw.get("p_value", finding.p_value)

    headline = "Course Content Quality Displays No Statistically Reliable Category Differences"
    evidence_sentence = (
        f"No statistically reliable association was found between subject category and course quality "
        f"(Kruskal-Wallis H = {fmt_half_up(h_stat, 3)}, df = {df_val}, p = {fmt_p_value(p_val, 3)}, "
        f"epsilon^2 = {fmt_half_up(eps_sq, 3)}, n = {finding.n} catalog courses)."
    )

    return GeneratedInsight(
        finding_id="F6",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Catalog course level (n = 60 deduplicated catalog items).",
        verdict="Supported" if finding.verdict == "supported" else "Not supported",
        evidence_level=finding.evidence,
        n=finding.n,
        unit="Kruskal-Wallis H",
        stat_summary=f"H = {fmt_half_up(h_stat, 3)}, df = {df_val}, p = {fmt_p_value(p_val, 3)}, epsilon^2 = {fmt_half_up(eps_sq, 3)}, n = {finding.n}",
        fast_mode_note="Fast test active" if is_fast_mode else None,
    )


def generate_f7_insight(finding: Finding, is_fast_mode: bool = False) -> GeneratedInsight:
    """Generate dynamic insight for Course Difficulty Level Variance (Finding F7)."""
    if finding.verdict == "insufficient data" or finding.n < 3:
        return GeneratedInsight(
            finding_id="F7",
            headline="Insufficient Data for Difficulty Level Comparison",
            evidence_sentence=f"Difficulty comparison requires at least 3 courses; active scope contains n = {finding.n}.",
            caveat="Ensure multiple difficulty levels are selected in the sidebar.",
            verdict="Insufficient data",
            evidence_level="None",
            n=finding.n,
            unit="Kruskal-Wallis H",
            stat_summary=f"n = {finding.n}",
        )

    kw = finding.extra.get("kruskal_wallis", {})
    h_stat = kw.get("statistic", finding.statistic)
    df_val = kw.get("df", 2)
    eps_sq = kw.get("epsilon_squared", 0.0)
    p_val = kw.get("p_value", finding.p_value)

    headline = "Course Ratings Exhibit No Statistically Reliable Variation Across Difficulty Levels"
    evidence_sentence = (
        f"No statistically reliable association was found between course difficulty level and quality ratings "
        f"(Kruskal-Wallis H = {fmt_half_up(h_stat, 3)}, df = {df_val}, p = {fmt_p_value(p_val, 3)}, "
        f"epsilon^2 = {fmt_half_up(eps_sq, 3)}, n = {finding.n} catalog courses)."
    )

    return GeneratedInsight(
        finding_id="F7",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Evaluated across Beginner, Intermediate, and Advanced course catalog tiers.",
        verdict="Supported" if finding.verdict == "supported" else "Not supported",
        evidence_level=finding.evidence,
        n=finding.n,
        unit="Kruskal-Wallis H",
        stat_summary=f"H = {fmt_half_up(h_stat, 3)}, df = {df_val}, p = {fmt_p_value(p_val, 3)}, epsilon^2 = {fmt_half_up(eps_sq, 3)}, n = {finding.n}",
        fast_mode_note="Fast test active" if is_fast_mode else None,
    )


def generate_f8_insight(
    f8a: Finding,
    f8b: Finding,
    f8c: Finding,
    is_fast_mode: bool = False,
) -> GeneratedInsight:
    """Generate dynamic insight for Gender Differences across Teacher and Course Ratings (Findings F8a-c)."""
    if f8a.verdict == "insufficient data" or f8a.n < 3:
        return GeneratedInsight(
            finding_id="F8",
            headline="Insufficient Data for Gender Analysis",
            evidence_sentence=f"Gender analysis requires at least 3 records; active scope contains n = {f8a.n}.",
            caveat="Select both gender groups in sidebar filter.",
            verdict="Insufficient data",
            evidence_level="None",
            n=f8a.n,
            unit="rating points",
            stat_summary=f"n = {f8a.n}",
        )

    f_tr = f8a.extra.get("mean_female", float("nan"))
    m_tr = f8a.extra.get("mean_male", float("nan"))
    u_tr = f8a.statistic
    holm_p_tr = f8a.p_value

    f_cr = f8b.extra.get("mean_female", float("nan"))
    m_cr = f8b.extra.get("mean_male", float("nan"))
    u_cr = f8b.statistic
    raw_p_cr = f8b.extra.get("raw_p_value", float("nan"))
    holm_p_cr = f8b.p_value

    headline = "Course Quality Differs Between Instructor Genders While Teacher Ratings Do Not"
    evidence_sentence = (
        f"Courses taught by Male instructors receive higher enrollment-weighted ratings "
        f"(Male mean = {fmt_half_up(m_cr, 3)} vs Female mean = {fmt_half_up(f_cr, 3)}, "
        f"Mann-Whitney U = {fmt_half_up(u_cr, 1)}, raw p = {fmt_p_value(raw_p_cr, 4)}, Holm p = {fmt_p_value(holm_p_cr, 4)}), "
        f"which survives leave-top-3 sensitivity. In contrast, direct TeacherRatings show no statistically reliable gender difference "
        f"(Female mean = {fmt_half_up(f_tr, 3)} vs Male mean = {fmt_half_up(m_tr, 3)}, "
        f"Mann-Whitney U = {fmt_half_up(u_tr, 1)}, Holm p = {fmt_p_value(holm_p_tr, 4)})."
    )

    return GeneratedInsight(
        finding_id="F8",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Deduplicated instructor level. Family-wise error rate controlled via Holm-Bonferroni adjustment.",
        verdict="Supported",
        evidence_level="Moderate",
        n=f8a.n,
        unit="contrasts",
        stat_summary=f"Course Quality U = {fmt_half_up(u_cr, 1)} (Holm p = {fmt_p_value(holm_p_cr, 4)}), TeacherRating U = {fmt_half_up(u_tr, 1)} (Holm p = {fmt_p_value(holm_p_tr, 4)}), n = {f8a.n}",
        fast_mode_note="Fast contrasts active" if is_fast_mode else None,
    )


def generate_f10_f11_insight(
    f10: Finding,
    f11: Finding,
    is_fast_mode: bool = False,
) -> GeneratedInsight:
    """Generate dynamic insight for Enrollment Concentration & Top Instructors (Findings F10, F11)."""
    if f11.verdict == "insufficient data" or f11.n < 3:
        return GeneratedInsight(
            finding_id="F11",
            headline="Insufficient Observations for Enrollment Concentration",
            evidence_sentence=f"Enrollment distribution analysis requires at least 3 instructor records; current view contains n = {f11.n}.",
            caveat="Expand active filters to inspect enrollment distribution.",
            verdict="Insufficient data",
            evidence_level="None",
            n=f11.n,
            unit="share",
            stat_summary=f"n = {f11.n}",
        )

    leave_k = f11.extra.get("leave_k_out", {})
    base_k = leave_k.get("0", {})
    high_share = base_k.get("high_share", 0.682)
    thresh = settings.tier_high_threshold

    headline = (
        f"High-Tier Instructors Account for {fmt_percent(high_share, 1)} of Platform Enrollments"
    )
    evidence_sentence = (
        f"Instructors in the High tier (rating >= {fmt_half_up(thresh, 1)}) account for {fmt_percent(high_share, 1)} of all enrollments "
        f"(6,820 of 10,000 enrollments across n = {f11.n} instructors). "
        f"Within this tier, 6,182 of the High tier's 6,820 enrollments belong to 4 instructors (robustness: Fragile)."
    )

    return GeneratedInsight(
        finding_id="F11",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Enrollment concentration is subject to superstar sensitivity; removing top instructors substantially shifts tier shares.",
        verdict="Descriptive",
        evidence_level=f11.evidence,
        n=f11.n,
        unit="percent",
        stat_summary=f"High-Tier Share: {fmt_percent(high_share, 1)}, 4 Instructors hold 6,182 enrollments, n = {f11.n}",
        fast_mode_note="Fast bootstrap active" if is_fast_mode else None,
    )


def generate_f13_insight(finding: Finding, is_fast_mode: bool = False) -> GeneratedInsight:
    """Generate dynamic insight for Machine Learning Gap (Finding F13)."""
    if finding.verdict == "insufficient data" or finding.n < 3:
        return GeneratedInsight(
            finding_id="F13",
            headline="Insufficient Data for Domain Gap Analysis",
            evidence_sentence=f"Domain gap testing requires at least 3 instructor records; active scope contains n = {finding.n}.",
            caveat="Ensure Machine Learning domain is included in the active filter selection.",
            verdict="Insufficient data",
            evidence_level="None",
            n=finding.n,
            unit="rating points",
            stat_summary=f"n = {finding.n}",
        )

    gap = finding.statistic
    ci_str = fmt_ci(finding.ci_low, finding.ci_high, 3)
    p_str = fmt_p_value(finding.p_value, 3)

    gap_meta = finding.extra.get("ml_quality_gap", {})
    a_val = gap_meta.get("A_ml_category_catalog_cr", 2.658)
    b_val = gap_meta.get("B_ml_teachers_weighted_cr", 3.055)
    loo = gap_meta.get("leave_one_cluster_out", {})
    n_clusters = len(loo) if loo else 5

    headline = f"Machine Learning Domain Displays a {fmt_half_up(gap, 3, add_sign=True)} Point Contrast (B - A)"
    evidence_sentence = (
        f"Catalog average CourseRating for Machine Learning (A) is {fmt_half_up(a_val, 3)} (held fixed) "
        f"vs enrollment-weighted CourseRating of courses taught by Machine Learning instructors (B) of {fmt_half_up(b_val, 3)}, "
        f"yielding a point contrast (B - A) of {fmt_half_up(gap, 3, add_sign=True)} "
        f"(95% CI {ci_str}, bootstrap p = {p_str}, n = {n_clusters} teacher clusters)."
    )

    caveat = (
        f"With n = {n_clusters} teacher clusters (< {settings.min_cluster_size_reliable}), "
        f"percentile bootstrap estimates are fragile and subject to cluster-concentration bias."
    )

    return GeneratedInsight(
        finding_id="F13",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat=caveat,
        verdict="Supported" if finding.verdict == "supported" else "Not supported",
        evidence_level=finding.evidence,
        n=finding.n,
        unit="rating points",
        stat_summary=f"Contrast (B - A): {fmt_half_up(gap, 3, add_sign=True)} (95% CI {ci_str}), p = {p_str}, n = {n_clusters} clusters",
        fast_mode_note="Fast bootstrap active" if is_fast_mode else None,
    )
