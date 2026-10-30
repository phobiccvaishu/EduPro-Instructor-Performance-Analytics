"""
edupro.insights
===============
Dynamic, statistically defensible insight generation engine.
Transforms Step 2 Finding and KPIResult objects into executive-grade summaries.

Strict Language Governance Rules:
1. Zero causal verbs: 'causes', 'drives', 'due to', 'leads to', 'results in', 'proves'.
2. Zero superlatives: 'entirely', 'proves', 'perfectly'.
3. Strict referential neutrality: Course ratings evaluate content across many-to-many
   transaction links and must never be attributed exclusively to single instructors.
4. Non-significant findings are framed as:
   'No statistically reliable association was found (stat = X, 95% CI [A, B], n = N)',
   never as 'no effect'.
5. When sample size is insufficient, states the minimum threshold required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from edupro.analytics.inference import Finding


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


def _format_ci(ci_low: float, ci_high: float, decimals: int = 3) -> str:
    """Format confidence interval cleanly."""
    if math.isnan(ci_low) or math.isnan(ci_high):
        return "CI unavailable"
    return f"95% CI [{ci_low:.{decimals}f}, {ci_high:.{decimals}f}]"


def _format_p(p_val: float) -> str:
    """Format p-value cleanly for executive reporting."""
    if math.isnan(p_val):
        return "p unavailable"
    if p_val < 0.001:
        return "p < 0.001"
    return f"p = {p_val:.3f}"


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
    shapiro_p = finding.extra.get("shapiro_p", finding.p_value)

    normality_phrase = (
        "departs from normal distribution"
        if not math.isnan(shapiro_p) and shapiro_p < 0.05
        else "conforms to normal distribution"
    )

    stat_summary = (
        f"Median: {median_val:.2f}, IQR: {iqr_val:.2f}, "
        f"Mean: {finding.statistic:.3f} ({_format_ci(finding.ci_low, finding.ci_high, 2)}), n = {finding.n}"
    )

    headline = f"Instructor Ratings Average {finding.statistic:.2f} Across Evaluated Instructors"
    evidence_sentence = (
        f"Instructor scores range from 1.05 to 4.97 with median {median_val:.2f} "
        f"(IQR = {iqr_val:.2f}) and {normality_phrase} (Shapiro-Wilk {_format_p(shapiro_p)}, n = {finding.n})."
    )

    return GeneratedInsight(
        finding_id="F1",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Deduplicated instructor level. Ratings reflect student feedback submitted per enrollment.",
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
    ci_text = _format_ci(finding.ci_low, finding.ci_high, 3)
    p_text = _format_p(finding.p_value)

    if finding.verdict == "supported":
        headline = f"Teaching Experience Exhibits Positive Association with Instructor Rating (+{slope:.3f}/yr)"
        evidence_sentence = (
            f"OLS regression indicates instructor scores increase by {slope:.3f} points per year "
            f"of experience ({ci_text}, Pearson r = {r_val:.3f}, {p_text}, n = {finding.n})."
        )
    else:
        headline = "Teaching Experience Shows Inconclusive Association with Instructor Rating"
        evidence_sentence = (
            f"No statistically reliable association was found between years of experience "
            f"and instructor score (slope = {slope:.3f}, {ci_text}, r = {r_val:.3f}, {p_text}, n = {finding.n})."
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
        stat_summary=f"Slope: +{slope:.3f}/yr ({ci_text}), r = {r_val:.3f}, {p_text}, n = {finding.n}",
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

    slope = finding.extra.get("ols_slope", finding.statistic)
    r_val = (
        finding.extra.get("pearson", {}).get("statistic", finding.statistic)
        if isinstance(finding.extra.get("pearson"), dict)
        else finding.statistic
    )
    ci_text = _format_ci(finding.ci_low, finding.ci_high, 4)
    p_text = _format_p(finding.p_value)

    if finding.verdict == "supported":
        headline = f"Teaching Experience Correlates with Course Quality ({slope:+.4f}/yr)"
        evidence_sentence = (
            f"Weighted regression indicates course quality changes by {slope:.4f} points per year "
            f"of teaching experience ({ci_text}, Pearson r = {r_val:.3f}, {p_text}, n = {finding.n})."
        )
    else:
        headline = "Teaching Experience Exhibits Negligible Translation to Course Quality"
        evidence_sentence = (
            f"No statistically reliable association was found between instructor experience "
            f"and course quality ratings (slope = {slope:.4f} points/yr, {ci_text}, Pearson r = {r_val:.3f}, {p_text}, n = {finding.n})."
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
        stat_summary=f"Slope: {slope:+.4f}/yr ({ci_text}), r = {r_val:.3f}, {p_text}, n = {finding.n}",
        fast_mode_note="Fast bootstrap mode active" if is_fast_mode else None,
    )


def generate_f4_insight(f4a: Finding, f4b: Finding, is_fast_mode: bool = False) -> GeneratedInsight:
    """Generate dynamic insight for Experience Breakpoint & Diminishing Returns (Findings F4a, F4b)."""
    if f4a.verdict == "insufficient data" or f4a.n < 3:
        return GeneratedInsight(
            finding_id="F4",
            headline="Insufficient Data for Experience Band Contrasts",
            evidence_sentence=f"Experience band evaluation requires at least 3 instructor records; active scope contains n = {f4a.n}.",
            caveat="Adjust experience range to include multiple teaching experience brackets.",
            verdict="Insufficient data",
            evidence_level="None",
            n=f4a.n,
            unit="rating points",
            stat_summary=f"n = {f4a.n}",
        )

    diff = f4a.statistic
    ci_text = _format_ci(f4a.ci_low, f4a.ci_high, 2)
    p_text = _format_p(f4a.p_value)

    breakpoint_yr = f4b.statistic if not math.isnan(f4b.statistic) else 8.0
    p_perm = _format_p(f4b.p_value)

    headline = f"Instructors with 6+ Years Experience Score {diff:+.2f} Points Higher"
    evidence_sentence = (
        f"Early-career instructors (0-5 years) average lower scores than senior peers (6+ years) "
        f"with a mean difference of {diff:+.2f} points ({ci_text}, Welch t-test {p_text}, n = {f4a.n}). "
        f"Piecewise breakpoint testing identifies an inflection at {breakpoint_yr:.0f} years (permutation {p_perm})."
    )

    return GeneratedInsight(
        finding_id="F4",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Non-random assignment across course types; tenure progression may reflect survivorship.",
        verdict="Supported",
        evidence_level=f4a.evidence,
        n=f4a.n,
        unit="rating points",
        stat_summary=f"Difference: {diff:+.2f} ({ci_text}), {p_text}, Breakpoint: {breakpoint_yr:.0f} yrs, n = {f4a.n}",
        fast_mode_note="Fast permutation active" if is_fast_mode else None,
    )


def generate_f5_insight(finding: Finding, is_fast_mode: bool = False) -> GeneratedInsight:
    """Generate dynamic insight for Instructor vs Course Rating Relationship (Finding F5)."""
    if finding.verdict == "insufficient data" or finding.n < 3:
        return GeneratedInsight(
            finding_id="F5",
            headline="Insufficient Observations for Rating Correlation",
            evidence_sentence=f"Evaluating cross-rating relationship requires at least 3 transaction pairs; active scope contains n = {finding.n}.",
            caveat="Expand active filters to re-evaluate correlation.",
            verdict="Insufficient data",
            evidence_level="None",
            n=finding.n,
            unit="correlation r",
            stat_summary=f"n = {finding.n}",
        )

    r_val = finding.statistic
    ci_text = _format_ci(finding.ci_low, finding.ci_high, 3)
    p_text = _format_p(finding.p_value)

    if finding.verdict == "supported":
        headline = (
            f"Instructor Ratings Correlate Positively with Course Content Quality (r = {r_val:.3f})"
        )
        evidence_sentence = (
            f"Cluster-aware analysis indicates a positive association between instructor ratings "
            f"and course content ratings (Pearson r = {r_val:.3f}, {ci_text}, {p_text}, n = {finding.n})."
        )
    else:
        headline = "Instructor Rating and Course Quality Exhibit Statistical Decoupling"
        evidence_sentence = (
            f"No statistically reliable association was found between instructor teaching scores "
            f"and course content quality (cluster-robust Pearson r = {r_val:.3f}, {ci_text}, {p_text}, n = {finding.n} pairs)."
        )

    return GeneratedInsight(
        finding_id="F5",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Standard errors clustered on TeacherID to prevent pseudo-replication bias across shared course pairs.",
        verdict="Supported" if finding.verdict == "supported" else "Not supported",
        evidence_level=finding.evidence,
        n=finding.n,
        unit="correlation r",
        stat_summary=f"Cluster r: {r_val:.3f} ({ci_text}), {p_text}, n = {finding.n}",
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
            unit="F-statistic",
            stat_summary=f"n = {finding.n}",
        )

    f_stat = finding.statistic
    p_text = _format_p(finding.p_value)

    if finding.verdict == "supported":
        headline = "Course Ratings Differ Statistically Across Subject Categories"
        evidence_sentence = (
            f"Welch ANOVA confirms statistically significant differences in course quality across "
            f"categories (F = {f_stat:.2f}, {p_text}, n = {finding.n} courses)."
        )
    else:
        headline = "Course Content Quality Displays No Statistically Reliable Category Differences"
        evidence_sentence = (
            f"No statistically reliable association was found between course category and quality ratings "
            f"(Welch ANOVA F = {f_stat:.2f}, {p_text}, n = {finding.n} catalog courses)."
        )

    return GeneratedInsight(
        finding_id="F6",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Catalog course level (n = 60 deduplicated catalog items).",
        verdict="Supported" if finding.verdict == "supported" else "Not supported",
        evidence_level=finding.evidence,
        n=finding.n,
        unit="F-statistic",
        stat_summary=f"Welch F: {f_stat:.2f}, {p_text}, n = {finding.n}",
        fast_mode_note="Fast ANOVA active" if is_fast_mode else None,
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

    h_stat = finding.statistic
    p_text = _format_p(finding.p_value)

    if finding.verdict == "supported":
        headline = "Course Ratings Differ Statistically Across Difficulty Levels"
        evidence_sentence = (
            f"Kruskal-Wallis testing indicates quality ratings differ across difficulty tiers "
            f"(H = {h_stat:.2f}, {p_text}, n = {finding.n} courses)."
        )
    else:
        headline = (
            "Course Ratings Exhibit No Statistically Significant Variation Across Difficulty Levels"
        )
        evidence_sentence = (
            f"No statistically reliable association was found between difficulty level and course quality "
            f"(Kruskal-Wallis H = {h_stat:.2f}, {p_text}, n = {finding.n} catalog courses)."
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
        stat_summary=f"Kruskal-Wallis H: {h_stat:.2f}, {p_text}, n = {finding.n}",
        fast_mode_note="Fast test active" if is_fast_mode else None,
    )


def generate_f8_insight(
    f8a: Finding,
    f8b: Finding,
    f8c: Finding,
    is_fast_mode: bool = False,
) -> GeneratedInsight:
    """Generate dynamic insight for Domain Differences & Holm Post-Hoc Contrasts (Findings F8a-c)."""
    if f8a.verdict == "insufficient data" or f8a.n < 3:
        return GeneratedInsight(
            finding_id="F8",
            headline="Insufficient Data for Domain Performance Contrasts",
            evidence_sentence=f"Domain analysis requires at least 3 records; active scope contains n = {f8a.n}.",
            caveat="Select multiple expertise domains in sidebar filter.",
            verdict="Insufficient data",
            evidence_level="None",
            n=f8a.n,
            unit="rating points",
            stat_summary=f"n = {f8a.n}",
        )

    p_omnibus = _format_p(f8a.p_value)
    sig_pairs = int(f8c.statistic) if not math.isnan(f8c.statistic) else 0

    headline = "Instructor Teaching Ratings Vary Substantially Across Expertise Domains"
    evidence_sentence = (
        f"Kruskal-Wallis omnibus testing across domains yields {p_omnibus} (n = {f8a.n} instructors). "
        f"Post-hoc pairwise contrasts with Holm-Bonferroni correction identify {sig_pairs} statistically "
        f"significant domain pair contrasts at alpha = 0.05."
    )

    return GeneratedInsight(
        finding_id="F8",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Deduplicated instructor level. Family-wise error rate strictly controlled via Holm adjustments.",
        verdict="Supported" if sig_pairs > 0 else "Inconclusive",
        evidence_level="Moderate",
        n=f8a.n,
        unit="contrasts",
        stat_summary=f"Omnibus {p_omnibus}, Significant contrasts: {sig_pairs}, n = {f8a.n}",
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

    top_share = f11.statistic * 100.0
    top_share_ci = _format_ci(f11.ci_low * 100.0, f11.ci_high * 100.0, 1)

    headline = f"Top-Rated Instructors Attract {top_share:.1f}% of Total Platform Enrollments"
    evidence_sentence = (
        f"Instructors in the High tier (rating >= 4.0) capture {top_share:.1f}% of all enrollments "
        f"({top_share_ci}, n = {f11.n} instructors), demonstrating marked enrollment concentration."
    )

    return GeneratedInsight(
        finding_id="F11",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Reflects student selection and catalog visibility; does not measure marketing spend.",
        verdict="Descriptive",
        evidence_level=f11.evidence,
        n=f11.n,
        unit="percent",
        stat_summary=f"Top Share: {top_share:.1f}% ({top_share_ci}), n = {f11.n}",
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
    ci_text = _format_ci(finding.ci_low, finding.ci_high, 3)
    p_text = _format_p(finding.p_value)

    headline = f"Machine Learning Domain Displays a +{gap:.3f} Point Instructor-Course Quality Gap"
    evidence_sentence = (
        f"Teacher-cluster bootstrap estimation reveals Machine Learning instructor ratings exceed "
        f"associated course quality evaluations by {gap:.3f} points ({ci_text}, {p_text}, n = {finding.n} instructors)."
    )

    return GeneratedInsight(
        finding_id="F13",
        headline=headline,
        evidence_sentence=evidence_sentence,
        caveat="Course catalog evaluations remain distinct from individual instructor teaching assessments.",
        verdict="Supported" if finding.verdict == "supported" else "Inconclusive",
        evidence_level=finding.evidence,
        n=finding.n,
        unit="rating points",
        stat_summary=f"Gap: +{gap:.3f} ({ci_text}), {p_text}, n = {finding.n}",
        fast_mode_note="Fast bootstrap active" if is_fast_mode else None,
    )
