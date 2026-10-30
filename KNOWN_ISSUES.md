# Known Issues & Hardening Backlog

This document tracks known analytical, statistical, architectural, UI, and report anomalies identified during Step 1 Foundation hardening. These issues are recorded here as a backlog and will be addressed in subsequent steps as indicated.

---

## Step 2 — Statistical Rigor & Analytical Methodology

- **[STEP-2-001] Pseudo-Replication & Invalid P-Values**: The master dataset comprises $n = 887$ instructor-course pairs derived from only 60 distinct teachers and 60 distinct courses. Because courses and teachers are repeatedly observed across pairs, observations are not independent. Classical asymptotic standard errors and p-values based on $n = 887$ are invalid due to pseudo-replication. Clustered standard errors, mixed-effects models, or instructor/course-level aggregation are required.
- **[STEP-2-002] Experience Impact Score Formulation**: The current formula computes the arithmetic mean of two distinct correlations: $|r(\text{Experience}, \text{TeacherRating})|$ and $|r(\text{Experience}, \text{CourseRating})|$. Combining an instructor-level trait with a course-level trait across pseudo-replicated pairs obscures the fact that course ratings do not correlate with instructor experience ($r = -0.0131$).
- **[STEP-2-003] Rating Consistency Index Dispersion**: The Rating Consistency Index ($1 - \sigma / \mu$) exhibits minimal variance across instructors (ranging only from 0.51 to 0.77). This metric largely reflects the platform course mix rather than individual instructor reliability.
- **[STEP-2-004] Enrollment Influence Confounding**: The enrollment influence ratio (3.72) is heavily driven by 4 outlier instructors in Machine Learning and Cybersecurity. At the instructor level, Spearman correlation between teacher rating and total enrollment is negligible ($\rho \approx -0.01$).
- **[STEP-2-005] Insignificant Category and Level Differences**: Differences in course ratings across categories and levels are statistically insignificant (Kruskal-Wallis $p = 0.96$ for category, $p = 0.38$ for level).
- **[STEP-2-006] Low-Rated Tier Aggregation Pooling**: The "low-rated 64.9 average enrollment" calculation currently pools Mid (3.0–3.9) and Low (<3.0) tiers together rather than isolating the true Low tier.

---

## Step 3 — Filter Logic & Dashboard State Management

- **[STEP-3-001] Slider Bounds Truncation**: Sidebar sliders use `round()` on min/max rating values. Under the default view, this rounding hides 3 instructors with ratings 1.05 to 1.07 (slider starts at 1.1) and hides the highest-rated course at 4.94 (slider ends at 4.9).
- **[STEP-3-002] Disconnected Tab Filters**: Several tabs compute statistics on the global `master` or raw DataFrames rather than respecting active sidebar filters.
- **[STEP-3-003] KPI Tile Filter Inconsistency**: Two top-level KPI summary tiles ignore active cohort filter selections.

---

## Step 4 — Code Quality, UI Modernization & Deprecations

- **[STEP-4-001] Hard-Coded Insight Text & Dynamic Data Mismatches**: Several textual insights and callout narratives in `app.py` are hard-coded strings rather than dynamically derived from the underlying data.
- **[STEP-4-002] Static Header Counts**: Header metrics and title captions contain hard-coded counts rather than reading from active dataset dimensions.
- **[STEP-4-003] Excessive Heading & Callout Emojis**: Informal emojis in section titles and alert banners should be replaced with professional UI iconography.
- **[STEP-4-004] Streamlit `use_container_width` Deprecation**: Streamlit 1.66+ emits deprecation notices for `use_container_width=True` in `st.plotly_chart` (to be replaced with `width='stretch'`).

---

## Step 5 — Analytical Views & Feature Completeness

- **[STEP-5-001] Missing Age & Experience Distribution Views**: The dashboard lacks visual distributions for instructor age and years of experience.
- **[STEP-5-002] Missing Rating-Tier Course-Quality Contrast**: No direct comparison view contrasting course quality outcomes across instructor tiers.
- **[STEP-5-003] Missing Per-Instructor Reliability View**: Absence of a detailed breakdown for individual instructor consistency and course portfolio distributions.
- **[STEP-5-004] Missing Data Export Capability**: No UI facility to export filtered datasets, validation reports, or analytical summaries to CSV/Excel.

---

## Step 6 — Technical Reports & Governance Alignment

- **[STEP-6-001] Discrepant Figures in Markdown Reports**:
  - Reports cite 16.7% high-rated instructors; actual data shows 18.3% (11 / 60).
  - Reports state "nearly half below 3.0"; actual data shows 35.0% (21 / 60).
  - Reports cite "10 instructors below 2.0"; actual data contains 9 instructors with rating < 2.0.
  - Reports quote Spearman correlation of 0.598; actual Spearman correlation between experience and teacher rating is 0.6705 (0.598 is the Pearson $r$).
- **[STEP-6-002] Machine Learning Domain Conflation**: Reports conflate course category rating (2.66) with instructor expertise-based course rating (2.95).
- **[STEP-6-003] Marketing Domain Generalization**: Reports describe Marketing as having "moderate" instructor quality, whereas Marketing has exactly one instructor rated 4.29.
- **[STEP-6-004] Unverifiable Institutional Claims**: Reports assert alignment with "Toronto Government Educational Standards" and "Unified Mentor Framework" which are external and unverifiable.
- **[STEP-6-005] Report Date Discrepancy**: Reports are dated October 2025 despite transaction records extending to December 30, 2025.
