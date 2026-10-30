# Known Issues & Hardening Backlog

This document tracks known analytical, statistical, architectural, UI, and report anomalies identified during Step 1 Foundation hardening. These issues are recorded here as a backlog and will be addressed in subsequent steps as indicated.

---

## Step 2 — Statistical Rigor & Analytical Methodology (Resolved in Step 2)

- **[STEP-2-001] [RESOLVED] Pseudo-Replication & Invalid P-Values**: Resolved via `edupro.analytics.aggregation` and `inference.py`. All inferential statistics are conducted at the true observational units: teacher level ($n = 60$), course catalog level ($n = 60$), or cluster-robust standard errors clustering on `TeacherID`.
- **[STEP-2-002] [RESOLVED] Experience Impact Score Formulation**: Resolved via `experience_impact_v2` and Finding `F2`/`F3`. Reports two decoupled slopes: teacher rating slope (+0.120 points/yr, $p < 0.001$) and course quality slope (-0.002 points/yr, $p = 0.696$).
- **[STEP-2-003] [RESOLVED] Rating Consistency Index Dispersion**: Resolved via `rating_consistency_index_v2`. Employs instructor-weighted consistency ($1 - \text{CV}_{weighted}$) benchmarked against permutation test on shuffled course assignments.
- **[STEP-2-004] [RESOLVED] Enrollment Influence Confounding**: Resolved via `enrollment_influence_v2` and Finding `F10`/`F11`. Implements bootstrap CIs, leave-top-$k$-out sensitivity testing, and domain-controlled OLS regression with HC3 robust standard errors.
- **[STEP-2-005] [RESOLVED] Insignificant Category and Level Differences**: Resolved via Findings `F6` and `F7`. Kurskal-Wallis tests report epsilon-squared effect sizes and gate post-hoc pairwise testing with Holm adjustments only when omnibus test is significant.
- **[STEP-2-006] [RESOLVED] Low-Rated Tier Aggregation Pooling**: Resolved via `enrollment_influence_v2` and `teacher_table`. Isolates High ($\ge 4.0$), Mid ($3.0-3.9$), and Low ($< 3.0$) tiers with discrete per-tier mean enrollments.

---

## Step 4 Deprecation Backlog (Legacy Keys Scheduled for Removal in Step 4)

The following legacy keys and methods are marked deprecated in Step 2 and scheduled for decommissioning in Step 4:
- `KPIEngine.avg_teacher_rating()` -> migrate to `avg_teacher_rating_v2()`
- `KPIEngine.avg_course_rating()` -> migrate to `avg_course_rating_v2()`
- `KPIEngine.rating_consistency_index()` -> migrate to `rating_consistency_index_v2()`
- `KPIEngine.experience_impact_score()` -> migrate to `experience_impact_v2()`
- `KPIEngine.enrollment_influence_ratio()` -> migrate to `enrollment_influence_v2()`
- `KPIEngine.compute_all_kpis()` -> migrate to `compute_all_kpis_v2()`
- Legacy keys in `AnalyticsEngine.run_full_analysis()` dictionary:
  `"kpis"`, `"experience_vs_performance"`, `"course_quality"`, `"instructor_impact"`, `"expertise_performance"`, `"teacher_course_corr"`.

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
