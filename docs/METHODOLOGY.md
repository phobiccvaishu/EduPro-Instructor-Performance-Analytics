# Statistical Methodology & Inferential Framework

This document outlines the formal statistical formulations, effect size definitions, hypothesis test corrections, and resampling procedures implemented across the EduPro Analytics platform.

---

## 1. Non-Parametric Group Comparisons

### 1.1 Kruskal-Wallis $H$ Test and Epsilon-Squared ($\epsilon^2$)
For multi-group independent comparisons (e.g., F6 across 12 instructor expertise categories, F7 across 3 course levels), the Kruskal-Wallis $H$ statistic tests the null hypothesis that all group medians originate from identical continuous distributions:

$$H = \frac{12}{N(N+1)} \sum_{i=1}^k \frac{R_i^2}{n_i} - 3(N+1)$$

where:
- $k$ is the number of comparison groups.
- $n_i$ is the number of observations in group $i$.
- $N = \sum_{i=1}^k n_i$ is the total number of observations.
- $R_i$ is the rank sum for group $i$.

The non-parametric effect size is evaluated using the epsilon-squared ($\epsilon^2$) metric (Tomczak & Tomczak, 2014), lower-bounded at zero:

$$\epsilon^2 = \max\left(0, \frac{H - k + 1}{N - k}\right)$$

#### Effect Size Thresholds for $\epsilon^2$:
- $\epsilon^2 < 0.01$: **Negligible**
- $0.01 \le \epsilon^2 < 0.08$: **Small**
- $0.08 \le \epsilon^2 < 0.26$: **Moderate**
- $\epsilon^2 \ge 0.26$: **Large**

---

### 1.2 Mann-Whitney $U$ Test and Rank-Biserial Correlation
For two-group independent comparisons (e.g., F4a 0–5 vs 6+ years experience, F8a/F8b gender comparisons), the Mann-Whitney $U$ statistic evaluates rank dominance:

$$U_1 = R_1 - \frac{n_1(n_1 + 1)}{2}$$

The standardized effect size is computed as the rank-biserial correlation $r_{rb}$:

$$r_{rb} = 1 - \frac{2 U}{n_1 n_2}$$

Interpretation follows standard correlation magnitude tiers:
- $|r_{rb}| < 0.10$: **Negligible**
- $0.10 \le |r_{rb}| < 0.30$: **Small**
- $0.30 \le |r_{rb}| < 0.50$: **Moderate**
- $|r_{rb}| \ge 0.50$: **Large**

---

## 2. Familywise Error Rate Control (Holm-Bonferroni)
When testing related sub-hypotheses simultaneously (e.g., gender differences across TeacherRating, CourseQuality, and Level cells in F8a, F8b, F8c), raw $p$-values are adjusted using the Holm-Bonferroni step-down procedure to guard against Type I error inflation:

$$p_{(i)}^{adj} = \min\left(1, \max_{j \le i} \left[ (m - j + 1) \cdot p_{(j)} \right]\right)$$

where $m$ is the number of hypothesis tests within the family and $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$ are ordered raw $p$-values.

---

## 3. Resampling and Sensitivity Procedures

### 3.1 Teacher-Cluster Percentile Bootstrap
To evaluate the curriculum quality contrast (F13) between catalog category CourseRating ($A$, held fixed) and enrollment-weighted CourseRating ($B$) of courses taught by specialized instructors:
1. Resampling unit: Entire instructor clusters (all courses taught by instructor $j$) are resampled with replacement to preserve intra-instructor error correlation.
2. The contrast statistic $B^* - A$ is computed across $B = 2,000$ bootstrap replicates.
3. Two-sided empirical $p$-value and $95\%$ percentile confidence interval $[\hat{\theta}_{0.025}^*, \hat{\theta}_{0.975}^*]$ are derived from the identical bootstrap distribution.
4. If the number of teacher clusters is small ($n_{cluster} < 8$), point estimates are subject to ratio-estimation bias, and evidence labels are capped at `"Weak"` with an explicit caveat note.

### 3.2 Superstar Concentration & Leave-Top-$k$-Out Sensitivity
To test whether platform metrics are driven by superstar outlier concentration (KPI 5, F11, F8b):
- Instructors are sorted descending by total platform enrollments.
- Top $k$ instructors ($k \in \{0, 1, 2, 3, 4\}$) are iteratively removed from the population.
- Ratios, group contrasts, and hypothesis tests are re-evaluated on the remaining cohort.
- A metric is classified as **Robust** if its effect size retains $\ge 75\%$ of baseline magnitude and its confidence interval continues to exclude the null parameter; otherwise, it is classified as **Fragile**.

---

## 4. Verdict Decision Rules
Every inferential finding (F1–F13) evaluates to one of the canonical verdicts governed by `AnalysisConfig.alpha` ($\alpha = 0.05$):
- **Supported**: $p < \alpha$, sample size satisfies $n \ge n_{min}$, and 95% confidence interval excludes zero / null effect.
- **Not Supported**: $p \ge \alpha$ or 95% confidence interval crosses zero.
- **Inconclusive**: $p < \alpha$ but confidence interval is uninformative or bounds span practically zero.
- **Descriptive**: Observational summaries without formal null-hypothesis testing.
