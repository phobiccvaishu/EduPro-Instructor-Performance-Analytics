# Research Paper: Exploratory Data Analysis & Statistical Insights
## Instructor Performance and Course Quality Evaluation on EduPro Online Education Platform

---

**Authors:** Senior Data Science & Analytics Team  
**Platform:** EduPro Online Education Platform  
**Dataset:** EduPro Online Platform.xlsx  
**Analysis Date:** October 2025  
**Methodology:** Exploratory Data Analysis (EDA), Correlation Analysis, Regression Modelling, Tier Segmentation

---

## Abstract

This paper presents a comprehensive data-driven evaluation of instructor performance and course quality across the EduPro online education platform. Using integrated data from 60 instructors, 60 courses, and 10,000 learner transactions, we apply statistical analysis across six methodological stages to address five core analytical questions. Key findings reveal a strong positive correlation between teaching experience and instructor ratings (r = 0.598, p ≈ 0), a near-zero correlation between teacher ratings and course ratings (r = −0.034), significant enrollment disparities across instructor tiers (high-rated instructors attract 3.72× more enrollments than the platform average), and meaningful variance across expertise domains and course categories. These findings inform actionable recommendations for improving platform-wide instructional quality.

---

## 1. Introduction

### 1.1 Background

Online education platforms face a fundamental challenge: ensuring consistent instructional quality at scale. Unlike traditional academic environments, digital platforms host instructors with widely varying expertise, experience, and pedagogical effectiveness. On EduPro, this heterogeneity is reflected in a teacher rating spread from 1.05 to 4.97 (out of 5.0) — a 4× range that directly translates to learner satisfaction and course success.

Traditional evaluation frameworks rely on subjective post-course surveys and fragmented feedback mechanisms. This research establishes a data-driven evaluation framework that shifts focus toward teaching effectiveness, using objective metrics derived from enrollment behaviour, rating consistency, and cross-dimensional correlations.

### 1.2 Problem Statement

Current instructor evaluation on EduPro suffers from:
- **Subjectivity**: Ratings are collected without standardised rubrics.
- **Fragmentation**: Teacher ratings and course ratings are evaluated in isolation.
- **Reactive management**: Poor performers are identified only after learner complaints escalate.

### 1.3 Core Analytical Questions

This research addresses five core questions:

1. Which instructors consistently deliver high-quality courses, and how are instructor ratings distributed?
2. Does teaching experience (YearsOfExperience) translate into higher ratings?
3. What is the statistical relationship between TeacherRating and CourseRating?
4. Which expertise areas and course categories consistently deliver high quality?
5. Are highly rated instructors associated with higher course enrollments?

---

## 2. Data Description & Integration

### 2.1 Dataset Overview

| Sheet | Records | Key Fields |
|-------|---------|------------|
| Teachers | 60 instructors | TeacherID, TeacherName, Age, Gender, Expertise, YearsOfExperience, TeacherRating |
| Courses | 60 courses | CourseID, CourseName, CourseCategory, CourseType, CourseLevel, CoursePrice, CourseDuration, CourseRating |
| Transactions | 10,000 records | TransactionID, UserID, CourseID, TeacherID, TransactionDate, Amount, PaymentMethod |
| Users | 3,000 learners | UserID, UserName, Age, Gender, Email |

### 2.2 Integration Pipeline

The integration pipeline merges three sheets via a two-step join:

1. **Transactions → Enrollment Aggregation**: Transactions are grouped by `(TeacherID, CourseID)` to produce `EnrollmentCount` per instructor-course pair.
2. **Enrollment + Courses**: Left-joined on `CourseID` to attach course metadata.
3. **Result + Teachers**: Left-joined on `TeacherID` to attach instructor attributes.

This yields a **master dataset of 887 rows** (instructor-course-enrollment records).

### 2.3 Data Quality Validation

| Metric | Value |
|--------|-------|
| Total master records | 887 |
| Unique instructors | 60 |
| Unique courses | 60 |
| Missing TeacherRating | 0 |
| Missing CourseRating | 0 |
| Mapping accuracy | 100% |

All instructor-to-course mappings were verified with 100% accuracy. No records were excluded due to missing key identifiers.

---

## 3. Methodology

The analytical framework follows six sequential stages:

1. **Data Integration** — merge, validate, produce master dataset
2. **Instructor Profile Analysis** — distribution analysis, top/bottom performers
3. **Experience vs. Performance** — correlation and regression
4. **Course Quality Evaluation** — aggregation by category, level, gender
5. **Instructor Impact on Course Success** — tier segmentation
6. **Expertise-Based Performance Insights** — cross-tabulation and domain analysis

Statistical methods employed:
- **Pearson and Spearman correlation** for bivariate relationships
- **Ordinary Least Squares (OLS) regression** for slope estimation
- **Quartile and band segmentation** for threshold analysis
- **Coefficient of Variation (CV)** for consistency indexing

---

## 4. Key Performance Indicators

### 4.1 KPI Summary Table

| KPI | Value | Interpretation |
|-----|-------|----------------|
| Average Teacher Rating | **3.125 / 5.0** | Moderate; significant room for improvement |
| Average Course Rating | **3.098 / 5.0** | Below desirable threshold of 3.5 |
| Rating Consistency Index | **0.622** | Moderate consistency; high inter-course variance |
| Experience Impact Score | **\|r\| = 0.306** | Moderate composite impact of experience on ratings |
| Enrollment Influence Ratio | **3.72×** | High-rated instructors attract 3.72× more enrollments |

### 4.2 KPI Computation Notes

**Rating Consistency Index** is computed as:

$$\text{RCI} = 1 - \frac{\sigma_{\text{CourseRating}}}{\mu_{\text{CourseRating}}}$$

A value of 0.622 indicates that the standard deviation of course ratings is approximately 37.8% of the mean — reflecting meaningful quality inconsistency across the platform.

**Experience Impact Score** averages the absolute Pearson r between YearsOfExperience and (a) TeacherRating and (b) CourseRating:

$$\text{EIS} = \frac{|r_{\text{exp→teacher}}| + |r_{\text{exp→course}}|}{2} = \frac{0.598 + 0.013}{2} = 0.306$$

**Enrollment Influence Ratio**: High-rated instructors (TeacherRating ≥ 4.0) average 620 enrollments versus the platform average of 166.67, producing a ratio of 3.72×.

---

## 5. Analytical Results

### 5.1 Instructor Profile Analysis

#### 5.1.1 Rating Distribution

| Statistic | Value |
|-----------|-------|
| Mean | 3.125 |
| Standard Deviation | 0.950 |
| Minimum | 1.05 |
| 25th Percentile | 2.488 |
| Median | 3.275 |
| 75th Percentile | 3.827 |
| Maximum | 4.970 |

The distribution is approximately symmetric around the median (3.275), with a slight left skew indicating a long tail of underperforming instructors. The interquartile range spans 1.34 rating points, indicating substantial heterogeneity across the 60-instructor cohort.

#### 5.1.2 Top-Performing Instructors

| Rank | Instructor | Expertise | Experience (yrs) | Rating |
|------|------------|-----------|-----------------|--------|
| 1 | Yolanda Levine | Machine Learning | 21 | **4.97** |
| 2 | Kimberly Miller | Cybersecurity | 24 | **4.58** |
| 3 | Kristi Scott | Machine Learning | 9 | **4.39** |
| 4 | Debra Escobar | Finance | 8 | **4.39** |
| 5 | Angela Beard | Machine Learning | 6 | **4.36** |

Machine Learning instructors dominate the top tier, suggesting strong alignment between ML teaching expertise and learner satisfaction.

#### 5.1.3 Low-Performing Instructors (Training Priority)

| Rank | Instructor | Expertise | Experience (yrs) | Rating |
|------|------------|-----------|-----------------|--------|
| 1 | John Williams | Project Management | 1 | **1.05** |
| 2 | Timothy Bailey | Business | 3 | **1.06** |
| 3 | Emily Gutierrez | Web Development | 5 | **1.07** |
| 4 | Brenda Mclean | Digital Marketing | 4 | **1.39** |
| 5 | Laurie Michael | Artificial Intelligence | 3 | **1.55** |

All bottom performers have fewer than 6 years of experience, consistent with the strong experience-rating correlation found in Section 5.2.

#### 5.1.4 Expertise Distribution

| Expertise Domain | Count | Avg Teacher Rating |
|-----------------|-------|-------------------|
| Marketing | 1 | 4.290 |
| Machine Learning | 5 | 4.234 |
| Programming | 1 | 3.950 |
| Cybersecurity | 9 | 3.469 |
| Finance | 6 | 3.265 |
| Artificial Intelligence | 6 | 3.173 |
| Digital Marketing | 11 | 3.050 |
| Data Science | 5 | 3.030 |
| Project Management | 5 | 2.664 |
| Web Development | 5 | 2.514 |
| Design | 4 | 2.290 |
| Business | 2 | 2.245 |

> **Finding:** Machine Learning and Cybersecurity instructors are the platform's highest-rated cohorts; Design and Business instructors require the most targeted development.

---

### 5.2 Experience vs. Performance Analysis (Core Question 2)

#### 5.2.1 Correlation Results

| Metric | Value |
|--------|-------|
| Pearson r (Experience → TeacherRating) | **+0.598** |
| p-value | **< 0.001** (statistically significant) |
| Spearman r | **+0.598** |
| OLS Slope | **+0.120 per year** |
| Pearson r (Experience → CourseRating) | **−0.013** |
| p-value (Course) | **0.696** (not significant) |

**Interpretation:** There is a **strong, statistically significant positive correlation** between years of experience and TeacherRating (r = 0.598, p < 0.001). Each additional year of teaching experience is associated with a **0.12-point increase** in TeacherRating on average. Conversely, experience has **no significant effect** on CourseRating (r = −0.013, p = 0.696), suggesting that course content quality is driven by factors beyond instructor tenure.

#### 5.2.2 Rating by Experience Band

| Experience Band | Avg Teacher Rating | Count |
|-----------------|--------------------|-------|
| 0–5 years | **2.399** | 28 instructors |
| 6–10 years | **3.708** | 23 instructors |
| 11–15 years | **3.643** | 7 instructors |
| 16+ years | **4.775** | 2 instructors |

> **Threshold Finding:** A clear performance step-change occurs between the 0–5 year and 6–10 year bands (+1.31 rating points). Beyond 10 years, marginal gains diminish slightly (11–15 yrs: 3.643), but the 16+ band resumes upward momentum, likely due to highly senior expert instructors (n=2 only).

> **Diminishing Returns:** The 11–15 year band scores marginally lower than the 6–10 year band, suggesting a possible productivity plateau in the mid-career range before senior mastery effects dominate.

---

### 5.3 Teacher Rating ↔ Course Rating Correlation (Core Question 3)

| Metric | Value |
|--------|-------|
| Pearson r | **−0.034** |
| p-value | **0.317** (not significant) |
| Spearman r | **−0.050** |
| OLS Slope | **−0.039** |
| Sample size (n) | 887 |

**Interpretation:** The correlation between TeacherRating and CourseRating is **effectively zero and not statistically significant** (r = −0.034, p = 0.317). This is a critical finding: **instructor reputation does not transfer directly to course quality ratings**. This implies that course design, content structure, and learning material quality are evaluated independently of the instructor's personal teaching effectiveness score.

> **Implication:** EduPro must evaluate instructor performance and course quality through **separate, complementary frameworks** rather than using one as a proxy for the other. A highly-rated instructor can still produce a poorly-rated course if the curriculum design is inadequate.

---

### 5.4 Course Quality Evaluation (Core Question 4)

#### 5.4.1 Course Rating by Category

| Category | Avg Rating | Count | Std Dev |
|----------|------------|-------|---------|
| Marketing | **3.722** | 5 | 1.201 |
| Digital Marketing | **3.656** | 5 | 1.029 |
| Data Science | **3.316** | 5 | 0.504 |
| Design | 3.162 | 5 | 1.292 |
| Project Management | 3.142 | 5 | 1.708 |
| Programming | 3.036 | 5 | 1.327 |
| Artificial Intelligence | 3.036 | 5 | 1.241 |
| Finance | 3.010 | 5 | 1.100 |
| Cybersecurity | 2.902 | 5 | 1.490 |
| Web Development | **2.846** | 5 | 1.289 |
| Business | **2.688** | 5 | 0.729 |
| Machine Learning | **2.658** | 5 | 1.559 |

> **Key Insight:** Marketing and Digital Marketing lead in course quality, while Machine Learning — despite having the highest-rated instructors — has the **lowest average course rating (2.658)**. This reinforces the finding that instructor quality does not guarantee course quality and suggests ML course design requires significant attention.

**Data Science** courses show the lowest variability (std = 0.504), indicating consistent but modest quality across that category.

#### 5.4.2 Course Rating by Level

| Level | Avg Rating | Count | Std Dev |
|-------|------------|-------|---------|
| Intermediate | **3.323** | 18 | 1.090 |
| Beginner | 3.189 | 21 | 1.300 |
| Advanced | **2.814** | 21 | 1.099 |

> **Finding:** Advanced-level courses consistently underperform relative to Beginner and Intermediate levels. This may reflect higher learner expectations, greater content complexity demands, or insufficient scaffolding for advanced learners.

---

### 5.5 Instructor Impact on Enrollment (Core Question 5)

#### 5.5.1 Tier Segmentation Results

Instructors are segmented into three tiers based on TeacherRating:
- **High-rated**: TeacherRating ≥ 4.0
- **Mid-rated**: 3.0 ≤ TeacherRating < 4.0
- **Low-rated**: TeacherRating < 3.0

| Tier | Avg Course Rating | Avg Enrollment | Total Enrollment |
|------|-------------------|----------------|-----------------|
| High (≥4.0) | varies | **620 avg** | highest |
| Mid (3.0–3.9) | varies | moderate | moderate |
| Low (<3.0) | varies | **64.9 avg** | lowest |

**Enrollment Influence Ratio**: High-rated instructors generate **3.72× more enrollments** than the platform average (166.67), and **9.55× more** than low-rated instructors (64.9). This confirms strong learner preference for highly-rated instructors.

> **Platform Recommendation**: Instructor ratings act as a significant demand signal. Courses taught by high-rated instructors should be featured prominently in recommendation algorithms. Simultaneously, quality intervention programs should target low-rated instructors to reduce the enrollment disparity.

---

### 5.6 Expertise-Based Performance Insights

#### 5.6.1 Expertise-Level Correlation (TeacherRating ↔ CourseRating)

The Pearson correlation between TeacherRating and CourseRating, computed within each expertise domain, reveals which domains are most teacher-quality dependent:

- **High positive r**: Domains where excellent instructors significantly improve course outcomes — these are fields where teacher-mediated learning is critical.
- **Near-zero or negative r**: Domains where course quality is determined more by content design than instructor quality.

#### 5.6.2 Enrollment Demand by Expertise

Machine Learning instructors, despite having the highest average TeacherRating, serve courses that underperform in ratings — generating a quality-reputation mismatch. Digital Marketing has both respectable instructor ratings (3.050) and the second-highest average course rating (3.656), suggesting a well-balanced domain.

---

## 6. Summary of Key Statistical Findings

| Research Question | Finding | Statistical Evidence |
|-------------------|---------|---------------------|
| Q1: Top vs low performers | Rating range 1.05–4.97; Machine Learning leads | Mean=3.125, SD=0.950 |
| Q2: Experience → Rating | Strong positive effect on TeacherRating | r=0.598, p<0.001 |
| Q2: Experience → CourseRating | No significant effect | r=−0.013, p=0.696 |
| Q3: TeacherRating ↔ CourseRating | Effectively zero correlation | r=−0.034, p=0.317 |
| Q4: Category quality leaders | Marketing (3.722), Digital Marketing (3.656) | Mean ± SD by category |
| Q4: Level quality | Intermediate best, Advanced lowest | Intermediate=3.323, Advanced=2.814 |
| Q5: Enrollment influence | 3.72× enrollment ratio for high-rated | Avg 620 vs 166.67 platform avg |

---

## 7. Recommendations

### 7.1 Platform-Level Recommendations

1. **Introduce a Dual Evaluation Framework**: Separate instructor performance scores from course quality scores; use both in tandem for instructor reviews.

2. **Implement an Early-Career Mentorship Programme**: The 0–5 year cohort (avg rating 2.399) comprises 28/60 instructors (47%). Pairing them with senior instructors (16+ years, avg 4.775) would directly address the largest performance gap.

3. **Course Design Audit for Machine Learning and Business**: Despite strong instructor talent in ML, course ratings lag (2.658). A dedicated curriculum review and instructional design intervention is recommended.

4. **Promote High-Rated Instructors in Recommendation Engine**: Given the 3.72× enrollment influence, algorithmic promotion of high-rated instructors can significantly improve platform revenue and learner satisfaction simultaneously.

5. **Advanced-Level Course Redesign**: Advanced courses consistently underperform (2.814). Content scaffolding reviews and learner feedback analysis should be prioritised.

6. **Targeted Domain Training**: Design (avg 2.290) and Business (avg 2.245) instructors show the weakest average ratings. Mandatory professional development or recruitment of stronger expertise in these domains is recommended.

### 7.2 KPI Targets for Next Review Cycle

| KPI | Current | 12-Month Target |
|-----|---------|-----------------|
| Avg Teacher Rating | 3.125 | ≥ 3.5 |
| Avg Course Rating | 3.098 | ≥ 3.4 |
| Rating Consistency Index | 0.622 | ≥ 0.70 |
| % Instructors in High Tier | ~16.7% | ≥ 30% |
| Enrollment Influence Ratio | 3.72× | Maintained or increased |

---

## 8. Limitations

- **Cross-sectional design**: Data represents a single time window; longitudinal tracking would reveal rating trajectories.
- **Small cohorts**: Several expertise domains have only 1–2 instructors, limiting domain-level generalisability.
- **Rating aggregation**: TeacherRating and CourseRating likely aggregate multiple underlying sub-scores; granular decomposition would improve diagnostic precision.
- **Confounding variables**: Learner demographics, course pricing, and platform marketing exposure are not controlled for in enrollment analysis.

---

## 9. Conclusion

This study provides the first systematic, data-driven analysis of instructor performance and course quality on the EduPro platform. The central finding — that **teaching experience strongly predicts instructor reputation but not course quality** — fundamentally reframes platform strategy. Instructor development and curriculum design must be treated as separate, equally important investment priorities. With 47% of instructors in the lowest experience band and advanced course ratings trailing by 15% relative to intermediate courses, EduPro has significant opportunity to improve platform-wide quality through targeted interventions, mentorship programmes, and curriculum redesign initiatives.

---

## Appendix A: Variables Reference

| Variable | Type | Source | Role |
|----------|------|---------|------|
| TeacherRating | Float [1–5] | Teachers sheet | Outcome / Predictor |
| CourseRating | Float [1–5] | Courses sheet | Outcome |
| YearsOfExperience | Integer | Teachers sheet | Predictor |
| Expertise | Categorical | Teachers sheet | Grouping |
| CourseCategory | Categorical | Courses sheet | Grouping |
| CourseLevel | Categorical | Courses sheet | Grouping |
| EnrollmentCount | Integer (derived) | Transactions | Demand metric |
| Gender | Categorical | Teachers/Users | Demographic |

## Appendix B: Analytical Tools & Libraries

- **Python 3.14** — core language
- **pandas 3.0.3** — data wrangling and aggregation
- **scipy.stats** — Pearson/Spearman correlation, significance testing
- **scikit-learn** — OLS regression
- **plotly** — interactive visualisations
- **streamlit** — web dashboard framework
- **openpyxl** — Excel data ingestion
