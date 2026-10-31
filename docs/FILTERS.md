# EduPro Filter Architecture Specification

## 1. Architectural Overview

The EduPro Analytics platform uses a centralized, single-source-of-truth filtering pipeline implemented in `edupro.filters` and exposed to the user interface via `edupro.ui_filters`.

The architecture enforces five fundamental design principles:
1. **Mathematical Consistency**: All dashboard tabs, metric cards, charts, and statistical inference tables consume data from a single immutable `FilteredView` container.
2. **Defensible Boundary Coverage**: Numeric slider boundaries are derived dynamically from data using precision flooring (`floor_2d`) and ceiling (`ceil_2d`) functions, ensuring default filter views encompass 100% of underlying records.
3. **Immutability & Cacheability**: User filter selections are represented as a frozen, hashable `FilterState` dataclass, enabling sub-20ms instant cache re-renders via `@st.cache_data`.
4. **Strict Referential Re-Aggregation**: Filtering course or instructor cohorts dynamically recomputes weighted averages, portfolio sizes, and enrollment sums from the intersecting transaction pairs. Instructors or courses with zero enrollments in the selected scope are dropped from entity tables with explicit diagnostic notes.
5. **Data Privacy Guard**: The `Users` table remains strictly segregated for pipeline validation and cannot be joined, surfaced, filtered, or exported in any UI layer.

---

## 2. Core Components

### 2.1 `FilterBounds`
Data-derived boundary object calculated via `FilterBounds.from_dataset(dataset)`:
- `teacher_rating_min`: `1.05` (`floor_2d(min(TeacherRating))`)
- `teacher_rating_max`: `4.97` (`ceil_2d(max(TeacherRating))`)
- `course_rating_min`: `1.13` (`floor_2d(min(CourseRating))`)
- `course_rating_max`: `4.94` (`ceil_2d(max(CourseRating))`)
- `experience_min`: `1.0` (`floor_2d(min(YearsOfExperience))`)
- `experience_max`: `24.0` (`ceil_2d(max(YearsOfExperience))`)
- Categorical domain options: all unique sorted values of `Expertise` (12), `CourseCategory` (10), `CourseLevel` (3), and `Gender` (2).

### 2.2 `FilterState`
Frozen, hashable dataclass storing active user selections:
- `expertise`: `tuple[str, ...]` (empty tuple `()` = all selected / unconstrained)
- `course_category`: `tuple[str, ...]` (empty tuple `()` = all selected / unconstrained)
- `course_level`: `tuple[str, ...]` (empty tuple `()` = all selected / unconstrained)
- `instructor_gender`: `tuple[str, ...]` (empty tuple `()` = all selected / unconstrained)
- `teacher_rating_range`: `tuple[float, float]` (default: `(1.05, 4.97)`)
- `course_rating_range`: `tuple[float, float]` (default: `(1.13, 4.94)`)
- `experience_range`: `tuple[float, float]` (default: `(1.0, 24.0)`)

Methods:
- `default(dataset)`: returns default unconstrained state.
- `is_default(dataset)`: returns `True` when all attributes match default bounds.
- `to_dict()`, `from_dict(d)`: lossless serialization for Streamlit cache keys.
- `sanitize(dataset)`: clamps out-of-range bounds and removes invalid categorical tokens.

### 2.3 `FilteredView`
Frozen container returned by `apply_filters(dataset, state)`:
- `teachers`: filtered subset of instructor records.
- `courses`: filtered subset of course catalog records.
- `master`: filtered subset of instructor-course pairs formed by joining surviving teachers and courses with transactions.
- `transactions`: filtered subset of transactions matching the surviving pairs.
- `teacher_table`: re-aggregated instructor metrics (enrollments, weighted course ratings, consistency scores).
- `course_table`: re-aggregated course metrics (enrollments, weighted teacher ratings).
- `counts`: dictionary containing active counts (`n_teachers`, `n_teachers_active`, `n_courses`, `n_courses_active`, `n_pairs`, `n_enrollments`, `total_revenue`).
- `notes`: sequence of informational messages (e.g., dropped zero-enrollment entities).
- `is_empty`: boolean flag (`True` if `n_enrollments == 0` or `len(master) == 0`).

---

## 3. Mathematical Semantics & Interaction Rules

### 3.1 Cross-Table Intersection
Because the data model links instructors to courses only through `Transactions` (887 pairs across 60 teachers and 60 courses):
1. **Teacher-side filters** (`expertise`, `instructor_gender`, `teacher_rating_range`, `experience_range`) subset `teachers`.
2. **Course-side filters** (`course_category`, `course_level`, `course_rating_range`) subset `courses`.
3. The **active pair network** is formed by:
   $$\text{master}_{filtered} = \text{master}[\text{TeacherID} \in \text{teachers}_{filtered} \land \text{CourseID} \in \text{courses}_{filtered}]$$
4. **Re-Aggregation Invariant**:
   The sum of enrollments in `teacher_table` and `course_table` strictly equals the total filtered transactions:
   $$\sum \text{teacher\_table}[\text{total\_enrollments}] = \sum \text{course\_table}[\text{total\_enrollments}] = N_{enrollments}$$

### 3.2 Handling 0-Enrollment Entities
When course-side filters are applied (e.g. narrowing to a single course category), instructors who do not teach any courses in that category produce zero transactions in $\text{master}_{filtered}$.
- These instructors are dropped from `view.teacher_table` so that tables and charts do not display phantom zero-enrollment rows.
- The number of dropped instructors is recorded in `view.notes` and reflected in the active header caption ($N_{active} \text{ of } N_{total} \text{ instructors}$).

### 3.3 Small Sample Robustness & Trendline Guards
When filters result in small subsets ($n < 3$ or constant feature vectors):
- Pearson and Spearman correlations return `np.nan` with evidence marked as `"Insufficient data"`.
- OLS trendlines in Plotly charts are dynamically disabled (`trendline = None`) when $n < 3$ or when $x$ has only 1 unique value to prevent statsmodels collinearity crashes.
- Quantile binning (`pd.qcut`) gracefully falls back to available unique levels when quantiles collapse.

---

## 4. Scope Matrix: Empirical Filter to UI Mapping

The following matrix records the exact empirical behavior of each sidebar filter across the 6 UI functional areas. Every single cell represents an automated test case verified by `test_filter_scope_matrix_cell` in `tests/test_filters.py` (42 parameterized test cases).

### 4.1 UI Functional Scope Matrix (42 Cases)

| Filter Widget | Platform KPIs (`view.teacher_table` & `view.course_table`) | Tab 1: Rankings (`view.teacher_table`) | Tab 2: Experience (`view.teacher_table`) | Tab 3: Course Quality (`view.course_table`) | Tab 4: Domain Analysis (`view.master`) | Tab 5: Relationships (`view.master`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Expertise** | Changed (recomputed v2 tiles) | Changed (filtered instructors) | Changed (recomputed slopes/CIs) | Changed (portfolio re-weighted) | Changed (filtered domain pairs) | Changed (filtered pair scatter) |
| **Course Category** | Changed (recomputed v2 tiles) | Changed (drops 0-enr teachers) | Changed (recomputed weighted CR) | Changed (filtered course catalog) | Changed (filtered category pairs) | Changed (filtered pair scatter) |
| **Course Level** | Changed (recomputed v2 tiles) | Changed (re-aggregated pairs) | Changed (filtered course scope) | Changed (filtered level courses) | Changed (recomputed domain means) | Changed (filtered pair scatter) |
| **Instructor Gender** | Changed (recomputed v2 tiles) | Changed (filtered instructors) | Changed (gender-specific slopes) | Changed (portfolio re-weighted) | Changed (filtered domain pairs) | Changed (filtered pair scatter) |
| **Teacher Rating Range** | Changed (recomputed v2 tiles) | Changed (thresholded ranking) | Changed (range-bound scatter) | Changed (portfolio re-weighted) | Changed (filtered teacher rating pairs) | Changed (truncated x-axis pairs) |
| **Course Rating Range** | Changed (recomputed v2 tiles) | Changed (drops non-qualifying) | Changed (recomputed weighted CR) | Changed (range-bound courses) | Changed (filtered course rating pairs) | Changed (truncated y-axis pairs) |
| **Years of Experience** | Changed (recomputed v2 tiles) | Changed (filtered experience) | Changed (truncated scatter & bands)| Changed (portfolio re-weighted) | Changed (filtered exp domain pairs) | Changed (truncated exp pairs) |

### 4.2 Raw Catalog vs Re-Aggregated Feeding Objects (Orthogonal Invariance)
While all 42 UI targets mutate because the transaction graph interconnects instructors and courses, raw underlying catalogs exhibit deliberate orthogonal invariance:
- **`view.courses` (raw catalog) under teacher-only filters** (`expertise`, `instructor_gender`, `teacher_rating_range`, `experience_range`): **Unchanged** (`view.courses` retains all 60 courses). *Reason*: A filter on instructor attributes does not alter the course catalog definitions; only the transaction-weighted aggregates (`view.course_table`) and transaction pairs (`view.master`) are updated.
- **`view.teachers` (raw roster) under course-only filters** (`course_category`, `course_level`, `course_rating_range`): **Unchanged** (`view.teachers` retains all 60 teachers). *Reason*: A filter on course metadata does not alter instructor demographic records; only active teaching status (`view.teacher_table`), enrollment counts, and transaction pairs (`view.master`) are updated. Instructors with zero active course enrollments are dropped from `view.teacher_table` with an explicit diagnostic note.

> **Verification Evidence**: Automated regression test `test_filter_scope_matrix_cell` in `tests/test_filters.py` executes all 42 (filter $\times$ UI target) combinations in under 0.2s, verifying that every filter induces measurable, mathematically defensible changes in its primary feeding object.

---

## 5. UI Controls & State Management

- **One-Click Reset**: The `"🔄 Reset all filters"` button restores all multiselects and sliders to default dataset bounds and clears internal widget session state keys.
- **Dynamic Scope Display**: Caption in the sidebar dynamically displays `"N of M instructors, N of M courses in view"`.
- **Empty-State Guard**: If a user selects mutually exclusive filters that yield zero records, the app gracefully halts execution with a user-friendly guidance banner (`st.info`) and does not crash or display empty broken charts.
- **Fast Estimates Notice**: When in dashboard mode, an explicit badge alerts users that fast estimates mode ($N_{boot} = 300$) is active for sub-second responsiveness, pointing to the CLI runner for publication-grade precision ($N_{boot} = 2,000$).
