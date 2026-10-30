# EduPro Analytics Platform — UI & Design System Guide

This document defines the design tokens, component architecture, chart standards, dynamic insight generation system, and governance rules for the EduPro Analytics Platform user interface.

---

## 1. Design System & Tokens

The EduPro dashboard follows an **Executive BI** design philosophy: restrained, accessible, data-dense, and completely free of decorative distractions, animations, and informal emojis.

### 1.1 Color Palette
The interface uses a calibrated neutral-and-navy light theme:

| Token Name | Hex Code | Purpose | Contrast vs Background |
| :--- | :--- | :--- | :--- |
| `COLOR_TEXT_PRIMARY` | `#0F172A` | Primary titles, body text, metric values | 15.8:1 (AAA) |
| `COLOR_TEXT_SECONDARY` | `#475569` | Metric labels, captions, metadata | 5.4:1 (AA) |
| `COLOR_PRIMARY` | `#1E40AF` | Primary accents, selected tabs, dominant bars | 7.9:1 (AAA) |
| `COLOR_ACCENT` | `#3B82F6` | Secondary chart highlights, trend lines | 3.5:1 (UI components) |
| `COLOR_BG_PANEL` | `#F8FAFC` | Metric cards, callout containers, sidebar | Neutral background |
| `COLOR_BORDER` | `#E2E8F0` | Structural dividers, card borders, gridlines | UI element boundary |
| `COLOR_BG_WHITE` | `#FFFFFF` | Plot areas, popovers, table surfaces | Main surface |

### 1.2 Heatmap Sequential Color Scale
For cross-tabulations and category-by-level matrices, the interface utilizes an accessible, single-hue sequential blues scale:
```python
BLUES_SCALE = [
    [0.0, "#EFF6FF"],  # Blue 50
    [0.2, "#DBEAFE"],  # Blue 100
    [0.4, "#93C5FD"],  # Blue 300
    [0.6, "#3B82F6"],  # Blue 500
    [0.8, "#1D4ED8"],  # Blue 700
    [1.0, "#172554"],  # Blue 950
]
```

### 1.3 Typography Scale
The typography stack is rooted in `Inter` and system sans-serif:
- **Platform Title**: `1.55rem` (`24.8px`), Weight 800, tracking `-0.01em`.
- **Section Headers**: `1.15rem` (`18.4px`), Weight 700.
- **Metric Values**: `clamp(0.95rem, 1.25vw, 1.45rem)`, Weight 700.
- **Metric Labels**: `0.80rem` (`12.8px`), Weight 600, uppercase, tracking `+0.03em`.
- **Insight Titles**: `0.92rem` (`14.7px`), Weight 700.
- **Body & Insight Text**: `0.85rem` (`13.6px`), Weight 400, line-height 1.45.
- **Footnotes & Metadata**: `0.78rem` (`12.5px`), Weight 400, italic or muted.

### 1.4 Zero-Animation Policy
All CSS animations, transitions, and hover-lift effects are disabled globally via `edupro/ui/theme.py`:
```css
*, *::before, *::after {
    transition: none !important;
    animation: none !important;
}
```

---

## 2. Reusable UI Components (`edupro/ui/components.py`)

All UI elements are encapsulated into reusable, tested presentation functions:

### 2.1 `kpi_tile(col, label, kpi, delta, delta_color, format_spec, unit_suffix)`
Renders an executive metric tile with statistical grounding:
- Point estimate formatted to specified precision (e.g. `.3f`, `.4f`).
- Optional secondary delta indicator (e.g. `-0.004/yr on courses`).
- Explanatory subtext displaying sample size $n$ and 95% confidence intervals `[ci_low, ci_high]`.
- Native tooltip (`help`) containing statistical definition, observational caveats, and evidence grade.
- Automatic `"Insufficient data"` fallback when value is `NaN`.

### 2.2 `insight_box(insight)`
Renders an executive callout card for analytical findings (`GeneratedInsight`):
- Strong left accent border (`#1E40AF`).
- Evidence and Category badge (e.g., `Descriptive · Evidence: Strong`).
- Point estimate summary aligned to the right.
- Narrative body describing the finding without causal language.
- Collapsible methodology footnote containing data-model caveats and statistical bounds.

### 2.3 `section_header(title, subtitle, scope_text)`
Provides consistent tab headers with descriptive subtitle and dynamic scope metadata.

### 2.4 `evidence_badge(evidence_level)`
Renders standardized pill badges for grading certainty:
- **Strong**: Dark green `#166534` text on light green `#DCFCE7` background.
- **Moderate**: Dark blue `#1E40AF` text on light blue `#DBEAFE` background.
- **Weak**: Amber `#92400E` text on light yellow `#FEF3C7` background.
- **Inconclusive**: Slate `#475569` text on gray `#F1F5F9` background.

### 2.5 `empty_state(message)`
Displays a structured informational panel with corrective instructions when user filters yield zero records.

---

## 3. Chart Standards (`edupro/ui/charts.py`)

Every chart in the dashboard adheres to rigorous BI standards:
1. **Layout**: Uses `PLOTLY_LAYOUT` with zero margins (`t=36, b=36, l=40, r=20`), transparent plot background, and muted gridlines (`#E2E8F0`).
2. **Left-Aligned Titles**: Titles and subtitles are placed above the plot and left-aligned with explicit units.
3. **Explicit Units**: Axis labels must state the metric and unit (e.g. `Years of Experience`, `Rating (Scale: 1.0 - 5.0)`).
4. **Sample Size Annotations**: Scatter plots and category bars explicitly state the sample size $n$ in captions or tooltips.
5. **Low-$n$ Marking**: Heatmaps annotate cells with sample size $n < 3$ using an asterisk (`*`) and include an explanatory legend footnote.
6. **Container Stretch**: All Plotly figures are rendered using `st.plotly_chart(fig, width="stretch")` to prevent deprecated parameter warnings.

---

## 4. Generated Insights Architecture (`edupro/insights.py`)

To eliminate hardcoded commentary, all textual conclusions are dynamically synthesized from `Finding` dataclass instances produced by `AnalyticsEngine`.

### 4.1 Strict Language Governance Rules
The codebase enforces the following governance rules, verified by automated test suites (`tests/test_insights.py` and `tests/test_no_hardcoded_results_or_emojis.py`):
- **Zero Causal Verbs**: Terms such as `causes`, `drives`, `due to`, `leads to`, `results in`, and `proves` are strictly prohibited. The system uses observational phrases like `is associated with`, `demonstrates`, `exhibits`, or `indicates`.
- **Zero Superlatives**: Words like `entirely` or `perfectly` are prohibited.
- **No Single-Instructor Attribution**: The derived many-to-many relationship between instructors and courses is preserved; the system never claims an instructor "owns" a course rating.
- **Non-Significant Findings**: Findings with $p \ge 0.05$ or inconclusive evidence must state:
  > *"No statistically reliable association was found between [variable X] and [variable Y] in the evaluated data."*
- **Insufficient Sample Size Guard**: Slices below minimum sample thresholds return explicit caveats:
  > *"Insufficient sample size (n = X < Y required) to evaluate this finding under active filters."*
- **Fast-Mode Reporting**: Filtered subsets run in fast estimation mode (300 bootstrap replicates) and dynamically append a caveat tag: `"(fast estimation mode active)"`.

---

## 5. Developer Walkthrough: Adding New Features

### 5.1 How to Add a New Generated Insight
1. Define the statistical computation in `edupro/analytics/inference.py` or `stats.py` and ensure it returns a valid `Finding` object.
2. In `edupro/insights.py`, implement a generator function:
   ```python
   def generate_fXX_insight(f: Finding, is_fast_mode: bool = False) -> GeneratedInsight:
       if f.is_insufficient_data or f.n < 5:
           return GeneratedInsight(
               finding_id=f.id,
               title="...",
               body=f"Insufficient sample size (n = {f.n} < 5 required) to evaluate ...",
               badge_text="Insufficient Data",
               badge_evidence="Inconclusive",
           )
       # Construct dynamic copy adhering to language governance
       ...
   ```
3. Register the function in `generate_all_insights(findings, is_fast_mode)`.
4. Add unit assertions in `tests/test_insights.py` verifying minimum-$n$ guards, banned word absence, and fast-mode tagging.

### 5.2 How to Add a New Chart to a Tab
1. Prepare filtered DataFrames in the target tab (e.g. `edupro/ui/tabs/tabX.py`) from `dataset` or `FilteredView`.
2. Construct the chart using `edupro/ui/charts.py` helpers (`build_horizontal_bar`, `build_scatter_with_trendline`, `build_heatmap_matrix`, etc.) or apply `apply_base_layout(fig)`.
3. Render using `st.plotly_chart(fig, width="stretch")`.
4. Include a collapsible `st.expander("Methodology & Governance Notes")` documenting the sample size, unit of analysis, and statistical caveats.
