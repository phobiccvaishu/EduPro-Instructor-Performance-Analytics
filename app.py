"""
app.py  —  EduPro Instructor & Course Quality Dashboard
Run:  python -m streamlit run app.py
"""

import warnings
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPro — Teaching Quality Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS — only layout tweaks, no colour overrides on Streamlit internals
st.markdown("""
<style>
/* Container padding with clearance for Streamlit fixed header bar */
.block-container {
    padding-top: 3.6rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
    max-width: 98% !important;
}

/* Medium vertical gap between Streamlit elements */
[data-testid="stVerticalBlock"] {
    gap: 0.85rem !important;
}

/* Medium comfortable metric cards */
[data-testid="stMetric"] {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.7rem 0.95rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    color: #4B5563 !important;
    margin-bottom: 0.15rem !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: #111827 !important;
    line-height: 1.2 !important;
}

/* Section heading */
.sec-header {
    font-size: 1.08rem;
    font-weight: 700;
    color: #111827;
    margin: 1.1rem 0 0.3rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid #2563EB;
}

/* Chart title + subtitle */
.chart-label {
    font-size: 0.95rem;
    font-weight: 600;
    color: #111827;
    margin: 0.8rem 0 0.15rem 0;
}
.chart-sub {
    font-size: 0.8rem;
    color: #6B7280;
    margin: 0 0 0.45rem 0;
}

/* Medium comfortable callout boxes */
.callout-blue, .callout-amber, .callout-green {
    padding: 0.55rem 0.95rem;
    margin: 0.45rem 0 0.75rem 0;
    font-size: 0.84rem;
    line-height: 1.55;
    border-radius: 0 6px 6px 0;
}
.callout-blue { background: #EFF6FF; border-left: 3px solid #2563EB; color: #1E3A8A; }
.callout-amber { background: #FFFBEB; border-left: 3px solid #D97706; color: #78350F; }
.callout-green { background: #F0FDF4; border-left: 3px solid #16A34A; color: #14532D; }

/* Medium dividers */
hr {
    margin: 0.8rem 0 !important;
    border: none !important;
    border-top: 1px solid #E5E7EB !important;
}

/* Medium tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    margin-top: 0.4rem;
    margin-bottom: 0.65rem;
}
.stTabs [data-baseweb="tab"] {
    padding: 7px 16px;
    font-size: 0.88rem;
    font-weight: 600;
}

/* Sidebar spacing */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.55rem !important;
}
[data-testid="stSidebar"] h1 {
    font-size: 1.25rem !important;
    margin-bottom: 0.2rem !important;
}
[data-testid="stSidebar"] hr {
    margin: 0.55rem 0 !important;
}
</style>
""", unsafe_allow_html=True)



# ── Helper functions ──────────────────────────────────────────────────────────
def section(title):
    st.markdown(f"<div class='sec-header'>{title}</div>", unsafe_allow_html=True)

def chart_label(title, sub=""):
    st.markdown(f"<div class='chart-label'>{title}</div>", unsafe_allow_html=True)
    if sub:
        st.markdown(f"<div class='chart-sub'>{sub}</div>", unsafe_allow_html=True)

def note(text, kind="blue"):
    icons = {"blue": "💡", "amber": "⚠️", "green": "✅"}
    st.markdown(
        f"<div class='callout-{kind}'>{icons[kind]} {text}</div>",
        unsafe_allow_html=True,
    )

def divider():
    st.markdown("---")

TIER_COLORS = {"High (≥4.0)": "#16a34a", "Mid (3.0–3.9)": "#d97706", "Low (<3.0)": "#dc2626"}
CHART_FONT  = dict(family="Segoe UI, Arial, sans-serif", size=12, color="#1a1a2e")
CHART_BASE  = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=CHART_FONT,
    margin=dict(t=30, b=10, l=10, r=10),
)



# ── Data loading (cached) ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading EduPro data…")
def load_data():
    from data_loader import get_all_data
    return get_all_data()

@st.cache_data(show_spinner="Computing analytics…")
def run_analytics(_master, _teachers, _courses, _transactions):
    from analytics import AnalyticsEngine
    engine = AnalyticsEngine(_master, _teachers, _courses, _transactions)
    return engine.run_full_analysis()

try:
    data = load_data()
except FileNotFoundError as e:
    st.error(f"Data file not found: {e}")
    st.stop()

master       = data["master"]
teachers_df  = data["teachers"]
courses_df   = data["courses"]
transactions = data["transactions"]
results      = run_analytics(master, teachers_df, courses_df, transactions)
kpis         = results["kpis"]
ep           = results["expertise_performance"]
impact       = results["instructor_impact"]
cq           = results["course_quality"]
evp          = results["experience_vs_performance"]
tc_corr      = results["teacher_course_corr"]


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 EduPro Analytics")
    st.caption("Instructor & Course Quality Dashboard")
    st.divider()

    st.subheader("Filters")

    all_expertise  = sorted(teachers_df["Expertise"].dropna().unique())
    sel_expertise  = st.multiselect("Instructor Expertise", all_expertise, default=all_expertise)

    all_categories = sorted(courses_df["CourseCategory"].dropna().unique())
    sel_categories = st.multiselect("Course Category", all_categories, default=all_categories)

    all_levels     = sorted(courses_df["CourseLevel"].dropna().unique())
    sel_levels     = st.multiselect("Course Level", all_levels, default=all_levels)

    st.divider()
    st.subheader("Rating Range")

    tr_min, tr_max = float(teachers_df["TeacherRating"].min()), float(teachers_df["TeacherRating"].max())
    sel_tr = st.slider("Teacher Rating", round(tr_min, 1), round(tr_max, 1),
                        (round(tr_min, 1), round(tr_max, 1)), 0.1)

    cr_min, cr_max = float(courses_df["CourseRating"].min()), float(courses_df["CourseRating"].max())
    sel_cr = st.slider("Course Rating", round(cr_min, 1), round(cr_max, 1),
                        (round(cr_min, 1), round(cr_max, 1)), 0.1)

    st.divider()
    n_shown = teachers_df["Expertise"].isin(sel_expertise).sum()
    st.caption(f"Showing {n_shown} of {len(teachers_df)} instructors")
    st.caption("EduPro Analytics Suite v1.0")


# ── Apply filters ─────────────────────────────────────────────────────────────
mask = (
    master["Expertise"].isin(sel_expertise) &
    master["CourseCategory"].isin(sel_categories) &
    master["CourseLevel"].isin(sel_levels) &
    master["TeacherRating"].between(sel_tr[0], sel_tr[1]) &
    master["CourseRating"].between(sel_cr[0], sel_cr[1])
)
filtered = master[mask].copy()

from analytics import KPIEngine
f_kpis = KPIEngine(filtered).compute_all_kpis() if len(filtered) > 0 else kpis

teachers_f = teachers_df[
    teachers_df["Expertise"].isin(sel_expertise) &
    teachers_df["TeacherRating"].between(sel_tr[0], sel_tr[1])
]


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='margin-bottom: 0.85rem;'>
    <h1 style='font-size: 1.7rem; font-weight: 800; color: #111827; margin: 0; padding: 0;'>
        🎓 Instructor Performance & Course Quality
    </h1>
    <div style='font-size: 0.86rem; color: #6B7280; margin-top: 0.25rem;'>
        EduPro Online Education Platform · 60 Instructors · 60 Courses · 10,000 Transactions
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# KPI BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='display: flex; justify-content: space-between; align-items: baseline; margin: 0.75rem 0 0.5rem 0;'>
    <span style='font-size: 1.02rem; font-weight: 700; color: #111827;'>Platform KPIs at a Glance</span>
    <span style='font-size: 0.8rem; color: #6B7280;'>Live platform benchmarks across evaluated courses & instructors</span>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)

avg_tr = f_kpis["avg_teacher_rating"]
avg_cr = f_kpis["avg_course_rating"]
rci    = f_kpis["rating_consistency_index"]
eis    = float(kpis["experience_impact"]["impact_score"])
eir    = float(kpis["enrollment_influence"]["enrollment_influence_ratio"])
tr_d   = avg_tr - kpis["avg_teacher_rating"]
cr_d   = avg_cr - kpis["avg_course_rating"]

k1.metric(
    "Avg Teacher Rating",
    f"{avg_tr:.2f} / 5.0",
    delta=f"{tr_d:+.2f} vs all" if abs(tr_d) > 0.001 else None,
    help="Mean rating given to instructors by learners (1–5 scale).",
)
k2.metric(
    "Avg Course Rating",
    f"{avg_cr:.2f} / 5.0",
    delta=f"{cr_d:+.2f} vs all" if abs(cr_d) > 0.001 else None,
    help="Mean rating given to course content by learners (1–5 scale).",
)
k3.metric(
    "Quality Consistency",
    f"{rci:.2f}",
    help="1 = perfectly uniform quality across courses. 0 = highly variable.",
)
k4.metric(
    "Experience Impact (r)",
    f"{eis:.2f}",
    help="Pearson correlation between years of experience and ratings. Closer to 1 = stronger link.",
)
k5.metric(
    "Enrollment Lift (Top Instructors)",
    f"{eir:.1f}×",
    delta="above platform avg",
    delta_color="normal",
    help="High-rated instructors attract this many times more enrollments than the platform average.",
)

# Glossary expander — compact, directly under KPIs without double dividers
with st.expander("📖  What do these terms mean?  (click to open)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**Teacher Rating (1–5)**
How learners score the instructor on teaching quality, clarity, and engagement.

**Course Rating (1–5)**
How learners score the course content and structure — separate from the instructor score.

**Quality Consistency (0–1)**
How similar an instructor's course ratings are across all their courses.
Closer to 1 = very consistent. Closer to 0 = unpredictable.
""")
    with c2:
        st.markdown("""
**Experience Impact (r)**
Pearson correlation between years of teaching and ratings.
Range −1 to +1. Values above 0.4 indicate a meaningful positive link.

**Enrollment Lift**
Ratio of average enrollments for top-rated instructors vs the platform average.
A value of 3.7× means top instructors attract 3.7 times more students.
""")
    with c3:
        st.markdown("""
**Instructor Tiers**
- 🟢 High — Teacher Rating ≥ 4.0
- 🟡 Mid — Teacher Rating 3.0 to 3.9
- 🔴 Low — Teacher Rating < 3.0

**p-value**
Probability the result is due to chance.
p < 0.05 = statistically significant (reliable finding).
""")

st.markdown("<div style='border-top: 1px solid #E5E7EB; margin: 0.85rem 0 0.75rem 0;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Instructor Rankings",
    "Experience & Ratings",
    "Course Quality Maps",
    "Domain Analysis",
    "Rating Relationships",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Instructor Rankings
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    section("Instructor Rankings")
    st.caption(
        "Compare all instructors by their rating, course quality, enrollment numbers, and consistency. "
        "Use the sidebar filters to narrow down by expertise or rating range."
    )

    tm_f = impact["teacher_metrics"][
        impact["teacher_metrics"]["Expertise"].isin(sel_expertise) &
        impact["teacher_metrics"]["TeacherRating"].between(sel_tr[0], sel_tr[1])
    ].copy()

    col_table, col_right = st.columns([3, 2], gap="large")

    with col_table:
        chart_label("Full Instructor Ranking Table",
                    "Click any column header to sort. Hover over a row for more detail.")
        rename_map = {
            "TeacherName": "Instructor", "Expertise": "Expertise Domain",
            "YearsOfExperience": "Experience (yrs)", "TeacherRating": "Teacher Rating",
            "avg_course_rating": "Avg Course Rating", "total_enrollments": "Enrollments",
            "course_count": "Courses Taught", "consistency_index": "Consistency Score",
            "InstructorTier": "Tier",
        }
        display_cols = list(rename_map.keys())
        styled_df = tm_f[display_cols].rename(columns=rename_map).reset_index(drop=True)
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=420,
            column_config={
                "Teacher Rating": st.column_config.ProgressColumn(
                    "Teacher Rating", min_value=0, max_value=5, format="%.2f"),
                "Avg Course Rating": st.column_config.ProgressColumn(
                    "Avg Course Rating", min_value=0, max_value=5, format="%.2f"),
            },
        )

    with col_right:
        chart_label("Instructors by Performance Tier",
                    "High ≥ 4.0  |  Mid 3.0–3.9  |  Low < 3.0")
        tier_counts = tm_f["InstructorTier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig_tier = px.pie(
            tier_counts, names="Tier", values="Count",
            color="Tier", color_discrete_map=TIER_COLORS, hole=0.5,
        )
        fig_tier.update_traces(textinfo="percent+label", textfont_size=12)
        fig_tier.update_layout(
            showlegend=False, height=230,
            margin=dict(t=10, b=10, l=0, r=0),
            paper_bgcolor="white", font=CHART_FONT,
        )
        st.plotly_chart(fig_tier, use_container_width=True)

        chart_label("Average Metrics per Tier")
        ts = impact["tier_summary"].copy()
        ts.columns = ["Tier", "Avg Course Rating", "Avg Enrollments",
                      "Total Enrollments", "Instructors", "Courses"]
        st.dataframe(
            ts.set_index("Tier")[["Avg Course Rating", "Avg Enrollments", "Instructors"]].round(2),
            use_container_width=True,
        )

    divider()

    # Top 15 bar
    chart_label("Top 15 Instructors by Teacher Rating",
                "Colour indicates tier: green = high, amber = mid, red = low.")
    top15 = tm_f.sort_values("TeacherRating", ascending=False).head(15)
    fig_top = px.bar(
        top15, x="TeacherName", y="TeacherRating",
        color="InstructorTier", color_discrete_map=TIER_COLORS,
        text="TeacherRating",
        hover_data={
            "Expertise": True, "YearsOfExperience": True,
            "avg_course_rating": True, "total_enrollments": True,
            "InstructorTier": False,
        },
        labels={
            "TeacherName": "Instructor", "TeacherRating": "Teacher Rating (out of 5)",
            "avg_course_rating": "Avg Course Rating", "total_enrollments": "Enrollments",
        },
    )
    fig_top.update_traces(
        texttemplate="<b>%{text:.2f}</b>",
        textposition="outside",
        textfont=dict(size=12, color="#111827"),
        cliponaxis=False,
    )
    fig_top.update_layout(height=380, xaxis_tickangle=-30, yaxis_range=[0, 5.8],
                          legend_title="Tier", **CHART_BASE)
    st.plotly_chart(fig_top, use_container_width=True)
    note("The top 3 instructors are all in <b>Machine Learning or Cybersecurity</b> — "
         "these domains are producing EduPro's highest-rated teachers.", "green")

    # Bottom 10 bar
    chart_label("Bottom 10 Instructors — Priority for Training Support",
                "These instructors have the lowest ratings and likely need coaching or mentorship.")
    bot10 = tm_f.sort_values("TeacherRating", ascending=True).head(10)
    fig_bot = px.bar(
        bot10, x="TeacherName", y="TeacherRating",
        color="Expertise", text="TeacherRating",
        hover_data={"YearsOfExperience": True, "avg_course_rating": True},
        labels={
            "TeacherName": "Instructor", "TeacherRating": "Teacher Rating (out of 5)",
            "avg_course_rating": "Avg Course Rating",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_bot.update_traces(
        texttemplate="<b>%{text:.2f}</b>",
        textposition="outside",
        textfont=dict(size=12, color="#111827"),
        cliponaxis=False,
    )
    fig_bot.update_layout(height=360, xaxis_tickangle=-30, yaxis_range=[0, 5.8],
                          **CHART_BASE)
    st.plotly_chart(fig_bot, use_container_width=True)
    note("All bottom-10 instructors have <b>fewer than 6 years of experience</b>. "
         "A mentorship programme pairing them with senior colleagues would directly improve these scores.", "amber")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Experience & Ratings
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    section("Does Teaching Experience Improve Ratings?")
    st.caption(
        "We test how strongly years of teaching experience relate to instructor ratings "
        "and course ratings — and where the biggest performance jump happens."
    )

    cs1, cs2, cs3, cs4 = st.columns(4)
    cs1.metric("Experience → Instructor Rating",
               f"r = {evp['pearson_r_teacher']:.3f}",
               help="Pearson correlation. Values above 0.4 indicate a meaningful link.")
    cs2.metric("Statistical Significance",
               "p < 0.001" if evp["pearson_p_teacher"] < 0.001 else f"p = {evp['pearson_p_teacher']:.3f}",
               help="p < 0.05 means the result is reliable and not due to chance.")
    cs3.metric("Experience → Course Rating",
               f"r = {evp['pearson_r_course']:.3f}",
               help="Near-zero value means experience does not predict course quality.")
    cs4.metric("Course Rating Significance",
               f"p = {evp['pearson_p_course']:.2f}",
               help="p > 0.05 means this result is not statistically significant.")

    note(
        f"Experience has a <b>strong positive effect on instructor ratings</b> "
        f"(r = {evp['pearson_r_teacher']:.3f}, p &lt; 0.001). "
        f"But it has <b>almost no effect on course quality</b> (r = {evp['pearson_r_course']:.3f}, p = {evp['pearson_p_course']:.2f}). "
        "Course content quality must be addressed separately from instructor development.",
    )
    divider()

    sc1, sc2 = st.columns(2, gap="large")
    with sc1:
        chart_label("Years of Experience vs Instructor Rating",
                    "Each dot = one instructor. The line shows the overall trend.")
        fig_s1 = px.scatter(
            teachers_f, x="YearsOfExperience", y="TeacherRating",
            color="Expertise", size="TeacherRating",
            hover_data={"TeacherName": True, "Age": True, "Gender": True},
            trendline="ols",
            labels={"YearsOfExperience": "Years of Teaching Experience",
                    "TeacherRating": "Instructor Rating (out of 5)"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_s1.update_layout(height=380, yaxis_range=[0, 5.5],
                              legend=dict(orientation="h", y=-0.3, font=dict(size=10)),
                              **CHART_BASE)
        st.plotly_chart(fig_s1, use_container_width=True)
        note("The upward trend is <b>clear</b> — more experienced instructors consistently score higher.", "green")

    with sc2:
        chart_label("Years of Experience vs Course Rating",
                    "Each dot = one course record. Notice the flat trend line.")
        fig_s2 = px.scatter(
            filtered, x="YearsOfExperience", y="CourseRating",
            color="Expertise",
            hover_data={"TeacherName": True, "CourseName": True, "CourseCategory": True},
            trendline="ols",
            labels={"YearsOfExperience": "Years of Teaching Experience",
                    "CourseRating": "Course Rating (out of 5)"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_s2.update_layout(height=380, showlegend=False, yaxis_range=[0, 5.5],
                              **CHART_BASE)
        st.plotly_chart(fig_s2, use_container_width=True)
        note("The trend line is <b>nearly flat</b> — course content quality is independent of instructor experience.", "amber")

    divider()

    chart_label("Average Instructor Rating by Career Stage",
                "The biggest rating jump happens between 0–5 years and 6–10 years of experience · 🟢 ≥3.5 | 🔵 3.0–3.4 | 🔴 <3.0")
    ga = evp["group_avg"].copy()
    ga.columns = ["Experience Band", "Avg Rating", "Count"]
    ga["TierColor"] = ga["Avg Rating"].apply(
        lambda r: "#16a34a" if r >= 3.5 else ("#2563eb" if r >= 3.0 else "#dc2626")
    )
    fig_ga = px.bar(
        ga, x="Experience Band", y="Avg Rating", text="Avg Rating",
        labels={"Avg Rating": "Avg Instructor Rating (out of 5)", "Experience Band": "Career Stage"},
        custom_data=["Count"],
    )
    fig_ga.update_traces(
        marker_color=ga["TierColor"],
        texttemplate="<b>%{text:.2f}</b>",
        textposition="outside",
        textfont=dict(size=12, color="#111827"),
        width=0.45,
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Average Rating: <b>%{y:.2f}</b> / 5.0<br>Number of Instructors: %{customdata[0]}<extra></extra>",
    )
    fig_ga.update_layout(height=340, yaxis_range=[0, 5.6], **CHART_BASE)
    st.plotly_chart(fig_ga, use_container_width=True)
    note(
        "Instructors with <b>0–5 years</b> average just <b>2.40/5</b>. "
        "Those with <b>6–10 years</b> jump to <b>3.71/5</b> — a +1.31 point gain. "
        "The 6-year mark is the clearest career milestone on EduPro.", "green"
    )

    divider()
    chart_label("Rating Spread by Expertise Domain",
                "The box shows the middle 50% of ratings. Individual dots are instructors. "
                "Wider boxes = more variability within that domain.")
    fig_vio = px.violin(
        teachers_f, x="Expertise", y="TeacherRating",
        box=True, points="all", color="Expertise",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={"Expertise": "Expertise Domain", "TeacherRating": "Instructor Rating (out of 5)"},
    )
    fig_vio.update_layout(height=400, showlegend=False, xaxis_tickangle=-20, **CHART_BASE)
    st.plotly_chart(fig_vio, use_container_width=True)
    note("Domains with wide, low shapes like <b>Design</b> and <b>Business</b> have both "
         "low average ratings and high inconsistency — these need the most targeted support.", "amber")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Course Quality Maps
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    section("Course Quality Across Categories & Levels")
    st.caption(
        "Explore which course categories and difficulty levels earn the highest ratings "
        "and where learner demand is concentrated."
    )

    h1, h2 = st.columns(2, gap="large")
    with h1:
        chart_label("Avg Course Rating — Category × Difficulty Level",
                    "Green = high quality. Red = needs improvement.")
        pivot = cq["cat_level_pivot"]
        fig_hm1 = px.imshow(pivot, text_auto=".2f",
                             color_continuous_scale="RdYlGn", aspect="auto",
                             labels=dict(x="Difficulty", y="Category", color="Avg Rating"),
                             zmin=1, zmax=5)
        fig_hm1.update_layout(height=420, margin=dict(t=10, b=10, l=4, r=4),
                               paper_bgcolor="white", font=CHART_FONT)
        st.plotly_chart(fig_hm1, use_container_width=True)

    with h2:
        chart_label("Enrollments — Category × Difficulty Level",
                    "Darker blue = more learners enrolled in that combination.")
        enr_pivot = cq["enr_pivot"]
        fig_hm2 = px.imshow(enr_pivot, text_auto=True,
                             color_continuous_scale="Blues", aspect="auto",
                             labels=dict(x="Difficulty", y="Category", color="Enrollments"))
        fig_hm2.update_layout(height=420, margin=dict(t=10, b=10, l=4, r=4),
                               paper_bgcolor="white", font=CHART_FONT)
        st.plotly_chart(fig_hm2, use_container_width=True)

    note("Compare both maps: cells with <b>high enrollment but low rating</b> are the most "
         "critical to improve — many learners are experiencing below-average content.", "amber")
    divider()

    chart_label("Avg Course Rating — Instructor Gender × Difficulty Level",
                "Shows whether course quality differs by gender or difficulty level.")
    fig_gl = px.imshow(cq["gender_level"], text_auto=".2f",
                        color_continuous_scale="RdYlBu", aspect="auto",
                        labels=dict(x="Difficulty", y="Instructor Gender", color="Avg Rating"),
                        zmin=1, zmax=5)
    fig_gl.update_layout(height=220, margin=dict(t=10, b=10, l=4, r=4),
                          paper_bgcolor="white", font=CHART_FONT)
    st.plotly_chart(fig_gl, use_container_width=True)
    divider()

    bc1, bc2 = st.columns(2, gap="large")
    with bc1:
        chart_label("Average Course Rating by Category",
                    "Sorted lowest to highest · Hover for variability & counts · 🟢 ≥3.5 | 🔵 3.0–3.4 | 🔴 <3.0")
        bc = cq["by_category"].sort_values("avg_rating", ascending=True).copy()
        bc["TierColor"] = bc["avg_rating"].apply(
            lambda r: "#16a34a" if r >= 3.5 else ("#2563eb" if r >= 3.0 else "#dc2626")
        )
        bc["std_clean"] = bc["std_rating"].fillna(0)
        fig_bc = px.bar(
            bc, x="avg_rating", y="CourseCategory", orientation="h",
            text="avg_rating",
            labels={"avg_rating": "Average Course Rating (out of 5)", "CourseCategory": ""},
            custom_data=["std_clean", "count"],
        )
        fig_bc.update_traces(
            marker_color=bc["TierColor"],
            texttemplate="<b>%{text:.2f}</b>",
            textposition="outside",
            textfont=dict(size=12, color="#111827"),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Average Rating: <b>%{x:.2f}</b> / 5.0<br>Standard Deviation: ±%{customdata[0]:.2f}<br>Number of Courses: %{customdata[1]}<extra></extra>",
        )
        fig_bc.update_layout(
            height=400,
            xaxis=dict(range=[0, 5.4], gridcolor="#f3f4f6", linecolor="#e5e7eb", dtick=1.0),
            yaxis=dict(linecolor="#e5e7eb"),
            **CHART_BASE,
        )
        st.plotly_chart(fig_bc, use_container_width=True)

    with bc2:
        chart_label("Average Course Rating by Difficulty Level",
                    "Course quality across difficulty levels · 🟢 ≥3.5 | 🔵 3.0–3.4 | 🔴 <3.0")
        bl = cq["by_level"].sort_values("avg_rating", ascending=True).copy()
        bl["TierColor"] = bl["avg_rating"].apply(
            lambda r: "#16a34a" if r >= 3.5 else ("#2563eb" if r >= 3.0 else "#dc2626")
        )
        bl["std_clean"] = bl["std_rating"].fillna(0)
        fig_bl = px.bar(
            bl, x="avg_rating", y="CourseLevel", orientation="h",
            text="avg_rating",
            labels={"avg_rating": "Average Course Rating (out of 5)", "CourseLevel": ""},
            custom_data=["std_clean", "count"],
        )
        fig_bl.update_traces(
            marker_color=bl["TierColor"],
            texttemplate="<b>%{text:.2f}</b>",
            textposition="outside",
            textfont=dict(size=12, color="#111827"),
            width=0.45,
            cliponaxis=False,
            hovertemplate="<b>%{y} Level</b><br>Average Rating: <b>%{x:.2f}</b> / 5.0<br>Standard Deviation: ±%{customdata[0]:.2f}<br>Number of Courses: %{customdata[1]}<extra></extra>",
        )
        fig_bl.update_layout(
            height=400,
            xaxis=dict(range=[0, 5.4], gridcolor="#f3f4f6", linecolor="#e5e7eb", dtick=1.0),
            yaxis=dict(linecolor="#e5e7eb"),
            **CHART_BASE,
        )
        st.plotly_chart(fig_bl, use_container_width=True)

    c_note1, c_note2 = st.columns(2, gap="large")
    with c_note1:
        note("<b>Marketing (3.72)</b> and <b>Digital Marketing (3.66)</b> lead in quality. "
             "<b>Machine Learning (2.66)</b> and <b>Business (2.69)</b> are lowest — "
             "despite ML having the highest-rated instructors.", "amber")
    with c_note2:
        note("<b>Advanced courses score lowest (2.81)</b>. Learners expect more at that level. "
             "A content review of advanced offerings is recommended.", "amber")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Domain Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    section("Performance by Expertise Domain")
    st.caption(
        "Which teaching domains produce the best-rated instructors and courses? "
        "Where is learner demand highest, and where are the biggest quality gaps?"
    )

    exp_t = ep["exp_teacher"][ep["exp_teacher"]["Expertise"].isin(sel_expertise)]
    exp_c = ep["exp_course"][ep["exp_course"]["Expertise"].isin(sel_expertise)]

    e1, e2 = st.columns(2, gap="large")
    with e1:
        chart_label("Instructor Rating by Domain",
                    "How instructors are rated in each domain · 🟢 ≥3.5 | 🔵 3.0–3.4 | 🔴 <3.0")
        exp_t_s = exp_t.sort_values("avg_teacher_rating", ascending=True).copy()
        exp_t_s["TierColor"] = exp_t_s["avg_teacher_rating"].apply(
            lambda r: "#16a34a" if r >= 3.5 else ("#2563eb" if r >= 3.0 else "#dc2626")
        )
        exp_t_s["std_clean"] = exp_t_s["std_teacher_rating"].fillna(0)
        fig_et = px.bar(
            exp_t_s, x="avg_teacher_rating", y="Expertise", orientation="h",
            text="avg_teacher_rating",
            labels={"avg_teacher_rating": "Avg Instructor Rating (out of 5)", "Expertise": ""},
            custom_data=["std_clean", "count"],
        )
        fig_et.update_traces(
            marker_color=exp_t_s["TierColor"],
            texttemplate="<b>%{text:.2f}</b>",
            textposition="outside",
            textfont=dict(size=12, color="#111827"),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Average Rating: <b>%{x:.2f}</b> / 5.0<br>Standard Deviation: ±%{customdata[0]:.2f}<br>Instructor Count: %{customdata[1]}<extra></extra>",
        )
        fig_et.update_layout(
            height=410,
            xaxis=dict(range=[0, 5.4], gridcolor="#f3f4f6", linecolor="#e5e7eb", dtick=1.0),
            yaxis=dict(linecolor="#e5e7eb"),
            **CHART_BASE,
        )
        st.plotly_chart(fig_et, use_container_width=True)

    with e2:
        chart_label("Course Rating by Domain",
                    "How course content is rated in each domain · 🟢 ≥3.5 | 🔵 3.0–3.4 | 🔴 <3.0")
        exp_c_s = exp_c.sort_values("avg_course_rating", ascending=True).copy()
        exp_c_s["TierColor"] = exp_c_s["avg_course_rating"].apply(
            lambda r: "#16a34a" if r >= 3.5 else ("#2563eb" if r >= 3.0 else "#dc2626")
        )
        exp_c_s["std_clean"] = exp_c_s["std_course_rating"].fillna(0)
        fig_ec = px.bar(
            exp_c_s, x="avg_course_rating", y="Expertise", orientation="h",
            text="avg_course_rating",
            labels={"avg_course_rating": "Avg Course Rating (out of 5)", "Expertise": ""},
            custom_data=["std_clean", "course_count"],
        )
        fig_ec.update_traces(
            marker_color=exp_c_s["TierColor"],
            texttemplate="<b>%{text:.2f}</b>",
            textposition="outside",
            textfont=dict(size=12, color="#111827"),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Average Rating: <b>%{x:.2f}</b> / 5.0<br>Standard Deviation: ±%{customdata[0]:.2f}<br>Course Count: %{customdata[1]}<extra></extra>",
        )
        fig_ec.update_layout(
            height=410,
            xaxis=dict(range=[0, 5.4], gridcolor="#f3f4f6", linecolor="#e5e7eb", dtick=1.0),
            yaxis=dict(linecolor="#e5e7eb"),
            **CHART_BASE,
        )
        st.plotly_chart(fig_ec, use_container_width=True)

    note(
        "<b>Machine Learning</b> instructors rank highest (avg 4.23) but their courses rank lowest (avg 2.66). "
        "This is the platform's biggest quality mismatch — great teachers but the course content needs a redesign. "
        "<b>Marketing</b> shows the opposite pattern.", "amber"
    )
    divider()

    chart_label("Instructor Rating Distribution by Domain",
                "The box shows the middle 50% of ratings. Individual dots are instructors. "
                "Wide boxes indicate inconsistent quality within that domain.")
    teachers_f2 = teachers_df[teachers_df["Expertise"].isin(sel_expertise)]
    fig_box = px.box(teachers_f2, x="Expertise", y="TeacherRating",
                     color="Expertise", points="all",
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     labels={"Expertise": "", "TeacherRating": "Instructor Rating (out of 5)"})
    fig_box.update_layout(height=400, showlegend=False, xaxis_tickangle=-20, **CHART_BASE)
    st.plotly_chart(fig_box, use_container_width=True)
    note("<b>Business</b> and <b>Design</b> have the lowest medians and widest spread. "
         "Instructor quality in these domains is both poor and inconsistent.", "amber")
    divider()

    e3, e4 = st.columns(2, gap="large")
    with e3:
        chart_label("Total Learner Enrollments by Domain",
                    "Which expertise areas attract the most learners overall?")
        exp_enr = exp_c.sort_values("total_enrollments", ascending=True)
        fig_enr = px.bar(exp_enr, x="total_enrollments", y="Expertise", orientation="h",
                         color="total_enrollments", text="total_enrollments",
                         color_continuous_scale="Blues",
                         labels={"total_enrollments": "Total Enrollments", "Expertise": ""})
        fig_enr.update_traces(texttemplate="%{text:,}", textposition="outside", textfont_size=11)
        fig_enr.update_layout(height=410, coloraxis_showscale=False, **CHART_BASE)
        st.plotly_chart(fig_enr, use_container_width=True)

    with e4:
        chart_label("Course Rating — Expertise Domain × Course Category",
                    "Each cell = average course rating. White = no courses in that combination.")
        exp_cat_f = ep["exp_cat_pivot"][ep["exp_cat_pivot"].index.isin(sel_expertise)]
        fig_exc = px.imshow(exp_cat_f, text_auto=".2f",
                             color_continuous_scale="RdYlGn", aspect="auto",
                             labels=dict(x="Course Category", y="Expertise Domain", color="Avg Rating"),
                             zmin=1, zmax=5)
        fig_exc.update_layout(height=410, margin=dict(t=10, b=10, l=4, r=4),
                               paper_bgcolor="white", font=CHART_FONT)
        fig_exc.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_exc, use_container_width=True)

    divider()
    chart_label("Does a Better Instructor Produce a Better Course? — By Domain",
                "Pearson r close to +1 means instructor quality drives course quality in that domain. "
                "Close to 0 means course quality depends on other factors like content design.")
    corr_df = ep["exp_corr"][ep["exp_corr"]["Expertise"].isin(sel_expertise)].copy()
    corr_df["Interpretation"] = corr_df["r"].apply(
        lambda r: "Strong link" if r > 0.5 else ("Moderate link" if r > 0.25 else
                  ("Weak link" if r > 0 else "No link / Inverse"))
    )
    corr_df = corr_df.rename(columns={"r": "Pearson r", "p": "p-value", "n": "Records"})
    st.dataframe(corr_df.reset_index(drop=True), use_container_width=True, height=280)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Rating Relationships
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    section("How Do Teacher Ratings Relate to Course Ratings?")
    st.caption(
        "Does a highly-rated instructor automatically produce a highly-rated course? "
        "The answer might surprise you."
    )

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Pearson Correlation (r)", f"{tc_corr['pearson_r']:.3f}",
               help="−1 to +1. Near 0 means no linear relationship.")
    rc2.metric("Spearman Correlation (r)", f"{tc_corr['spearman_r']:.3f}",
               help="Rank-based correlation. Also near 0 here.")
    rc3.metric("Regression Slope", f"{tc_corr['slope']:.3f}",
               help="Change in course rating for each 1-point increase in teacher rating.")
    rc4.metric("Records Analysed", f"{tc_corr['n']:,}",
               help="Total instructor-course pairs used in the analysis.")

    note(
        f"The correlation between instructor rating and course rating is "
        f"<b>r = {tc_corr['pearson_r']:.3f}</b> (p = {tc_corr['pearson_p']:.2f}) — "
        "statistically <b>not significant</b>. A 5-star instructor does not automatically "
        "produce a 5-star course. Course content quality must be evaluated and improved "
        "<b>separately</b> from instructor quality.",
        "amber",
    )
    divider()

    chart_label("Instructor Rating vs Course Rating — All Courses",
                "Each bubble = one course. Size = number of learners enrolled. "
                "The near-flat trend line confirms the weak relationship.")
    fig_tc = px.scatter(
        filtered, x="TeacherRating", y="CourseRating",
        color="Expertise", size="EnrollmentCount",
        hover_data={"TeacherName": True, "CourseName": True, "CourseCategory": True,
                    "CourseLevel": True, "YearsOfExperience": True, "EnrollmentCount": True},
        trendline="ols", opacity=0.72,
        labels={"TeacherRating": "Instructor Rating (out of 5)",
                "CourseRating": "Course Rating (out of 5)",
                "Expertise": "Domain", "EnrollmentCount": "Enrollments"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig_tc.update_layout(height=500,
                          legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
                          **CHART_BASE)
    st.plotly_chart(fig_tc, use_container_width=True)
    divider()

    dist1, dist2 = st.columns(2, gap="large")
    with dist1:
        chart_label("Instructor Rating Distribution",
                    "How ratings are spread across all instructors.")
        fig_th = px.histogram(
            teachers_df[teachers_df["Expertise"].isin(sel_expertise)],
            x="TeacherRating", nbins=20, marginal="box",
            labels={"TeacherRating": "Instructor Rating (out of 5)"},
            color_discrete_sequence=["#2563eb"],
        )
        fig_th.update_layout(height=360, **CHART_BASE)
        st.plotly_chart(fig_th, use_container_width=True)
        note(f"Instructor ratings range from <b>1.05 to 4.97</b>, average "
             f"<b>{kpis['avg_teacher_rating']:.2f}</b>. Nearly half fall below 3.0.")

    with dist2:
        chart_label("Course Rating Distribution",
                    "How ratings are spread across all courses.")
        courses_f2 = courses_df[
            courses_df["CourseCategory"].isin(sel_categories) &
            courses_df["CourseLevel"].isin(sel_levels)
        ]
        fig_ch = px.histogram(
            courses_f2, x="CourseRating", nbins=20, marginal="box",
            labels={"CourseRating": "Course Rating (out of 5)"},
            color_discrete_sequence=["#16a34a"],
        )
        fig_ch.update_layout(height=360, **CHART_BASE)
        st.plotly_chart(fig_ch, use_container_width=True)
        note(f"Course ratings average <b>{kpis['avg_course_rating']:.2f}/5</b> — "
             "driven by content quality, not instructor reputation.")

    divider()
    chart_label("Instructor vs Course Rating — Side-by-Side by Domain",
                "Blue = instructor rating. Green = course rating. "
                "Large gaps show where instructor quality does not translate to course quality.")
    exp_combined = pd.merge(
        ep["exp_teacher"][["Expertise", "avg_teacher_rating"]],
        ep["exp_course"][["Expertise", "avg_course_rating"]],
        on="Expertise",
    )
    exp_combined = exp_combined[exp_combined["Expertise"].isin(sel_expertise)]
    exp_melt = exp_combined.melt(
        id_vars="Expertise",
        value_vars=["avg_teacher_rating", "avg_course_rating"],
        var_name="Rating Type", value_name="Average Rating",
    )
    exp_melt["Rating Type"] = exp_melt["Rating Type"].map({
        "avg_teacher_rating": "Instructor Rating",
        "avg_course_rating":  "Course Rating",
    })
    fig_grouped = px.bar(
        exp_melt, x="Expertise", y="Average Rating",
        color="Rating Type", barmode="group", text="Average Rating",
        color_discrete_map={"Instructor Rating": "#2563eb", "Course Rating": "#16a34a"},
        labels={"Expertise": "Domain", "Average Rating": "Rating (out of 5)"},
    )
    fig_grouped.update_traces(texttemplate="%{text:.2f}", textposition="outside", textfont_size=10)
    fig_grouped.update_layout(height=420, xaxis_tickangle=-20, yaxis_range=[0, 5.8],
                               legend=dict(orientation="h", y=1.05), **CHART_BASE)
    st.plotly_chart(fig_grouped, use_container_width=True)
    note(
        "<b>Machine Learning</b> shows the largest gap: instructor rating 4.23 vs course rating 2.66 — "
        "a <b>1.57-point mismatch</b>. This is the clearest signal that curriculum redesign is needed "
        "independently of instructor quality.",
        "amber",
    )


# ── Footer ────────────────────────────────────────────────────────────────────
divider()
st.caption(
    "EduPro Analytics Suite v1.0  ·  "
    "Built with Streamlit & Plotly  ·  "
    "Data: 60 Instructors · 60 Courses · 10,000 Transactions"
)
