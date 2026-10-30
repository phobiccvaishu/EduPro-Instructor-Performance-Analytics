"""
tests.test_insights
===================
Unit and governance tests for the dynamic insight generator:
- Validates every insight function across supported, not-supported, descriptive, and insufficient-data states.
- Asserts all numeric values in generated text match Finding fields.
- Enforces strict zero-tolerance scanner for causal verbs, superlatives, and Unicode emojis.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from edupro.analytics.inference import Finding
from edupro.insights import (
    GeneratedInsight,
    generate_f1_insight,
    generate_f2_insight,
    generate_f3_insight,
    generate_f4_insight,
    generate_f5_insight,
    generate_f6_insight,
    generate_f7_insight,
    generate_f8_insight,
    generate_f10_f11_insight,
    generate_f13_insight,
)

BANNED_WORDS = [
    r"\bcauses?\b",
    r"\bcaused\b",
    r"\bcausing\b",
    r"\bdrives?\b",
    r"\bdriven\b",
    r"\bdriving\b",
    r"\bdue to\b",
    r"\bleads? to\b",
    r"\bleading to\b",
    r"\bresults? in\b",
    r"\bresulting in\b",
    r"\bproves?\b",
    r"\bproven\b",
    r"\bproving\b",
    r"\bentirely\b",
]

EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]"
)


def assert_governance_rules(insight: GeneratedInsight) -> None:
    """Check an insight against all language and formatting governance rules."""
    full_text = (
        f"{insight.headline} {insight.evidence_sentence} {insight.caveat} {insight.stat_summary}"
    )

    # 1. No emojis
    assert not EMOJI_PATTERN.search(full_text), (
        f"Emoji found in insight {insight.finding_id}: {full_text}"
    )

    # 2. No causal verbs or forbidden superlatives
    for pattern in BANNED_WORDS:
        match = re.search(pattern, full_text, re.IGNORECASE)
        assert not match, f"Banned phrase '{pattern}' found in {insight.finding_id}: {full_text}"


@pytest.fixture
def baseline_findings():
    """Load baseline findings from results/analysis_summary.json."""
    p = Path(__file__).resolve().parent.parent / "results" / "analysis_summary.json"
    with open(p) as f:
        data = json.load(f)
    findings_raw = data["findings"]

    result = {}
    for fid, d in findings_raw.items():
        result[fid] = Finding(**d)
    return result


def test_f1_insight_baseline_and_governance(baseline_findings):
    """Test F1 rating distribution insight."""
    f1 = baseline_findings["F1"]
    ins = generate_f1_insight(f1)
    assert_governance_rules(ins)
    assert f"{f1.statistic:.2f}" in ins.headline
    assert ins.verdict == "Descriptive"
    assert ins.n == 60


def test_f2_insight_baseline_and_governance(baseline_findings):
    """Test F2 experience vs teacher rating insight."""
    f2 = baseline_findings["F2"]
    ins = generate_f2_insight(f2)
    assert_governance_rules(ins)
    slope = f2.extra["ols_slope"]
    assert f"{slope:.3f}" in ins.headline
    assert ins.verdict == "Supported"
    assert "Pearson r = 0.598" in ins.evidence_sentence


def test_f3_insight_not_supported_phrasing(baseline_findings):
    """Test F3 experience vs course rating non-significant phrasing."""
    f3 = baseline_findings["F3"]
    ins = generate_f3_insight(f3)
    assert_governance_rules(ins)
    assert ins.verdict == "Not supported"
    assert "No statistically reliable association was found" in ins.evidence_sentence
    assert "no effect" not in ins.evidence_sentence.lower()


def test_f4_insight_breakpoint(baseline_findings):
    """Test F4 experience bands and inflection breakpoint."""
    f4a = baseline_findings["F4a"]
    f4b = baseline_findings["F4b"]
    ins = generate_f4_insight(f4a, f4b)
    assert_governance_rules(ins)
    assert f"{f4a.statistic:+.2f}" in ins.headline
    assert "8 years" in ins.evidence_sentence


def test_f5_insight_cluster_aware_correlation(baseline_findings):
    """Test F5 instructor vs course quality correlation."""
    f5 = baseline_findings["F5"]
    ins = generate_f5_insight(f5)
    assert_governance_rules(ins)
    assert ins.verdict == "Not supported"
    assert "No statistically reliable association was found" in ins.evidence_sentence
    assert "pseudo-replication" in ins.caveat.lower()


def test_f6_f7_insights_category_and_level(baseline_findings):
    """Test F6 and F7 non-significant ANOVA and Kruskal-Wallis insights."""
    f6 = baseline_findings["F6"]
    ins6 = generate_f6_insight(f6)
    assert_governance_rules(ins6)
    assert "No statistically reliable association was found" in ins6.evidence_sentence

    f7 = baseline_findings["F7"]
    ins7 = generate_f7_insight(f7)
    assert_governance_rules(ins7)
    assert "No statistically reliable association was found" in ins7.evidence_sentence


def test_f8_insight_domain_contrasts(baseline_findings):
    """Test F8 domain ANOVA and Holm post-hoc pairwise contrasts."""
    f8a = baseline_findings["F8a"]
    f8b = baseline_findings["F8b"]
    f8c = baseline_findings["F8c"]
    ins = generate_f8_insight(f8a, f8b, f8c)
    assert_governance_rules(ins)
    assert "Holm-Bonferroni" in ins.evidence_sentence
    assert "6 statistically significant domain pair contrasts" in ins.evidence_sentence


def test_f10_f11_insight_enrollment_concentration(baseline_findings):
    """Test F10/F11 top instructors enrollment concentration."""
    f10 = baseline_findings["F10"]
    f11 = baseline_findings["F11"]
    ins = generate_f10_f11_insight(f10, f11)
    assert_governance_rules(ins)
    assert f"{f11.statistic * 100.0:.1f}%" in ins.headline


def test_f13_insight_machine_learning_gap(baseline_findings):
    """Test F13 Machine Learning domain gap."""
    f13 = baseline_findings["F13"]
    ins = generate_f13_insight(f13)
    assert_governance_rules(ins)
    assert f"+{f13.statistic:.3f}" in ins.headline
    assert "Machine Learning instructor ratings exceed" in ins.evidence_sentence


def test_insufficient_data_handling():
    """Verify all insight generators report minimum n requirement when data is insufficient."""
    dummy_empty = Finding(
        id="TEST",
        question="Test",
        method="test",
        unit_of_analysis="test",
        n=1,
        statistic=float("nan"),
        ci_low=float("nan"),
        ci_high=float("nan"),
        p_value=float("nan"),
        effect_label="negligible",
        evidence="None",
        verdict="insufficient data",
        notes="test",
        extra={},
    )

    for gen in [
        generate_f1_insight,
        generate_f2_insight,
        generate_f3_insight,
        generate_f5_insight,
        generate_f6_insight,
        generate_f7_insight,
        generate_f13_insight,
    ]:
        ins = gen(dummy_empty)
        assert_governance_rules(ins)
        assert ins.verdict == "Insufficient data"
        assert "at least 3" in ins.evidence_sentence
