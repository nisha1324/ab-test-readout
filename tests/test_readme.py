"""Every figure the README quotes, tied to the generated results."""
from pathlib import Path

import pytest

from scripts import data_checks_report as R

README = (Path(__file__).resolve().parents[1] / "README.md").read_text()


@pytest.fixture(scope="module")
def b():
    return R.build()


@pytest.fixture(scope="module")
def f(b):
    return R.facts(b)


def statements(b, f):
    arms, metrics = b["arms"], b["metrics"].set_index("metric_id")
    small = arms[arms.split == "small imbalance"]
    uneq = arms[arms.split == "unequal"]
    worst = small.loc[small.users_short.idxmax()]
    sd = metrics["median_sd_over_mean"]
    return [
        f"on {f['experiments']} real experiments",
        f"results of {f['experiments']} online experiments",
        f"{f['rows']:,} rows",
        f"each of the {f['arms']} treatment arms",
        f"**{f['clean_arms']} of the {f['arms']} arms ({f['clean_experiments']} of {f['experiments']} experiments) pass every check. "
        f"{f['arms'] - f['clean_arms']} carry at least one caution.**",
        f"raises **{f['naive_alarms']} alarms**",
        f"Only **{f['small']}** look like users going missing",
        f"short by {f['small_short_min']:,.0f} to {f['small_short_max']:,.0f} users",
        f"The largest shortfall is {worst.users_short:,.0f} users out of {worst.users / 1e6:.1f} million "
        f"(a treatment share of {100 * worst.treatment_share:.3f}%)",
        f"The other **{f['unequal']}** arms",
        f"treatment shares from {100 * uneq.treatment_share.min():.2f}% to {100 * uneq.treatment_share.max():.2f}%",
        f"{f['unequal_named']} of the {f['unequal']} are consistent with a named design",
        f"finds **{f['stability']['drifting']}** arms where it drifts",
        f"**{f['n_under a week']} arms in {f['short_experiments']} experiments end before day 7.**",
        f"**{f['n_first week not recorded']} arms** have their first snapshot after day 7",
        f"**{f['n_gap in snapshots']}** have 3 or more days with no snapshot",
        f"{100 * f['half_time'][1]:.0f}% of its final value for metric 1 and {100 * f['half_time'][4]:.0f}% for metric 4",
        f"of {sd[2]:.2f}, {sd[3]:.2f} and {sd[4]:.2f}, against {sd[1]:.2f}",
        f"{f['variance_end']} arms have no variance for metrics 2 to 4 at their last snapshot",
        f"would have stopped {f['naive_alarms']} arms here; {f['unequal']} of them",
        f"{f['coverage_caution_arms']} of {f['arms']} arms here ended within a week",
    ]


def test_every_readme_figure_matches_the_results(b, f):
    missing = [s for s in statements(b, f) if s not in README]
    assert not missing, missing


def test_small_imbalances_are_under_a_tenth_of_a_point(b):
    small = b["arms"][b["arms"].split == "small imbalance"]
    assert ((small.treatment_share - 0.5).abs() * 100 < 0.1).all()
    assert "less than a tenth of a point away from 50%" in README


def test_named_designs_in_readme_match_the_code():
    assert set(R.C.DESIGNS) == {"1:2", "34:33"}
    assert "a three-way 34/33/33 split or a 1:2 split" in README


def test_thresholds_in_readme_match_the_code():
    assert (R.C.ALPHA, R.C.NEAR_EVEN, R.C.MIN_DAYS, R.C.LATE_START, R.C.LONG_GAP) == (0.001, 0.005, 7, 7, 3)
    assert "(p < 0.001, half a point around 50%, 7 days, a 3-day gap)" in README


def test_charts_and_links_exist():
    root = Path(__file__).resolve().parents[1]
    for rel in ("results/DATA_CHECKS.md", "results/arm_checks.csv", "results/charts/01_sample_ratio.png",
                "results/charts/02_coverage.png", "results/charts/03_metrics_grow.png",
                "sql/integrity.sql", "sql/arm_summary.sql", "sql/metric_profile.sql",
                "abtest/checks.py", "scripts/download_data.py", "requirements-lock.txt"):
        assert rel.split("/")[-1] in README and (root / rel).exists(), rel
    assert len(list((root / "results" / "charts").glob("*.png"))) == README.count("](results/charts/")


def test_status_line():
    assert "**Status: Part 2 of 4 done**" in README
    assert README.count("~~") == 4      # two roadmap steps struck out


# ------------------------------------------------------------ Part 2
@pytest.fixture(scope="module")
def q():
    from scripts import readout_report as Q
    b2 = Q.build()
    return Q, b2, Q.facts(b2)


def part2_statements(Q, f):
    A = Q.R
    return [
        f"the {f['arms_not_decided']} arms with an unexplained split or under a week of data are not decided on",
        f"{f['arms_warned']} arms are read with a warning",
        f"**{f['tests']} tests in {f['arms_read']} arms**",
        f"**{f['significant']} of {f['tests']} tests ({100 * f['significant'] / f['tests']:.0f}%) are significant at p < {A.ALPHA}.**",
        f"about {f['expected_by_chance']:.0f} would be",
        f"The median significant change is {100 * f['sig_abs_median']:.2f}% of the control mean and the largest is {100 * f['sig_abs_max']:.2f}%",
        f"{f['sig_under_reference']} of the {f['significant']} are under 1%",
        f"higher in {f['up']} of the significant tests and lower in {f['down']}",
        f"**{f['can_detect']} of the {f['tests']} tests could detect it** {100 * A.POWER:.0f}% of the time",
        f"The most sensitive could detect {100 * f['mde_min']:.2f}%, the least sensitive only {100 * f['mde_max']:.1f}%",
        f"about {round(f['need'][1], -3):,} users per group to detect a 1% change in metric 1 and about {round(f['need'][4], -3):,} for metric 4",
        f"Of the {f['quiet']} quiet tests, {f['quiet_inside']} have a confidence interval entirely between −1% and +1%",
        f"The other {f['quiet_open']} are too wide",
        f"up to {100 * f['false_alarm_4']:.1f}% per arm, not 5%",
        f"**{f['arms_any']} of {f['arms_read']} arms ({100 * f['arms_any'] / f['arms_read']:.0f}%) have at least one significant metric.**",
        f"**{f['arms_any_holm']} arms** do, and {f['holm']} of the {f['significant']} significant tests survive. The other {f['significant'] - f['holm']} are unconfirmed",
        f"{f['arms_all']} arms are significant on every metric they report",
        f"None of the {f['warned_significant']} significant tests in arms read with a warning survives",
        f"The median significant change here is {100 * f['sig_abs_median']:.2f}%.",
        f"roughly {f['need'][1] / 1e6:.1f} to {f['need'][4] / 1e6:.1f} million users per group",
        f"would call {f['arms_any']} of {f['arms_read']} arms a result; after correction {f['arms_any_holm']} are",
        f"{f['quiet_inside']} of {f['quiet']} quiet results are tight enough",
        f"The other {f['quiet_open']} say only that the test was too small",
        f"leaves {f['up']} changes up and {f['down']} down",
        f"at the {100 * A.ALPHA:.0f}% level with {100 * A.POWER:.0f}% power and a {100 * A.REFERENCE_LIFT:.0f}% reference change",
    ]


def test_every_part2_figure_matches_the_results(q):
    Q, _, f = q
    missing = [s for s in part2_statements(Q, f) if s not in README]
    assert not missing, missing


def test_part2_statements_in_words(q):
    Q, b2, f = q
    rd = Q.read_rows(b2["table"])
    assert f["arms_mixed_direction"] == 0
    assert "No arm has one metric significantly higher and another significantly lower." in README
    assert (rd[rd.reading == "read with warning"].p_holm < Q.R.ALPHA).sum() == 0
    assert f["z_corr_min"] > 0.5 and "here they move together" in README
    for rel in ("results/READOUT.md", "results/readout.csv", "results/charts/04_effects.png",
                "results/charts/05_detectable_change.png", "results/charts/06_p_values.png",
                "sql/final_snapshot.sql", "abtest/readout.py", "scripts/readout_report.py"):
        root = Path(__file__).resolve().parents[1]
        assert rel.split("/")[-1] in README and (root / rel).exists(), rel
