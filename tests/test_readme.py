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
    assert "**Status: Part 1 of 4 done**" in README
    assert README.count("~~") == 2      # one roadmap step struck out
