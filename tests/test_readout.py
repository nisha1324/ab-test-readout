"""Part 2: the readout formulas by hand, against scipy, by simulation, and on
the committed data."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abtest import readout as R
from scripts import readout_report as Q


# ------------------------------------------------------------ by hand
def test_welch_by_hand():
    # control: mean 10, variance 4, n 100; treatment: mean 11, variance 9, n 50
    diff, se, df, p = R.welch(10, 11, 4, 9, 100, 50)
    assert diff == 1
    assert se == pytest.approx(math.sqrt(4 / 100 + 9 / 50))            # sqrt(0.22)
    assert df == pytest.approx(0.22 ** 2 / (0.04 ** 2 / 99 + 0.18 ** 2 / 49))
    assert p == pytest.approx(2 * stats.t.sf(1 / math.sqrt(0.22), df))


@pytest.mark.parametrize("args", [(10, 11, 4, 9, 100, 50), (0.2, 0.21, 0.16, 0.1659, 5000, 4000),
                                  (3.0, 2.9, 80.0, 75.0, 1200, 3600)])
def test_welch_agrees_with_scipy(args):
    mc, mt, vc, vt, nc, nt = args
    expected = stats.ttest_ind_from_stats(mt, math.sqrt(vt), nt, mc, math.sqrt(vc), nc, equal_var=False)
    diff, se, df, p = R.welch(*args)
    assert diff / se == pytest.approx(expected.statistic)
    assert p == pytest.approx(expected.pvalue)


def test_welch_is_symmetric_and_rejects_empty_groups():
    a = R.welch(10, 11, 4, 9, 100, 50)
    b = R.welch(11, 10, 9, 4, 50, 100)
    assert a[0] == -b[0] and a[1:] == pytest.approx(b[1:])
    assert R.welch(5, 5, 1, 1, 10, 10)[3] == 1.0
    with pytest.raises(ValueError):
        R.welch(1, 1, 1, 1, 1, 10)
    with pytest.raises(ValueError):
        R.welch(1, 2, 0, 0, 10, 10)


def test_relative_lift_by_hand():
    # control mean 2 (variance 1, n 100), treatment mean 2.2 (variance 1, n 100)
    lift, se, lo, hi = R.relative_lift(2, 2.2, 1, 1, 100, 100)
    assert lift == pytest.approx(0.10)
    # 0.01 / 4 + 2.2^2 * 0.01 / 16 = 0.0025 + 0.003025
    assert se == pytest.approx(math.sqrt(0.005525))
    assert (lo, hi) == pytest.approx((0.10 - 1.959964 * se, 0.10 + 1.959964 * se), rel=1e-6)
    assert R.relative_lift(2, 2.2, 1, 1, 100, 100, alpha=0.01)[3] > hi
    with pytest.raises(ValueError):
        R.relative_lift(0, 1, 1, 1, 10, 10)


def test_relative_lift_with_no_change_matches_the_plain_standard_error():
    # equal means: the delta-method SE is the SE of the difference over the mean
    _, se, lo, hi = R.relative_lift(5, 5, 4, 4, 400, 400)
    assert se == pytest.approx(math.sqrt(4 / 400 + 4 / 400) / 5)
    assert lo == pytest.approx(-hi)


def test_delta_interval_covers_the_truth_about_95_times_in_100():
    """Simulated skewed data (exponential), true lift 2%: the interval should
    contain 0.02 in roughly 95% of 2,000 experiments."""
    rng = np.random.default_rng(20261123)
    reps, n_c, n_t, hits = 2000, 4000, 3000, 0
    for _ in range(reps):
        c = rng.exponential(10.0, n_c)
        t = rng.exponential(10.2, n_t)
        _, _, lo, hi = R.relative_lift(c.mean(), t.mean(), c.var(ddof=1), t.var(ddof=1), n_c, n_t)
        hits += lo <= 0.02 <= hi
    assert 0.935 <= hits / reps <= 0.965


def test_power_multiplier_and_mde():
    assert R.power_multiplier() == pytest.approx(1.959964 + 0.841621, abs=1e-5)
    assert R.power_multiplier(0.05, 0.5) == pytest.approx(1.959964, abs=1e-5)   # z(0.5) = 0
    assert R.mde_relative(mean_c=20, se_diff=0.1) == pytest.approx(2.801585 * 0.1 / 20, rel=1e-5)
    assert R.mde_relative(20, 0.1, power=0.9) > R.mde_relative(20, 0.1)


def test_mde_is_detected_with_the_stated_power():
    """A test whose true change equals its MDE is significant about 80% of the time."""
    rng = np.random.default_rng(7)
    n, sd, mean = 2000, 3.0, 10.0
    se = math.sqrt(2 * sd ** 2 / n)
    true_t = mean * (1 + R.mde_relative(mean, se))
    found, reps = 0, 2000
    for _ in range(reps):
        c, t = rng.normal(mean, sd, n), rng.normal(true_t, sd, n)
        found += R.welch(c.mean(), t.mean(), c.var(ddof=1), t.var(ddof=1), n, n)[3] < R.ALPHA
    assert 0.77 <= found / reps <= 0.83


def test_users_per_group():
    # SD / mean = 2, detect 1%: 2 * (2.801585 * 2 / 0.01)^2 = 627,910.3...
    assert R.users_per_group(2.0, 0.01) == math.ceil(2 * (R.power_multiplier() * 200) ** 2)
    assert 627_000 < R.users_per_group(2.0, 0.01) < 629_000
    assert R.users_per_group(2.0, 0.005) == pytest.approx(4 * R.users_per_group(2.0, 0.01), rel=1e-5)
    assert R.users_per_group(2.0, -0.01) == R.users_per_group(2.0, 0.01)
    with pytest.raises(ValueError):
        R.users_per_group(2.0, 0)


def test_users_per_group_and_mde_are_two_views_of_one_formula():
    cv, n = 2.5, R.users_per_group(2.5, 0.01)
    se_over_mean = math.sqrt(2 * cv ** 2 / n)
    assert R.mde_relative(1.0, se_over_mean) == pytest.approx(0.01, rel=1e-4)


def test_any_false_alarm():
    assert R.any_false_alarm(1) == pytest.approx(0.05)
    assert R.any_false_alarm(4) == pytest.approx(1 - 0.95 ** 4) == pytest.approx(0.18549375)
    assert R.any_false_alarm(4, 0.01) == pytest.approx(1 - 0.99 ** 4)


def test_holm_by_hand():
    # sorted: 0.01*4, 0.02*3, 0.03*2, 0.04*1 -> 0.04, 0.06, 0.06, 0.06 (kept non-decreasing)
    assert R.holm([0.04, 0.01, 0.03, 0.02]) == pytest.approx([0.06, 0.04, 0.06, 0.06])
    assert R.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert R.holm([0.03]) == pytest.approx([0.03])
    assert len(R.holm([])) == 0


def test_benjamini_hochberg_by_hand():
    # sorted: 0.01*4/1, 0.02*4/2, 0.03*4/3, 0.20*4/4 -> 0.04, 0.04, 0.04, 0.20
    assert R.benjamini_hochberg([0.20, 0.01, 0.03, 0.02]) == pytest.approx([0.20, 0.04, 0.04, 0.04])
    assert R.benjamini_hochberg([0.9, 0.8]) == pytest.approx([0.9, 0.9])


def test_corrections_agree_with_scipy_and_never_lower_a_p_value():
    rng = np.random.default_rng(3)
    p = np.concatenate([rng.uniform(size=40), rng.uniform(0, 0.01, size=10)])
    assert R.benjamini_hochberg(p) == pytest.approx(stats.false_discovery_control(p, method="bh"))
    for adj in (R.holm(p), R.benjamini_hochberg(p)):
        assert (adj >= p - 1e-15).all() and (adj <= 1).all()
        assert (np.argsort(adj, kind="stable") == np.argsort(p, kind="stable")).all() or \
            (np.diff(adj[np.argsort(p)]) >= -1e-15).all()
    assert (R.holm(p) >= R.benjamini_hochberg(p) - 1e-15).all()   # Holm is the stricter one
    assert (R.holm(p) <= np.minimum(p * len(p), 1) + 1e-15).all()  # and never stricter than Bonferroni


@pytest.mark.parametrize("cautions, has_var, expected", [
    ("", True, "read"),
    ("gap in snapshots; first week not recorded", True, "read"),
    ("sample-ratio mismatch", True, "read with warning"),
    ("split drifts; gap in snapshots", True, "read with warning"),
    ("unequal split", True, "not decided"),
    ("under a week", True, "not decided"),
    ("sample-ratio mismatch; under a week", True, "not decided"),
    ("unequal split; variance missing", False, "not decided"),
    ("variance missing", False, "not testable"),
    ("variance missing", True, "read"),          # metric 1 of an arm that lacks variances for 2-4
    ("split drifts; variance missing", False, "not testable"),
])
def test_reading(cautions, has_var, expected):
    assert R.reading(cautions, has_var) == expected


def toy():
    final = pd.DataFrame({
        "experiment_id": ["a", "a", "a", "a", "b", "b", "c"],
        "variant_id": [1, 1, 2, 2, 1, 1, 1],
        "metric_id": [1, 2, 1, 2, 1, 2, 1],
        "day": 14.0,
        "users_c": 10_000, "users_t": 10_000,
        "mean_c": [0.20, 2.0, 0.20, 2.0, 0.30, 1.0, 0.5],
        "mean_t": [0.23, 2.0, 0.20, 2.1, 0.30, 1.0, 0.6],
        "variance_c": [0.16, 9.0, 0.16, 9.0, 0.21, np.nan, 0.25],
        "variance_t": [0.1771, 9.0, 0.16, 9.0, 0.21, 4.0, 0.24],
    })
    checks = pd.DataFrame({"experiment_id": ["a", "a", "b", "c"], "variant_id": [1, 2, 1, 1],
                           "cautions": ["", "split drifts", "variance missing", "under a week"]})
    return final, checks


def test_read_arms_on_a_toy_table():
    final, checks = toy()
    t = R.read_arms(final, checks)
    assert list(t.reading) == ["read", "read", "read with warning", "read with warning",
                               "read", "not testable", "not decided"]
    # corrections: experiment a is one family of 4, b a family of 1, c is left out
    a = t[t.experiment_id == "a"]
    assert a.p_holm.tolist() == pytest.approx(R.holm(a.p).tolist())
    assert t.p_holm[4] == pytest.approx(t.p[4])
    assert math.isnan(t.p_holm[5]) and math.isnan(t.p_holm[6]) and math.isnan(t.p_bh[6])
    assert t.p_bh[:5].tolist() == pytest.approx(R.benjamini_hochberg(t.p[:5]).tolist())
    # the untestable row keeps its point estimate but has no test
    assert t.lift[5] == pytest.approx(0) and math.isnan(t.p[5]) and math.isnan(t.mde[5])
    # the undecided row is computed but not corrected or counted
    assert t.p[6] < 0.05 and t.lift[6] == pytest.approx(0.2)
    assert t.lift[0] == pytest.approx(0.15) and t.p[0] < 1e-6 and t.p[1] == 1.0


# ------------------------------------------------------------ the real data
@pytest.fixture(scope="module")
def b():
    return Q.build()


@pytest.fixture(scope="module")
def f(b):
    return Q.facts(b)


def test_final_snapshot_sql_agrees_with_pandas(b):
    from abtest import load
    raw = load.read_raw()
    last = raw[raw.metric_id == 1].groupby(["experiment_id", "variant_id"]).time_since_start.max().rename("last")
    exp = raw.merge(last.reset_index(), on=["experiment_id", "variant_id"])
    exp = exp[exp.time_since_start == exp["last"]].sort_values(["experiment_id", "variant_id", "metric_id"])
    t = b["table"]
    assert len(t) == len(exp) == 99 * 4
    assert not t.duplicated(["experiment_id", "variant_id", "metric_id"]).any()
    for mine, theirs in (("users_c", "count_c"), ("users_t", "count_t"), ("mean_c", "mean_c"),
                         ("mean_t", "mean_t"), ("variance_c", "variance_c"), ("day", "time_since_start")):
        assert np.allclose(t[mine], exp[theirs].to_numpy(), equal_nan=True), mine


def test_real_tests_agree_with_scipy(b):
    rd = Q.read_rows(b["table"])
    exp = stats.ttest_ind_from_stats(rd.mean_t, np.sqrt(rd.variance_t), rd.users_t,
                                     rd.mean_c, np.sqrt(rd.variance_c), rd.users_c, equal_var=False)
    assert np.allclose(rd.p, exp.pvalue, rtol=1e-8, atol=1e-300)
    assert np.allclose(rd["diff"] / rd.se_diff, exp.statistic)


def test_binary_metric_agrees_with_a_two_proportion_test(b):
    """Metric 1 is yes/no, so an unpooled two-proportion z-test must give
    almost the same p-value as Welch (same standard error up to n vs n-1)."""
    m1 = Q.read_rows(b["table"]).query("metric_id == 1")
    se = np.sqrt(m1.mean_c * (1 - m1.mean_c) / m1.users_c + m1.mean_t * (1 - m1.mean_t) / m1.users_t)
    p = 2 * stats.norm.sf(np.abs(m1.mean_t - m1.mean_c) / se)
    assert np.allclose(m1.p, p, rtol=1e-3, atol=1e-12)


def test_reading_counts_follow_part_one(b, f):
    t, arms = b["table"], b["arms"]
    items = arms.cautions.str.split("; ")
    undecided = items.apply(lambda i: "unequal split" in i or "under a week" in i)
    assert f["arms_not_decided"] == undecided.sum() == f["arms_unequal"] + f["arms_short"] == 14
    assert f["reading"]["not decided"] == 4 * 14
    assert f["arms_read"] == 99 - 14 == 85
    assert f["reading"]["not testable"] == 3 * f["arms_untestable"] == 15
    assert sum(f["reading"].values()) == f["rows"] == 396
    assert f["tests"] == f["reading"]["read"] + f["reading"]["read with warning"] == 325
    warned = arms[~undecided & items.apply(lambda i: "sample-ratio mismatch" in i or "split drifts" in i)]
    assert f["arms_warned"] == len(warned) == 6
    assert set(t[t.reading == "not testable"].metric_id) == {2, 3, 4}


def test_counts_add_up(b, f):
    rd = Q.read_rows(b["table"])
    assert f["significant"] + f["quiet"] == f["tests"]
    assert f["up"] + f["down"] == f["significant"] == b["by_metric"].significant.sum()
    assert f["quiet_inside"] + f["quiet_open"] == f["quiet"]
    assert sum(f["arms_by_count"].values()) == f["arms_read"]
    assert sum(k * v for k, v in f["arms_by_count"].items()) == f["significant"]
    assert f["arms_any"] == f["arms_read"] - f["arms_by_count"][0]
    assert f["holm"] <= f["significant"] and f["bh"] <= f["significant"]
    assert f["arms_any_holm"] <= f["arms_any"]
    assert b["by_metric"].tests.sum() == f["tests"]
    assert b["by_metric"].significant_holm.sum() == f["holm"]
    assert b["by_metric"].can_detect_reference.sum() == f["can_detect"]
    assert f["ci_test_disagree"] == 0
    assert f["arms_mixed_direction"] == 0
    assert (rd.p_holm >= rd.p).all() and (rd.p_bh >= rd.p).all()
    assert (rd.lift_lo < rd.lift).all() and (rd.lift < rd.lift_hi).all()
    assert (rd.mde > 0).all()
    assert f["max_family"] == 8 and f["expected_by_chance"] == pytest.approx(16.25)


def test_statements_made_in_words(b, f):
    rd = Q.read_rows(b["table"])
    sig = rd[rd.p < R.ALPHA]
    # "the largest is arm ee6ff7 / 1"; "no significant change is as large as 5%"
    assert f["biggest"].experiment_id == "ee6ff7" and sig.lift.abs().max() < 0.05
    # "almost half": at least one significant metric in 45-50% of arms
    assert 0.45 < f["arms_any"] / f["arms_read"] < 0.5
    # "the four metrics move together"
    assert f["z_corr_min"] >= 0.7
    # "most are small" in chart 04: more than half of the significant changes are under 1%
    assert f["sig_under_reference"] > f["significant"] / 2
    # none of the significant tests in warned arms survives the correction
    warned = rd[rd.reading == "read with warning"]
    assert (warned.p < R.ALPHA).sum() == f["warned_significant"] == 4 and (warned.p_holm < R.ALPHA).sum() == 0
    # sample sizes use the Part 1 spread figures
    assert f["need"][4] / f["need"][1] == pytest.approx((f["cv"][4] / f["cv"][1]) ** 2, rel=1e-4)
    assert f["min_group"] == rd[["users_c", "users_t"]].to_numpy().min() > 30_000


def test_chart_inputs(b, f):
    counts = Q.p_histogram(b["table"])
    assert counts.sum() == f["tests"] and counts[0] == f["significant"] and len(counts) == 20


def test_report_csvs_and_charts_regenerate_byte_for_byte(b, f, tmp_path):
    assert Q.render(b, f) == (Q.RESULTS / "READOUT.md").read_text()
    b["table"].to_csv(tmp_path / "t.csv", index=False, float_format="%.6g", lineterminator="\n")
    b["by_metric"].to_csv(tmp_path / "m.csv", index=False, float_format="%.6g", lineterminator="\n")
    assert (tmp_path / "t.csv").read_bytes() == (Q.RESULTS / "readout.csv").read_bytes()
    assert (tmp_path / "m.csv").read_bytes() == (Q.RESULTS / "readout_by_metric.csv").read_bytes()
    Q.chart_effects(b["table"], tmp_path / "4.png")
    Q.chart_mde(b["table"], tmp_path / "5.png")
    Q.chart_pvalues(b["table"], tmp_path / "6.png")
    for made, kept in (("4.png", "04_effects.png"), ("5.png", "05_detectable_change.png"), ("6.png", "06_p_values.png")):
        assert (tmp_path / made).read_bytes() == (Q.CHARTS / kept).read_bytes(), kept
