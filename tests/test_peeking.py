"""Part 3: the peeking rules by hand, the simulation against known results,
and the replay on the committed data."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abtest import peeking as P
from abtest import readout as R
from scripts import peeking_report as Q


# ------------------------------------------------------------ the rule, by hand
def test_mixture_ratio_by_hand():
    # lift 2%, standard error 1%, tau 1%: V = tau^2 = 1e-4
    # sqrt(V / 2V) * exp(1e-4 * 4e-4 / (2 * 1e-4 * 2e-4)) = sqrt(0.5) * e
    assert P.mixture_ratio(0.02, 0.01, tau=0.01) == pytest.approx(math.sqrt(0.5) * math.e)
    assert P.mixture_ratio(-0.02, 0.01, tau=0.01) == pytest.approx(math.sqrt(0.5) * math.e)
    # no difference at all is evidence for "no change": the ratio is below 1
    assert P.mixture_ratio(0.0, 0.01, tau=0.01) == pytest.approx(math.sqrt(0.5))
    for bad in ((0.01, 0.0, 0.01), (0.01, 0.01, 0.0)):
        with pytest.raises(ValueError):
            P.mixture_ratio(*bad)


def test_mixture_ratio_is_the_average_likelihood_ratio():
    # Integrate the likelihood ratio N(lift; theta, se) / N(lift; 0, se) over theta ~ N(0, tau).
    lift, se, tau = 0.013, 0.004, 0.01
    theta = np.linspace(-0.1, 0.1, 400_001)
    lr = stats.norm.pdf(lift, theta, se) / stats.norm.pdf(lift, 0, se)
    integral = float(np.sum(lr * stats.norm.pdf(theta, 0, tau)) * (theta[1] - theta[0]))
    assert P.mixture_ratio(lift, se, tau) == pytest.approx(integral, rel=1e-6)


@pytest.mark.parametrize("se", [0.001, 0.004, 0.02, 0.2])
def test_bound_is_where_the_ratio_reaches_one_over_alpha(se):
    z = float(P.always_valid_bound(se))
    assert P.mixture_ratio(z * se, se) == pytest.approx(1 / P.ALPHA)
    assert z > stats.norm.isf(P.ALPHA / 2)          # always stricter than a single test


def test_ratio_from_z_matches_the_ratio_at_each_look():
    se_end, fractions = 0.004, np.array([0.25, 0.5, 1.0])
    z = np.array([[1.0, -2.5, 3.2]])
    got = P.ratio_from_z(z, fractions, P.TAU / se_end)
    for j, t in enumerate(fractions):
        se = se_end / math.sqrt(t)                  # the standard error shrinks as users arrive
        assert got[0, j] == pytest.approx(P.mixture_ratio(z[0, j] * se, se))


def test_info_fractions():
    # 100 + 100 users -> 50; 300 + 100 -> 75; 400 + 400 -> 200
    assert P.info_fractions([100, 300, 400], [100, 100, 400]) == pytest.approx([0.25, 0.375, 1.0])
    with pytest.raises(ValueError):
        P.info_fractions([100, 50], [100, 50])


# ------------------------------------------------------------ blips
@pytest.mark.parametrize("z, expected", [
    ([0, 0, 5, 0, 0], [0, 0, 1, 0, 0]),                     # one look
    ([0, 0, -5, 0, 0], [0, 0, 1, 0, 0]),                    # downwards
    ([0, 5, 5.5, 0, 0], [0, 1, 1, 0, 0]),                   # two looks
    ([0, 5, 5.5, 4.8, 0, 0], [0, 1, 1, 1, 0, 0]),           # three looks
    ([0, 5, 5, 5, 5, 0, 0], [0, 0, 0, 0, 0, 0, 0]),         # four looks: too long to be called a blip
    ([0, 0, 5, 5, 5], [0, 0, 0, 0, 0]),                     # a jump that stays is not a blip
    ([0, 0, 2.9, 0, 0], [0, 0, 0, 0, 0]),                   # too small
    ([0, 0, 3.0, 0, 0], [0, 0, 0, 0, 0]),                   # exactly 3 is not "more than 3"
    ([0, 0, 3.01, 0, 0], [0, 0, 1, 0, 0]),
    ([0, 0, 3.0, -0.5, -0.5], [0, 0, 0, 0, 0]),             # up by exactly 3, back by 3.5
    ([0, 0, 3.5, 0.5, 0.5], [0, 0, 0, 0, 0]),               # up by 3.5, back by exactly 3
    ([0, 0, 5, 2.5, 0], [0, 0, 0, 0, 0]),                   # comes back slowly
    ([0, 4, 8, 4, 0], [0, 0, 1, 0, 0]),                     # the peak of a staircase jumps both ways
    ([0, 5, 0, 5, 0], [0, 1, 0, 1, 0]),                     # two separate one-look blips
    ([5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),               # the first look cannot be judged
    ([0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 0]),               # nor the last
    ([0, 5, 0, 0, 5, 0], [0, 1, 0, 0, 1, 0]),
    ([0, 5, 6, 0, -5, 0], [0, 1, 1, 0, 1, 0]),              # measured from the last look outside a blip
    ([0, 5], [0, 0]),
    ([], []),
])
def test_blip_mask_by_hand(z, expected):
    assert P.blip_mask(np.array(z, dtype=float)).tolist() == [bool(v) for v in expected]


def test_blip_mask_threshold_is_strict_and_rows_are_independent():
    z = np.array([[0, 3.0, 0, 0], [0, 3.01, 0, 0], [0, 0, 0, 0]])
    assert P.blip_mask(z).tolist() == [[False] * 4, [False, True, False, False], [False] * 4]


def test_episodes_and_confirmed():
    assert P.episodes([False, True, True, False, True]) == 2
    assert P.episodes([True, False]) == 1
    assert P.episodes([]) == 0
    assert P.confirmed([True, True, False, True, True, True]).tolist() == [False, True, False, False, True, True]
    assert P.confirmed(np.array([[True, False, True], [True, True, True]])).tolist() == [
        [False, False, False], [False, True, True]]


# ------------------------------------------------------------ deciding one test
def looks_frame(days, p, ratio, lift=None, z=None, blip=None):
    k = len(days)
    return pd.DataFrame({
        "day": days, "users_c": 1000, "users_t": 1000,
        "lift": lift if lift is not None else [0.01] * k, "lift_se": 0.005,
        "z": z if z is not None else [0.0] * k, "p": p, "ratio": ratio,
        "blip": blip if blip is not None else [False] * k})


def test_decision_looks_boundary_and_last_look():
    assert P.decision_looks([6.0, 6.5, 7.0, 8.0]).tolist() == [False, False, True, True]
    assert P.decision_looks([3.0, 4.0, 5.0]).tolist() == [False, False, True]     # the end is always a decision
    assert P.decision_looks([7.0, 14.0, 15.0], first_day=14).tolist() == [False, True, True]
    assert P.decision_looks([]).tolist() == []


def test_decide_by_hand():
    s = looks_frame(days=[5, 6, 7, 8, 9, 10, 11],
                    p=[0.01, 0.20, 0.20, 0.04, 0.20, 0.03, 0.02],
                    ratio=[30, 1, 1, 25, 1, 21, 40],
                    lift=[0.09, 0.01, 0.01, 0.05, 0.01, 0.03, 0.02],
                    blip=[False, False, False, True, False, False, False])
    d = P.decide(s)
    assert (d["looks"], d["decision_looks"], d["last_day"]) == (7, 5, 11)
    # day 5 is significant but before day 7; the first p < 0.05 that counts is day 8
    assert (d["peek_stops"], d["peek_day"], d["peek_lift"], d["peek_on_blip"]) == (True, 8, 0.05, True)
    # two in a row: days 10 and 11
    assert (d["peek_confirmed_stops"], d["peek_confirmed_day"], d["peek_confirmed_on_blip"]) == (True, 11, False)
    assert (d["always_stops"], d["always_day"]) == (True, 8)
    assert (d["always_confirmed_stops"], d["always_confirmed_day"], d["always_confirmed_lift"]) == (True, 11, 0.02)
    assert d["sig_end"] and d["p_end"] == 0.02 and d["lift_end"] == 0.02
    assert (d["blips"], d["blip_looks"]) == (1, 1)


def test_decide_when_nothing_is_ever_significant():
    d = P.decide(looks_frame(days=[7, 8, 9], p=[0.5, 0.06, 0.05], ratio=[1, 19.99, 19.99]))
    assert not any(d[f"{r}_stops"] for r in P.RULES)        # p = 0.05 is not below 0.05
    assert all(math.isnan(d[f"{r}_day"]) for r in P.RULES)
    assert not d["sig_end"]


def test_decide_ratio_of_exactly_twenty_counts():
    d = P.decide(looks_frame(days=[7, 8], p=[0.5, 0.5], ratio=[20.0, 20.0]))
    assert d["always_stops"] and d["always_day"] == 7 and d["always_confirmed_day"] == 8


def test_decide_confirmation_cannot_use_a_look_before_the_first_decision_day_alone():
    # significant on days 6 and 7: day 7 is confirmed by day 6, and day 7 is a decision look
    d = P.decide(looks_frame(days=[6, 7, 8], p=[0.01, 0.01, 0.5], ratio=[1, 1, 1]))
    assert d["peek_confirmed_day"] == 7
    # significant on day 6 only: nothing at a decision look
    d = P.decide(looks_frame(days=[6, 7, 8], p=[0.01, 0.5, 0.5], ratio=[1, 1, 1]))
    assert not d["peek_stops"]


def test_decide_with_a_longer_minimum_run():
    s = looks_frame(days=[7, 8, 14, 15], p=[0.01, 0.01, 0.5, 0.5], ratio=[1, 1, 1, 1])
    assert P.decide(s)["peek_day"] == 7
    assert not P.decide(s, first_day=14)["peek_stops"]
    # a test shorter than the minimum is still decided at its end
    s = looks_frame(days=[7, 8, 9], p=[0.01, 0.5, 0.01], ratio=[1, 1, 1])
    assert P.decide(s, first_day=28)["peek_day"] == 9


def test_end_jump_ignores_a_blip_just_before_the_end():
    calm = looks_frame(days=[7, 8, 9, 10], p=[0.5] * 4, ratio=[1] * 4, z=[0.1, 0.2, 5.0, 0.3],
                       blip=[False, False, True, False])
    assert not P.decide(calm)["end_jump"]                   # 0.3 against 0.2, the last look outside the blip
    jump = looks_frame(days=[7, 8, 9, 10], p=[0.5] * 4, ratio=[1] * 4, z=[0.1, 0.2, 0.3, 3.4])
    assert P.decide(jump)["end_jump"]
    assert not P.decide(looks_frame(days=[7], p=[0.5], ratio=[1]))["end_jump"]


def toy_looks():
    rows = [  # day, users_c, users_t, mean_c, mean_t, variance_c, variance_t
        (1.0, 1000, 1000, 2.0, 2.1, 4.0, 4.0),
        (2.0, 2000, 2000, 2.0, 2.2, 4.0, 4.0),
        (3.0, 3000, 3000, 2.0, 2.1, 4.0, None),      # no variance
        (4.0, 4000, 4000, 2.0, 2.1, 4.0, 0.0),       # every treatment user has the same value
        (5.0, 1, 5000, 2.0, 2.1, 4.0, 4.0),          # an empty group
        (6.0, 6000, 6000, 0.0, 2.1, 4.0, 4.0),       # zero control mean
        (7.0, 7000, 7000, 2.0, 2.1, 4.0, 4.0),
    ]
    return pd.DataFrame(rows, columns=["day", "users_c", "users_t", "mean_c", "mean_t",
                                       "variance_c", "variance_t"])


def test_look_stats_uses_the_part_two_test_and_drops_untestable_looks():
    s = P.look_stats(toy_looks().sample(frac=1, random_state=0))      # order must not matter
    assert s["day"].tolist() == [1.0, 2.0, 7.0]
    first = s.iloc[0]
    assert first.p == pytest.approx(R.welch(2.0, 2.1, 4.0, 4.0, 1000, 1000)[3])
    lift, se, _, _ = R.relative_lift(2.0, 2.1, 4.0, 4.0, 1000, 1000)
    assert (first.lift, first.lift_se, first.z) == pytest.approx((lift, se, lift / se))
    assert first.ratio == pytest.approx(P.mixture_ratio(lift, se))
    assert not s["blip"].any()


def test_replayed_arms_drops_only_the_two_coverage_cautions():
    arm_checks = pd.DataFrame({
        "experiment_id": ["a", "b", "c", "d", "e"], "variant_id": [1, 1, 1, 1, 1],
        "cautions": ["", "gap in snapshots", "split drifts; first week not recorded",
                     "sample-ratio mismatch", "unequal split"]})
    table = pd.DataFrame({
        "experiment_id": ["a", "a", "b", "c", "d", "e"], "variant_id": 1,
        "reading": ["read", "not testable", "read", "read with warning", "read with warning", "not decided"]})
    assert P.replayed_arms(arm_checks, table)["experiment_id"].tolist() == ["a", "d"]


def test_daily_looks_sql_on_a_toy_table():
    from abtest import load
    rows = []
    #        time, count_c, count_t
    steps = [(0.5, 100, 100), (1.0, 200, 200),      # day 1: the later snapshot is the look
             (1.5, 300, 300),                       # day 2
             (3.0, 250, 400),                       # control count below an earlier one: not a look
             (3.5, 280, 450), (4.0, 290, 460),      # still below the 300 seen at 1.5: not looks either
             (5.0, 500, 500), (5.5, 600, 600)]      # day 5 and day 6
    for t, c, u in steps:
        for metric in (1, 2):
            rows.append(("e", 1, metric, t, float(c), float(u), 1.0 * metric, 1.1, 2.0, 2.0))
    raw = pd.DataFrame(rows, columns=load.COLUMNS)
    looks = load.run_sql(load.connect(raw), "daily_looks")
    for metric in (1, 2):
        g = looks[looks.metric_id == metric]
        assert g.day.tolist() == [1.0, 1.5, 5.0, 5.5]
        assert g.look_day.tolist() == [1, 2, 5, 6]
        assert g.users_c.tolist() == [200, 300, 500, 600]
        assert (g.mean_c == 1.0 * metric).all()


# ------------------------------------------------------------ the simulation
def test_null_z_is_standard_normal_with_random_walk_correlation():
    t = np.array([0.1, 0.4, 1.0])
    z = P.null_z(t, 200_000, np.random.default_rng(1))
    assert z.shape == (200_000, 3)
    assert z.mean(axis=0) == pytest.approx([0, 0, 0], abs=0.01)
    assert z.std(axis=0) == pytest.approx([1, 1, 1], abs=0.01)
    corr = np.corrcoef(z.T)
    assert corr[0, 1] == pytest.approx(math.sqrt(0.1 / 0.4), abs=0.01)
    assert corr[0, 2] == pytest.approx(math.sqrt(0.1 / 1.0), abs=0.01)
    assert corr[1, 2] == pytest.approx(math.sqrt(0.4 / 1.0), abs=0.01)
    assert (P.null_z(t, 10, np.random.default_rng(5)) == P.null_z(t, 10, np.random.default_rng(5))).all()
    for bad in ([], [0.0, 1.0], [0.5, 0.4]):
        with pytest.raises(ValueError):
            P.null_z(bad, 10, np.random.default_rng(1))


def test_a_frozen_look_repeats_the_previous_one():
    z = P.null_z([0.5, 0.5, 1.0], 1000, np.random.default_rng(2))
    assert (z[:, 0] == z[:, 1]).all()


def test_two_looks_agree_with_the_exact_bivariate_normal_answer():
    # P(|Z1| > c or |Z2| > c) with corr(Z1, Z2) = sqrt(1/2)
    c, rho = stats.norm.isf(0.025), math.sqrt(0.5)
    mvn = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    inside = mvn.cdf([c, c]) - mvn.cdf([c, -c]) - mvn.cdf([-c, c]) + mvn.cdf([-c, -c])
    assert P.equal_looks_rate(2) == pytest.approx(1 - inside, abs=0.003)
    assert P.equal_looks_rate(1) == pytest.approx(0.05, abs=0.002)


@pytest.mark.parametrize("looks, published", [(5, 0.142), (10, 0.193), (20, 0.248)])
def test_equal_looks_agree_with_the_published_table(looks, published):
    # Armitage, McPherson and Rowe (1969), repeated significance tests at the 5% level
    assert P.equal_looks_rate(looks) == pytest.approx(published, abs=0.003)


def test_null_rates_order_and_the_always_valid_guarantee():
    t = np.arange(1, 201) / 200                             # 200 looks, all of them decisions
    z = P.null_z(t, 20_000, np.random.default_rng(3))
    ks = {1: 0.5, 2: 2.0, 3: 5.0, 4: 20.0}                  # tau from half to 20 final standard errors
    o = P.null_rates(z, t, np.ones(200, dtype=bool), ks)
    assert o["end"] == pytest.approx(0.05, abs=0.005)
    assert o["peek"] > 0.35                                 # 200 looks: far above 5%
    assert o["peek"] > o["peek_confirmed"] > o["end"]
    for m in ks:
        assert o["always_confirmed"][m] <= o["always"][m] <= P.ALPHA
    assert o["looks_simulated"] == 20_000 * 200


def test_null_rates_respect_the_decision_looks():
    t = np.arange(1, 11) / 10
    z = P.null_z(t, 20_000, np.random.default_rng(4))
    last_only = np.array([False] * 9 + [True])
    o = P.null_rates(z, t, last_only, {1: 3.0})
    assert o["peek"] == o["end"]
    # confirmed at the last look needs looks 9 and 10; check against a direct count
    c = stats.norm.isf(0.025)
    assert o["peek_confirmed"] == pytest.approx(((np.abs(z[:, 8]) > c) & (np.abs(z[:, 9]) > c)).mean())
    everything = P.null_rates(z, t, np.ones(10, dtype=bool), {1: 3.0})
    assert everything["peek"] > o["peek"] and everything["always"][1] >= o["always"][1]


def test_the_always_valid_rule_finds_a_real_change():
    t = np.arange(1, 51) / 50
    z = P.null_z(t, 5_000, np.random.default_rng(6)) + 5.0 * np.sqrt(t)      # true change: 5 final standard errors
    o = P.null_rates(z, t, np.ones(50, dtype=bool), {1: 3.0})
    assert o["always"][1] > 0.85 and o["always_confirmed"][1] > 0.80


def test_simulate_null_is_seeded():
    t, d = [0.2, 0.6, 1.0], [False, True, True]
    assert P.simulate_null(t, d, {1: 2.0}, paths=2000) == P.simulate_null(t, d, {1: 2.0}, paths=2000)
    assert P.simulate_null(t, d, {1: 2.0}, paths=2000, seed=1) != P.simulate_null(t, d, {1: 2.0}, paths=2000, seed=2)


# ------------------------------------------------------------ the committed data
@pytest.fixture(scope="module")
def b():
    return Q.build()


@pytest.fixture(scope="module")
def f(b):
    return Q.facts(b)


def test_daily_looks_sql_agrees_with_pandas(b):
    from abtest import load
    raw = load.read_raw()
    looks = load.run_sql(load.connect(raw), "daily_looks")
    m1 = raw[raw.metric_id == 1].sort_values(["experiment_id", "variant_id", "time_since_start"])
    expected = set()
    for (e, v), g in m1.groupby(["experiment_id", "variant_id"]):
        best_c = best_t = 0.0
        day_look = {}
        for r in g.itertuples(index=False):
            if r.count_c >= best_c and r.count_t >= best_t:
                day_look[math.ceil(r.time_since_start)] = r.time_since_start     # later snapshots overwrite
            best_c, best_t = max(best_c, r.count_c), max(best_t, r.count_t)
        expected |= {(e, v, t) for t in day_look.values()}
    got = looks[looks.metric_id == 1]
    assert set(zip(got.experiment_id, got.variant_id, got.day)) == expected
    assert len(got) == len(expected)
    # every look carries the source row unchanged
    merged = looks.merge(raw, left_on=["experiment_id", "variant_id", "metric_id", "day"],
                         right_on=["experiment_id", "variant_id", "metric_id", "time_since_start"])
    assert len(merged) == len(looks)
    assert (merged.mean_c_x == merged.mean_c_y).all() and (merged.users_t == merged.count_t).all()
    assert not looks.duplicated(["experiment_id", "variant_id", "metric_id", "look_day"]).any()


def test_last_look_is_the_part_two_reading(b):
    res = b["res"].merge(b["table"], on=["experiment_id", "variant_id", "metric_id"], suffixes=("", "_p2"))
    assert len(res) == len(b["res"])
    assert (res.last_day == res.day).all()
    assert res.p_end.to_numpy() == pytest.approx(res.p.to_numpy(), rel=1e-12)
    assert res.lift_end.to_numpy() == pytest.approx(res.lift.to_numpy(), rel=1e-12)
    assert res.lift_se_end.to_numpy() == pytest.approx(res.lift_se.to_numpy(), rel=1e-12)
    assert (res.sig_end == (res.p < R.ALPHA)).all()


def test_what_is_replayed(b, f):
    res, arms = b["res"], b["arms"].set_index(["experiment_id", "variant_id"])
    cautions = arms.loc[list(zip(res.experiment_id, res.variant_id)), "cautions"]
    assert not cautions.str.contains("first week not recorded|gap in snapshots|under a week|unequal split").any()
    assert set(res.reading) <= {"read", "read with warning"}
    assert f["arms"] + f["arms_skipped"] == f["arms_part2"]
    assert f["arms"] == res.groupby(["experiment_id", "variant_id"]).ngroups
    assert f["tests"] == len(res) == len(b["paths"])
    assert f["looks_total"] == sum(len(s) for s in b["paths"].values())
    assert (res.last_day >= P.FIRST_DECISION_DAY).all()
    # every look of every metric is one of the arm's metric-1 looks (the simulation relies on it)
    for (e, v, m), s in b["paths"].items():
        assert set(s.day) <= set(b["paths"][(e, v, 1)].day)


def test_rules_are_nested_on_the_real_tests(b, f):
    res = b["res"]
    assert (res.peek_stops >= res.sig_end).all()                    # the last look is one of peeking's looks
    assert (res.peek_stops >= res.peek_confirmed_stops).all()
    assert (res.always_stops >= res.always_confirmed_stops).all()
    assert (res.peek_stops >= res.always_stops).all()               # the always-valid bar is the higher one
    assert f["bound_min"] > f["z_crit"]
    both = res[res.always_stops & res.peek_stops]
    assert (both.peek_day <= both.always_day).all()
    conf = res[res.always_confirmed_stops]
    assert (conf.always_day <= conf.always_confirmed_day).all()
    for rule in P.RULES:
        r = f["rules"][rule]
        assert r["stops"] == r["sig_end"] + r["not_sig_end"] == int(res[f"{rule}_stops"].sum())
        assert r["on_blip"] <= r["stops"] and r["on_blip_not_sig_end"] <= r["not_sig_end"]
    assert f["missed"] == f["sig_end"] - f["rules"]["always_confirmed"]["sig_end"]
    assert f["quiet_peek"] == f["rules"]["peek"]["not_sig_end"]
    assert f["sig_end"] + f["quiet_end"] == f["tests"]


def test_replay_agrees_with_a_direct_reading_of_the_looks(b):
    # Recompute every rule's verdict from the look tables without `decide`.
    for (e, v, m), s in list(b["paths"].items())[::7]:
        row = b["res"][(b["res"].experiment_id == e) & (b["res"].variant_id == v) & (b["res"].metric_id == m)].iloc[0]
        ok = (s.day >= 7).to_numpy()
        ok[-1] = True
        sig, strong = (s.p < 0.05).to_numpy(), (s.ratio >= 20).to_numpy()
        assert row.peek_stops == (sig & ok).any()
        assert row.always_stops == (strong & ok).any()
        two = [j for j in range(1, len(s)) if strong[j] and strong[j - 1] and ok[j]]
        assert row.always_confirmed_stops == bool(two)
        if two:
            assert row.always_confirmed_day == s.day.iloc[two[0]]
        # Welch's test and the z of the relative change agree once the change is small
        near = s[s.lift.abs() < 0.02]
        assert near.p.to_numpy() == pytest.approx(2 * stats.norm.sf(np.abs(near.z)), abs=0.01)


def test_simulation_on_the_real_schedules(b, f):
    res, na = b["res"], b["null_arms"]
    assert len(na) == f["arms"]
    assert na["end"].to_numpy() == pytest.approx(0.05, abs=0.006)
    assert (na.peek >= na.peek_confirmed).all() and (na.peek >= na.end - 1e-12).all()
    assert (res.null_always <= P.ALPHA).all() and (res.null_always_confirmed <= res.null_always).all()
    assert f["rules"]["peek"]["expected_null"] == pytest.approx(res.null_peek.sum())
    assert f["rules"]["peek"]["null_mean"] > 3 * P.ALPHA
    assert f["sim_looks"] == P.PATHS * int(na.looks.sum())
    assert f["sim_blip_looks"] / f["sim_looks"] < 1e-6 < 1e-3 < f["blip_looks"] / f["looks_total"]
    # arm table and test table hold the same rates
    one = res[res.metric_id == 1].merge(na, on=["experiment_id", "variant_id"])
    assert (one.null_peek == one.peek).all() and (one.null_always == one.always_m1).all()


def test_blip_counts(b, f):
    res = b["res"]
    assert f["blips"] == sum(f["blips_by_metric"].values()) == int(res.blips.sum())
    assert f["blip_looks"] >= f["blips"] >= f["blips_one_look"] > 0
    assert f["blip_looks"] == sum(int(s.blip.sum()) for s in b["paths"].values())
    assert f["blips_by_metric"][2] > 0.8 * f["blips"]           # "mostly one metric"
    assert f["rules"]["always_confirmed"]["on_blip"] == f["fading_on_blip"]
    for (e, v, m), s in b["paths"].items():                     # a confirmed call is never inside a one-look blip
        row = res[(res.experiment_id == e) & (res.variant_id == v) & (res.metric_id == m)].iloc[0]
        if row.always_confirmed_on_blip:
            j = s.index[s.day == row.always_confirmed_day][0]
            assert s.blip.iloc[j - 1] and s.blip.iloc[j]


def test_statements_that_name_a_test(b, f):
    jump = f["end_jumps"]
    assert len(jump) == f["end_jump"] == 1 and f["end_jump_sig"] == 1
    r = jump.iloc[0]
    assert (r.experiment_id, r.variant_id, r.metric_id) == ("adea31", 2, 4)
    s = b["paths"][("adea31", 2, 4)]
    assert abs(s.z.iloc[-1] - s.z.iloc[-2]) > P.BLIP_Z and not s.blip.iloc[-2]
    ex = f["examples"]
    assert ex["holds"].sig_end and ex["holds"].always_confirmed_stops and ex["holds"].blips == 0
    assert not ex["fades"].sig_end and ex["fades"].always_confirmed_stops and ex["fades"].blips == 0
    assert not ex["glitch"].sig_end and ex["glitch"].always_on_blip
    report = (Q.RESULTS / "PEEKING.md").read_text()
    for name in ("holds", "fades", "glitch"):
        assert f"`{ex[name].experiment_id}` / {ex[name].variant_id}, metric {ex[name].metric_id}" in report
    fading = f["fading"]
    assert len(fading) == f["rules"]["always_confirmed"]["not_sig_end"]
    assert (fading.always_confirmed_lift.abs() > fading.lift_end.abs()).all()       # "the change then shrank"


def test_statements_made_in_words(b, f):
    r = f["rules"]
    # "Waiting for a second look in a row helps a little"
    assert 0.6 * r["peek"]["null_mean"] < r["peek_confirmed"]["null_mean"] < r["peek"]["null_mean"]
    # "stays under 5%", "stricter than it needs to be"
    assert r["always"]["null_max"] < P.ALPHA / 2
    # "A result called early is overstated"
    assert r["peek"]["overstatement"] > 1.2 and r["always_confirmed"]["overstatement"] > 1.2
    assert r["peek"]["larger_at_stop"] > 0.7 * r["peek"]["sig_end"]
    # "By arm it is worse"
    assert f["arms_any_peek"] > f["arms_any_end"]
    # "It decides early"
    assert r["always_confirmed"]["days_saved_median"] > 14
    # "they are not all borderline"
    assert f["missed_holm"] > 0 and f["missed_p_median"] > 0.001
    # far more results that do not last than chance alone would give
    assert r["always_confirmed"]["not_sig_end"] - f["fading_on_blip"] > 2 * r["always_confirmed"]["expected_null"]
    # "A longer minimum run removes some of them"
    mn = f["minimum"]
    assert mn.always_confirmed_not_significant_at_end.is_monotonic_decreasing
    assert mn.always_confirmed_not_significant_at_end.iloc[-1] > 0


def test_minimum_run_table(b, f):
    mn, res = f["minimum"].set_index("first_decision_day"), b["res"]
    assert list(mn.index) == list(Q.MINIMUM_RUNS)
    for rule in P.RULES:
        assert mn.loc[7.0, f"{rule}_stops"] == res[f"{rule}_stops"].sum()
        assert mn.loc[7.0, f"{rule}_not_significant_at_end"] == (res[f"{rule}_stops"] & ~res.sig_end).sum()
        assert mn[f"{rule}_stops"].is_monotonic_decreasing
        assert (mn[f"{rule}_stops"] >= res.sig_end.sum() * (rule == "peek")).all()
    assert (mn.significant_at_end == f["sig_end"]).all() and (mn.tests == f["tests"]).all()


def test_equal_looks_curve(b, f):
    rates = list(f["curve"].values())
    assert list(f["curve"]) == list(Q.EQUAL_LOOKS)
    assert all(a < c for a, c in zip(rates, rates[1:]))
    assert f["curve"][1] == pytest.approx(0.05, abs=0.002)


def test_report_csvs_and_charts_regenerate_byte_for_byte(b, f, tmp_path):
    (tmp_path / "charts").mkdir()
    Q.write_outputs(b, f, results=tmp_path, charts=tmp_path / "charts")
    names = ["PEEKING.md", "peeking.csv", "peeking_null_by_arm.csv", "peeking_equal_looks.csv",
             "peeking_by_minimum_run.csv", "charts/07_false_alarms_by_looks.png",
             "charts/08_three_tests.png", "charts/09_results_by_rule.png"]
    for name in names:
        assert (tmp_path / name).read_bytes() == (Q.RESULTS / name).read_bytes(), name
