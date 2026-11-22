"""The checks on small, hand-worked inputs."""
import math

import pandas as pd
import pytest

from abtest import checks as C, load


# ---------------------------------------------------------------- sample ratio
def test_srm_even_counts_raise_no_alarm():
    chi2, p = C.srm_test(500, 500)
    assert chi2 == 0 and p == 1


def test_srm_by_hand():
    # expected 5,000 each; (200^2 / 5000) * 2 = 16
    chi2, p = C.srm_test(5200, 4800)
    assert chi2 == pytest.approx(16)
    assert p == pytest.approx(6.334e-5, rel=1e-3)


def test_srm_is_symmetric():
    assert C.srm_test(5200, 4800) == C.srm_test(4800, 5200)


def test_srm_against_an_unequal_design():
    chi2, p = C.srm_test(1000, 2000, expected_share=2 / 3)
    assert chi2 == pytest.approx(0) and p == pytest.approx(1)
    # the same counts are a glaring mismatch against 50/50
    assert C.srm_test(1000, 2000)[1] < 1e-50


def test_srm_expected_share_is_the_treatment_share():
    # control 3,000, treatment 1,000: fits a 25% treatment share, not 75%
    assert C.srm_test(3000, 1000, 0.25)[0] == pytest.approx(0)
    assert C.srm_test(3000, 1000, 0.75)[0] > 1000


def test_srm_rejects_bad_input():
    with pytest.raises(ValueError):
        C.srm_test(0, 0)
    with pytest.raises(ValueError):
        C.srm_test(10, 10, 1.0)


def test_users_short():
    assert C.users_short(5200, 4800) == 200
    assert C.users_short(4800, 5200) == -200
    assert C.users_short(1000, 2000, 2 / 3) == pytest.approx(0)


def test_split_verdict_bands():
    assert C.split_verdict(0.499, 0.5) == "even"
    assert C.split_verdict(0.499, 1e-9) == "small imbalance"
    assert C.split_verdict(0.4951, 1e-9) == "small imbalance"
    assert C.split_verdict(0.4949, 1e-9) == "unequal"
    assert C.split_verdict(0.667, 0.0) == "unequal"
    # exactly at the alarm level is not an alarm
    assert C.split_verdict(0.499, C.ALPHA) == "even"


def test_fitting_design():
    assert C.fitting_design(1_000_000, 2_000_000) == "1:2"
    assert C.fitting_design(3_400_000, 3_300_000) == "34:33"
    assert C.fitting_design(1_000_000, 1_000_000) == ""
    assert C.fitting_design(2_000_000, 1_000_000) == ""  # the reverse of 1:2 is not on the list


# ---------------------------------------------------------------- new users
def test_new_users_from_cumulative_counts():
    inc = C.new_users([1, 2, 3], [100, 150, 175], [90, 160, 170])
    assert inc["new_c"].tolist() == [100, 50, 25]
    assert inc["new_t"].tolist() == [90, 70, 10]


def test_new_users_sorts_by_day():
    inc = C.new_users([3, 1, 2], [175, 100, 150], [170, 90, 160])
    assert inc["day"].tolist() == [1, 2, 3]
    assert inc["new_c"].tolist() == [100, 50, 25]


def test_new_users_skips_a_falling_snapshot():
    # day 2's treatment count is below day 1's; its users land in day 3
    inc = C.new_users([1, 2, 3], [100, 150, 200], [100, 90, 210])
    assert inc["day"].tolist() == [1, 3]
    assert inc["new_c"].tolist() == [100, 100]
    assert inc["new_t"].tolist() == [100, 110]
    assert inc["new_c"].sum() == 200 and inc["new_t"].sum() == 210


def test_weekly_blocks():
    inc = pd.DataFrame({"day": [0.5, 7.0, 7.5, 14.0, 14.5], "new_c": [1, 2, 4, 8, 16], "new_t": [1, 1, 1, 1, 1]})
    week = C.weekly(inc)
    assert week["week"].tolist() == [1, 2, 3]
    assert week["new_c"].tolist() == [3, 12, 16]
    assert week["new_t"].tolist() == [2, 2, 1]


# ---------------------------------------------------------------- stability
def frame(rows):
    return pd.DataFrame(rows, columns=["day", "new_c", "new_t"])


def test_stability_identical_shares():
    chi2, dof, p = C.stability_test(frame([(1, 5000, 5000), (2, 2000, 2000), (3, 3000, 3000)]))
    assert chi2 == pytest.approx(0) and dof == 2 and p == pytest.approx(1)


def test_stability_by_hand():
    # 2 x 2 table [[6000, 4000], [4000, 6000]]: every expected count is 5,000,
    # every cell is 1,000 off, so chi-square = 4 * 1000^2 / 5000 = 800
    chi2, dof, p = C.stability_test(frame([(1, 6000, 4000), (2, 4000, 6000)]))
    assert chi2 == pytest.approx(800) and dof == 1 and p < 1e-100


def test_stability_ignores_the_level_of_the_split():
    # a constant 1:2 split is stable
    chi2, _, p = C.stability_test(frame([(1, 1000, 2000), (2, 3000, 6000)]))
    assert chi2 == pytest.approx(0) and p == pytest.approx(1)


def test_stability_pools_small_intervals():
    # the three small intervals (600 users each) are pooled into one row
    rows = [(1, 5000, 5000), (2, 300, 300), (3, 300, 300), (4, 300, 300)]
    assert C.stability_test(frame(rows))[1] == 1
    assert C.stability_test(frame(rows), min_new=100)[1] == 3


def test_stability_pooled_interval_still_counts():
    # each small interval is all-treatment; pooled they are big enough to alarm
    rows = [(1, 50_000, 50_000)] + [(d, 0, 900) for d in range(2, 12)]
    assert C.stability_test(frame(rows))[2] < 1e-50


def test_stability_needs_two_intervals():
    chi2, dof, p = C.stability_test(frame([(1, 5000, 5000)]))
    assert math.isnan(chi2) and dof == 0 and math.isnan(p)


def test_stability_verdict():
    nan = float("nan")
    assert C.stability_verdict(0.5, 0.5) == "stable"
    assert C.stability_verdict(1e-9, 0.5) == "timing only"
    assert C.stability_verdict(1e-9, nan) == "timing only"
    assert C.stability_verdict(1e-9, 1e-9) == "drifting"
    assert C.stability_verdict(0.5, 1e-9) == "drifting"
    assert C.stability_verdict(0.5, nan) == "stable"
    assert C.stability_verdict(nan, nan) == "not testable"


def test_weekly_share_range():
    week = pd.DataFrame({"week": [1, 2, 3], "new_c": [6000, 4000, 10], "new_t": [4000, 6000, 90]})
    assert C.weekly_share_range(week) == (0.4, 0.6)       # week 3 is too small to count
    assert C.weekly_share_range(week, min_new=50) == (0.4, 0.9)
    assert all(math.isnan(x) for x in C.weekly_share_range(week, min_new=10**6))


# ---------------------------------------------------------------- SQL on a toy table
def toy() -> pd.DataFrame:
    """Two experiments. `aaa` has one clean arm, days 1 to 10, 1,000 new users
    a day per group. `bbb` has two arms sharing a control, days 9 to 12 with a
    hole after day 9; arm 2 gets twice the control's traffic."""
    rows = []
    for day in range(1, 11):
        for m in (1, 2, 3, 4):
            mean = 0.2 if m == 1 else 3.0
            var = 0.16 if m == 1 else 9.0
            rows.append(("aaa", 1, m, float(day), 1000.0 * day, 1000.0 * day, mean, mean, var, var))
    for day, cum in ((9.0, 9000.0), (12.0, 12000.0)):
        for arm, mult in ((1, 1), (2, 2)):
            for m in (1, 2, 3, 4):
                mean = 0.5 if m == 1 else 3.0
                var = 0.25 if m == 1 else 9.0
                rows.append(("bbb", arm, m, day, cum, cum * mult, mean, mean, var, var))
    return pd.DataFrame(rows, columns=load.COLUMNS)


def violations(df: pd.DataFrame) -> dict:
    out = load.run_sql(load.connect(df), "integrity")
    return dict(zip(out["check_name"], out["violations"]))


def test_integrity_clean_toy_has_no_violations():
    v = violations(toy())
    assert len(v) == 13 and set(v.values()) == {0}


@pytest.mark.parametrize("check,edit", [
    ("duplicate_keys", lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True)),
    ("arms_without_four_metrics", lambda d: d[~((d.experiment_id == "aaa") & (d.metric_id == 4))]),
    ("snapshots_missing_a_metric", lambda d: d.drop(index=3)),
    ("user_counts_differ_by_metric", lambda d: d.assign(count_t=d.count_t.where(d.index != 1, 7.0))),
    ("control_differs_between_arms",
     lambda d: d.assign(count_c=d.count_c.where(~((d.experiment_id == "bbb") & (d.variant_id == 2)), d.count_c + 1))),
    ("non_integer_user_counts", lambda d: d.assign(count_c=d.count_c.where(d.experiment_id != "aaa", d.count_c + 0.5))),
    ("empty_groups", lambda d: d.assign(count_t=d.count_t.where(d.index >= 4, 0.0))),
    ("missing_means", lambda d: d.assign(mean_t=d.mean_t.where(d.index != 5))),
    ("negative_means", lambda d: d.assign(mean_c=d.mean_c.where(d.index != 5, -1.0))),
    ("binary_metric_out_of_range", lambda d: d.assign(mean_t=d.mean_t.where(d.index != 0, 1.2))),
    ("binary_variance_mismatch", lambda d: d.assign(variance_c=d.variance_c.where(d.index != 0, 0.2))),
    ("rows_missing_variance", lambda d: d.assign(variance_t=d.variance_t.where(d.index != 6))),
    ("user_count_falls",
     lambda d: d.assign(count_t=d.count_t.where(~((d.experiment_id == "aaa") & (d.time_since_start == 5)), 100.0))),
])
def test_integrity_each_rule_catches_its_own_fault(check, edit):
    assert violations(edit(toy()))[check] > 0


def test_arm_summary_on_toy():
    arms = load.run_sql(load.connect(toy()), "arm_summary").set_index(["experiment_id", "variant_id"])
    a = arms.loc[("aaa", 1)]
    assert (a.arms_in_experiment, a.snapshots, a.first_day, a.last_day, a.max_gap_days) == (1, 10, 1, 10, 1)
    assert (a.users_c, a.users_t, a.users, a.treatment_share) == (10_000, 10_000, 20_000, 0.5)
    assert (a.frozen_snapshots, a.one_sided_snapshots, a.falling_snapshots) == (0, 0, 0)
    b = arms.loc[("bbb", 2)]
    assert (b.arms_in_experiment, b.snapshots, b.first_day, b.last_day, b.max_gap_days) == (2, 2, 9, 12, 3)
    assert (b.users_c, b.users_t) == (12_000, 24_000)
    assert b.treatment_share == pytest.approx(2 / 3)
    assert arms["metrics_missing_variance"].sum() == 0


def test_arm_summary_counts_frozen_one_sided_and_missing_variance():
    d = toy()
    aaa = d.experiment_id == "aaa"
    d.loc[aaa & (d.time_since_start == 4), ["count_c", "count_t"]] = 3000.0   # same as day 3: frozen
    d.loc[aaa & (d.time_since_start == 7), "count_c"] = 6000.0                # control did not move
    d.loc[aaa & (d.metric_id == 2) & (d.time_since_start == 2), "variance_c"] = None
    d.loc[aaa & (d.metric_id == 3) & (d.time_since_start == 10), "variance_t"] = None
    a = load.run_sql(load.connect(d), "arm_summary").set_index(["experiment_id", "variant_id"]).loc[("aaa", 1)]
    assert a.frozen_snapshots == 1
    assert a.one_sided_snapshots == 1
    assert a.metrics_missing_variance == 2
    assert a.metrics_missing_variance_at_end == 1


def test_metric_profile_on_toy():
    m = load.run_sql(load.connect(toy()), "metric_profile").set_index("metric_id")
    assert m["arms"].tolist() == [3, 3, 3, 3]
    assert (m.loc[1, "min_mean"], m.loc[1, "median_mean"], m.loc[1, "max_mean"]) == (0.2, 0.5, 0.5)
    assert m.loc[1, "median_sd_over_mean"] == pytest.approx(1.0)   # sqrt(0.25) / 0.5
    assert m.loc[2, "median_sd_over_mean"] == pytest.approx(1.0)   # sqrt(9) / 3


def test_check_arms_on_toy():
    d = toy()
    out = C.check_arms(d, load.run_sql(load.connect(d), "arm_summary")).set_index(["experiment_id", "variant_id"])
    assert out.loc[("aaa", 1), "split"] == "even"
    assert out.loc[("aaa", 1), "stability"] == "stable"
    assert out.loc[("aaa", 1), "cautions"] == ""
    assert out.loc[("bbb", 1), "cautions"] == "first week not recorded; gap in snapshots"
    b2 = out.loc[("bbb", 2)]
    assert b2["split"] == "unequal" and b2["fits_design"] == "1:2" and math.isnan(b2["users_short"])
    assert b2["cautions"] == "unequal split; first week not recorded; gap in snapshots"


def test_check_arms_flags_short_tests_and_small_imbalances():
    d = toy()
    d = d[(d.experiment_id == "aaa") & (d.time_since_start <= 6)].copy()
    d["count_c"] *= 1000
    d["count_t"] = d["count_c"] * 0.99          # treatment share 49.75%
    out = C.check_arms(d, load.run_sql(load.connect(d), "arm_summary")).iloc[0]
    assert out["split"] == "small imbalance"
    assert out["users_short"] == pytest.approx((6_000_000 + 5_940_000) / 2 - 5_940_000)
    assert out["stability"] == "stable"
    assert out["cautions"] == "sample-ratio mismatch; under a week"


# ---------------------------------------------------------------- loading
def test_read_raw_snaps_days_to_half_day_steps(tmp_path):
    d = toy()
    d.loc[d.time_since_start == 3, "time_since_start"] = 3.0000000000000004
    path = tmp_path / "t.csv"
    d.to_csv(path, index=False)
    back = load.read_raw(path)
    assert sorted(back.time_since_start.unique())[:4] == [1.0, 2.0, 3.0, 4.0]
    assert back.experiment_id.tolist() == d.experiment_id.tolist()


def test_read_raw_refuses_days_off_the_grid_and_wrong_columns(tmp_path):
    d = toy()
    d.loc[0, "time_since_start"] = 1.3
    d.to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(ValueError):
        load.read_raw(tmp_path / "t.csv")
    toy().rename(columns={"count_c": "users_c"}).to_csv(tmp_path / "u.csv", index=False)
    with pytest.raises(ValueError):
        load.read_raw(tmp_path / "u.csv")


def test_day_seven_boundaries():
    # a test whose only snapshots are on days 7 and 8 is neither short nor late
    d = toy()
    d = d[(d.experiment_id == "aaa") & d.time_since_start.isin([7, 8])]
    out = C.check_arms(d, load.run_sql(load.connect(d), "arm_summary")).iloc[0]
    assert out["cautions"] == ""
    # ending exactly on day 7 is a full week
    d = toy()
    d = d[(d.experiment_id == "aaa") & (d.time_since_start <= 7)]
    assert C.check_arms(d, load.run_sql(load.connect(d), "arm_summary")).iloc[0]["cautions"] == ""
    d = d[d.time_since_start <= 6]
    assert C.check_arms(d, load.run_sql(load.connect(d), "arm_summary")).iloc[0]["cautions"] == "under a week"
