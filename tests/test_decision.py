"""Part 4: the decision rules by hand, on toy arms and on the committed data,
and every sentence of the memo that depends on which arm was picked."""
import numpy as np
import pandas as pd
import pytest

from abtest import decision as D
from abtest import peeking as P
from abtest import readout as R
from scripts import decision_report as Q

KEY = D.KEY


# ------------------------------------------------------------ the rules, by hand
def test_worst_case_is_the_bad_end_of_the_interval():
    assert D.worst_case(-0.004, 0.012, 1) == -0.004      # higher is better: the lower end
    assert D.worst_case(-0.004, 0.012, -1) == -0.012     # lower is better: the upper end, sign turned
    with pytest.raises(ValueError):
        D.worst_case(-0.004, 0.012, 0)


@pytest.mark.parametrize("lift, lo, hi, p, p_arms, direction, expected", [
    (0.02, 0.01, 0.03, 0.001, 0.002, 1, "better"),
    (0.02, 0.01, 0.03, 0.001, 0.002, -1, "worse"),
    (-0.02, -0.03, -0.01, 0.001, 0.002, 1, "worse"),
    (-0.02, -0.03, -0.01, 0.001, 0.002, -1, "better"),
    (0.02, 0.001, 0.039, 0.04, 0.08, 1, "unconfirmed"),      # fails the correction across arms
    (-0.02, -0.039, -0.001, 0.04, 0.08, 1, "worse"),         # bad news needs no correction
    (0.002, -0.003, 0.007, 0.40, 0.40, 1, "no change"),
    (0.002, -0.003, 0.007, 0.40, 0.40, -1, "no change"),
    (0.002, -0.011, 0.015, 0.80, 0.80, 1, "too wide"),
    (0.004, -0.002, 0.0101, 0.20, 0.20, 1, "too wide"),      # only the upper end leaves the margin
    (0.004, -0.010, 0.006, 0.20, 0.20, 1, "too wide"),       # an end exactly on the margin is not inside it
    (0.02, 0.0, 0.04, 0.05, 0.05, 1, "too wide"),            # p = 0.05 exactly is not significant
])
def test_primary_state(lift, lo, hi, p, p_arms, direction, expected):
    assert D.primary_state(lift, lo, hi, p, p_arms, direction, margin=0.01, alpha=0.05) == expected


@pytest.mark.parametrize("lift, lo, hi, p, direction, expected", [
    (-0.02, -0.03, -0.01, 0.001, 1, "worse"),
    (-0.02, -0.03, -0.01, 0.001, -1, "cleared"),          # a fall is good news if lower is better
    (0.02, 0.01, 0.03, 0.001, -1, "worse"),
    (-0.004, -0.0079, -0.0001, 0.045, 1, "worse"),        # significant beats "inside the margin"
    (0.001, -0.004, 0.006, 0.70, 1, "cleared"),
    (0.001, -0.004, 0.012, 0.70, 1, "cleared"),           # only the bad end matters
    (0.001, -0.004, 0.012, 0.70, -1, "not cleared"),
    (-0.003, -0.015, 0.009, 0.60, 1, "not cleared"),
    (-0.003, -0.010, 0.004, 0.25, 1, "not cleared"),      # a bad end exactly on the margin is not cleared
    (0.03, -0.02, 0.08, 0.20, 1, "not cleared"),          # a likely gain does not clear a wide interval
])
def test_guard_state(lift, lo, hi, p, direction, expected):
    assert D.guard_state(lift, lo, hi, p, direction, margin=0.01, alpha=0.05) == expected


@pytest.mark.parametrize("reading, primary, guards, jump, expected", [
    ("not decided", "better", ["cleared"] * 3, False, "no decision"),
    ("not decided", None, [], True, "no decision"),
    ("read", None, [], False, "no decision"),                                  # primary not testable
    ("read with warning", "better", ["cleared"] * 3, False, "check the data"),
    ("read with warning", "worse", ["worse"] * 3, False, "check the data"),    # data checks come first
    ("read", "better", ["cleared"] * 3, True, "check the data"),
    ("read", "worse", ["cleared"] * 3, True, "check the data"),
    ("read", "worse", ["cleared"] * 3, False, "do not launch"),
    ("read", "better", ["cleared", "worse", "cleared"], False, "do not launch"),
    ("read", "no change", ["cleared", "cleared", "worse"], False, "do not launch"),
    ("read", "too wide", ["worse", "not cleared", "not testable"], False, "do not launch"),
    ("read", "unconfirmed", ["worse", "cleared", "cleared"], False, "do not launch"),
    ("read", "better", ["cleared"] * 3, False, "launch"),
    ("read", "better", [], False, "launch"),                                   # no guard rails at all
    ("read", "better", ["cleared", "not cleared", "cleared"], False, "run longer"),
    ("read", "better", ["cleared", "not testable", "cleared"], False, "run longer"),
    ("read", "unconfirmed", ["cleared"] * 3, False, "run longer"),
    ("read", "no change", ["cleared"] * 3, False, "no change"),
    ("read", "no change", ["not cleared", "not testable", "cleared"], False, "no change"),
    ("read", "too wide", ["cleared"] * 3, False, "run longer"),
])
def test_verdict_applies_the_rules_in_order(reading, primary, guards, jump, expected):
    v, reason = D.verdict(reading, primary, guards, jump)
    assert v == expected and v in D.VERDICTS and reason


def test_verdict_reasons_tell_the_two_stops_apart():
    assert D.verdict("read", "worse", ["worse"], False)[1] == "primary worse"
    assert D.verdict("read", "no change", ["worse"], False)[1] == "guard rail worse"
    assert D.verdict("read", "better", ["worse"], False)[1] == "guard rail worse although the primary is better"
    assert D.verdict("read", "better", ["not testable", "not cleared"], False)[1].endswith("cannot be tested")


# ------------------------------------------------------------ toy arms
def toy_row(e, v, m, lift, half, p, reading="read", cautions=""):
    return {"experiment_id": e, "variant_id": v, "metric_id": m, "reading": reading, "cautions": cautions,
            "lift": lift, "lift_lo": lift - half if half is not None else np.nan,
            "lift_hi": lift + half if half is not None else np.nan,
            "p": p if half is not None else np.nan, "mde": 0.004}


def toy_table():
    rows = []
    # "win": everything up and tight
    rows += [toy_row("win", 1, m, 0.02, 0.005, 1e-6) for m in (1, 2, 3, 4)]
    # "trade": primary up, metric 3 down
    rows += [toy_row("trade", 1, 1, 0.02, 0.005, 1e-6), toy_row("trade", 1, 2, 0.0, 0.004, 0.9),
             toy_row("trade", 1, 3, -0.015, 0.005, 1e-4), toy_row("trade", 1, 4, 0.0, 0.004, 0.9)]
    # "blind": primary up, metric 4 has no variance; metric 2 too wide in the same arm of another experiment
    rows += [toy_row("blind", 1, 1, 0.02, 0.005, 1e-6), toy_row("blind", 1, 2, 0.0, 0.004, 0.9),
             toy_row("blind", 1, 3, 0.0, 0.004, 0.9), toy_row("blind", 1, 4, 0.01, None, None, "not testable")]
    rows += [toy_row("foggy", 1, 1, 0.02, 0.005, 1e-6), toy_row("foggy", 1, 2, 0.0, 0.03, 0.9),
             toy_row("foggy", 1, 3, 0.0, 0.004, 0.9), toy_row("foggy", 1, 4, 0.0, 0.004, 0.9)]
    # "three": three arms, primary p = 0.03, 0.04 and 0.0001 -> Holm 0.06, 0.06, 0.0003
    for v, p in ((1, 0.03), (2, 0.04), (3, 0.0001)):
        rows += [toy_row("three", v, 1, 0.006, 0.003, p)] + [toy_row("three", v, m, 0.0, 0.004, 0.9) for m in (2, 3, 4)]
    # "quiet", "small", "short", "warned", "jumpy"
    rows += [toy_row("quiet", 1, m, 0.001, 0.004, 0.6) for m in (1, 2, 3, 4)]
    rows += [toy_row("small", 1, m, 0.01, 0.04, 0.6) for m in (1, 2, 3, 4)]
    rows += [toy_row("short", 1, m, 0.02, 0.005, 1e-6, "not decided", "under a week") for m in (1, 2, 3, 4)]
    rows += [toy_row("warned", 1, m, 0.02, 0.005, 1e-6, "read with warning", "split drifts") for m in (1, 2, 3, 4)]
    rows += [toy_row("jumpy", 1, m, 0.02, 0.005, 1e-6) for m in (1, 2, 3, 4)]
    return pd.DataFrame(rows)


def toy_ends(table):
    ends = table[table["reading"].isin(D.READ)][KEY + ["metric_id"]].copy()
    ends["looks"] = 10
    ends["end_jump"] = (ends["experiment_id"] == "jumpy") & (ends["metric_id"] == 3)
    return ends


def test_primary_p_is_corrected_across_the_arms_of_one_experiment():
    t = toy_table()
    p = D.primary_p_across_arms(t)
    assert p[("three", 1)] == pytest.approx(0.06) and p[("three", 2)] == pytest.approx(0.06)
    assert p[("three", 3)] == pytest.approx(0.0003)
    assert p[("win", 1)] == pytest.approx(1e-6)                # a single arm is unchanged
    assert ("short", 1) not in p.index                         # arms that are not read do not count
    assert ("warned", 1) in p.index


def test_decide_arms_on_toy_experiments():
    t = toy_table()
    d = D.decide_arms(t, toy_ends(t)).set_index(KEY)
    expected = {("win", 1): "launch", ("trade", 1): "do not launch", ("blind", 1): "run longer",
                ("foggy", 1): "run longer", ("three", 1): "run longer", ("three", 2): "run longer",
                ("three", 3): "launch", ("quiet", 1): "no change", ("small", 1): "run longer",
                ("short", 1): "no decision", ("warned", 1): "check the data", ("jumpy", 1): "check the data"}
    assert d["verdict"].to_dict() == expected
    assert d.loc[("trade", 1), "guards_worse"] == "3"
    assert d.loc[("blind", 1), "guards_not_testable"] == "4"
    assert d.loc[("foggy", 1), "guards_not_cleared"] == "2"
    assert d.loc[("three", 1), "primary_state"] == "unconfirmed"
    assert d.loc[("jumpy", 1), "end_jump_metrics"] == "3"
    assert pd.isna(d.loc[("short", 1), "primary_state"]) and pd.isna(d.loc[("short", 1), "primary_p"])
    assert d.loc[("short", 1), "reason"].startswith("Part 1") and d.loc[("win", 1), "reason"].startswith("primary better")


def test_turning_the_direction_on_toy_experiments():
    t = toy_table()
    d = D.decide_arms(t, toy_ends(t), direction=-1).set_index(KEY)["verdict"].to_dict()
    assert d[("win", 1)] == "do not launch"            # everything rose, and rising is now bad
    assert d[("trade", 1)] == "do not launch"          # the primary is now the one that got worse
    assert d[("three", 3)] == "do not launch"
    assert d[("three", 1)] == "do not launch"          # p = 0.03 is enough for bad news
    assert d[("quiet", 1)] == "no change" and d[("small", 1)] == "run longer"
    assert d[("short", 1)] == "no decision" and d[("warned", 1)] == "check the data"


def test_another_primary_and_another_margin_on_toy_experiments():
    t = toy_table()
    ends = toy_ends(t)
    by3 = D.decide_arms(t, ends, primary=3).set_index(KEY)
    assert by3.loc[("trade", 1), "verdict"] == "do not launch" and by3.loc[("trade", 1), "reason"] == "primary worse"
    by4 = D.decide_arms(t, ends, primary=4).set_index(KEY)
    assert by4.loc[("blind", 1), "verdict"] == "no decision"       # the primary has no variance
    tight = D.decide_arms(t, ends, margin=0.002).set_index(KEY)
    assert tight.loc[("quiet", 1), "verdict"] == "run longer"      # +/-0.4% no longer fits inside +/-0.2%
    assert tight.loc[("win", 1), "verdict"] == "launch"            # the guard rails' bad end is +1.5%
    loose = D.decide_arms(t, ends, margin=0.06).set_index(KEY)   # "small" runs from -3% to +5%
    assert loose.loc[("small", 1), "verdict"] == "no change" and loose.loc[("foggy", 1), "verdict"] == "launch"


def toy_looks(last_mean_t):
    rows = []
    for day in range(1, 9):
        users = 100_000 * day
        mean_t = last_mean_t if day == 8 else 0.500
        for m in (1,):
            rows.append({"experiment_id": "x", "variant_id": 1, "metric_id": m, "look_day": day, "day": float(day),
                         "users_c": users, "users_t": users, "mean_c": 0.500, "mean_t": mean_t,
                         "variance_c": 0.25, "variance_t": 0.25})
    return pd.DataFrame(rows)


def test_end_checks_marks_a_last_look_that_jumps():
    table = pd.DataFrame([{"experiment_id": "x", "variant_id": 1, "metric_id": 1, "reading": "read"}])
    # at 800,000 users per group the standard error of the difference is sqrt(0.5 / 800,000) = 0.00079
    calm = D.end_checks(toy_looks(0.502), table)      # 2.5 standard errors from the look before
    jump = D.end_checks(toy_looks(0.503), table)      # 3.8
    assert not calm["end_jump"].iloc[0] and jump["end_jump"].iloc[0]
    assert calm["looks"].iloc[0] == 8
    unread = D.end_checks(toy_looks(0.503), table.assign(reading="not decided"))
    assert len(unread) == 0 and list(unread.columns) == KEY + ["metric_id", "looks", "end_jump"]


# ------------------------------------------------------------ the committed data
@pytest.fixture(scope="module")
def b():
    return Q.build()


@pytest.fixture(scope="module")
def f(b):
    return Q.facts(b)


def test_every_arm_gets_exactly_one_verdict(b, f):
    for d in [b["up"], b["down"], *b["per_primary"].values()]:
        assert len(d) == 99 and not d[KEY].duplicated().any() and d["verdict"].isin(D.VERDICTS).all()
    for table in (b["by_primary"], b["by_margin"]):
        assert (table[list(D.VERDICTS)].sum(axis=1) == 99).all()
    assert sum(f["up"].values()) == sum(f["down"].values()) == 99
    assert b["by_primary"].iloc[0][list(D.VERDICTS)].to_dict() == f["up"]
    assert b["by_margin"].set_index("margin").loc[D.MARGIN, list(D.VERDICTS)].to_dict() == f["up"]


def test_launch_and_stop_lists_agree_with_a_direct_reading_of_the_readout(b):
    """The same verdicts worked out another way: straight from Part 2's table,
    one metric per column, without the functions in abtest/decision.py."""
    t = b["table"]
    wide = t.pivot(index=KEY, columns="metric_id", values=["lift", "lift_lo", "lift_hi", "p"])
    reading = t.groupby(KEY)["reading"].agg(lambda s: "not decided" if (s == "not decided").any()
                                            else "warned" if (s == "read with warning").any() else "read")
    jumpy = set(map(tuple, b["ends"][b["ends"]["end_jump"]][KEY].to_numpy()))
    ok = (reading == "read") & ~pd.Series([k in jumpy for k in wide.index], index=wide.index)
    arms_read = (reading != "not decided").groupby(level="experiment_id").sum()
    for direction, d in ((1, b["up"]), (-1, b["down"])):
        gain = direction * wide["lift"]
        low = wide["lift_lo"] if direction == 1 else -wide["lift_hi"]
        sig = wide["p"] < 0.05
        any_worse = (sig & (gain < 0)).any(axis=1)
        # Holm across k arms: the smallest p times k; with one arm, p itself. Enough for a launch list
        # when checked against the stricter Bonferroni bound and the looser raw p.
        k = arms_read.reindex(wide.index.get_level_values("experiment_id")).to_numpy()
        guards_ok = (low[[2, 3, 4]] > -0.01).all(axis=1)       # NaN (no variance) fails
        sure = ok & sig[1] & (gain[1] > 0) & (wide["p"][1] * k < 0.05) & ~any_worse & guards_ok
        maybe = ok & sig[1] & (gain[1] > 0) & ~any_worse & guards_ok
        launches = set(map(tuple, d[d["verdict"] == "launch"][KEY].to_numpy()))
        assert set(wide.index[sure]) <= launches <= set(wide.index[maybe])
        stops = set(map(tuple, d[d["verdict"] == "do not launch"][KEY].to_numpy()))
        assert stops == set(wide.index[ok & any_worse])


def test_the_two_directions_mirror_each_other(b, f):
    up, down = b["up"].set_index(KEY), b["down"].set_index(KEY)
    fixed = up["verdict"].isin(["no decision", "check the data"])
    assert (up["verdict"][fixed] == down["verdict"][fixed]).all()
    assert (down["verdict"][up["verdict"] == "launch"] == "do not launch").all()
    assert (up["verdict"][down["verdict"] == "launch"] == "do not launch").all()
    assert f["launch_flips"] == f["up"]["launch"]                       # "All 13 launches become ..."
    # the verdicts that are the same either way: no metric significant, or not decided on the metrics
    same = up["verdict"] == down["verdict"]
    assert f["same_both_directions"] == int(same.sum())
    t = b["table"][b["table"]["reading"].isin(D.READ)]
    any_sig = (t["p"] < D.ALPHA).groupby([t["experiment_id"], t["variant_id"]]).any()
    quiet = same & ~fixed
    assert not any_sig.reindex(up.index[quiet]).any()


def test_end_checks_cover_every_test_and_agree_with_part_three(b, f):
    ends, replay = b["ends"], b["replay"]
    assert len(ends) == 325 and f["ends"]["new_tests"] == 325 - len(replay) == 60
    m = ends.merge(replay[KEY + ["metric_id", "end_jump", "looks"]], on=KEY + ["metric_id"], suffixes=("", "_p3"))
    assert len(m) == len(replay) and (m["end_jump"] == m["end_jump_p3"]).all() and (m["looks"] == m["looks_p3"]).all()
    jr = f["ends"]["jump_rows"]
    assert f["ends"]["jumps"] == 1 and (jr.iloc[0].experiment_id, jr.iloc[0].variant_id, jr.iloc[0].metric_id) == ("adea31", 2, 4)


def test_funnel_and_guard_rail_statements(b, f):
    fn, g, up = f["funnel"], f["guards"], b["up"]
    assert fn["naive"] == fn["no_decision"] + fn["check"] + fn["unconfirmed"] + fn["guard"] + fn["launch"]
    assert fn["launch"] == f["up"]["launch"]
    # "They never stopped a launch ... no guard rail is worse and every one is cleared"
    assert g["better_with_worse"] == 0 and g["better_not_cleared"] == 0 and fn["guard"] == 0
    better = up[up["primary_state"] == "better"]
    assert (better["guards_not_testable"] == "").all()
    assert set(better["verdict"]) == {"launch", "check the data"}
    # "a primary that is not significantly different and a guard rail that is significantly worse"
    gw = up[up["reason"] == "guard rail worse"]
    assert (gw["primary_p"] >= D.ALPHA).all() and set(gw["primary_state"]) <= {"no change", "too wide"}
    assert g["flat_with_worse"] + g["primary_worse"] == f["up"]["do not launch"]
    assert g["flat_with_worse_down"] + g["primary_worse_down"] == f["down"]["do not launch"]
    assert f["any_metric_up"] > fn["naive"] > fn["launch"]


def test_launches_against_the_always_valid_rule(b, f):
    lc, dec = f["launch_check"], f["decisions"]
    la = dec[dec["verdict_if_higher_better"] == "launch"]
    assert lc["confirmed"] + lc["not_confirmed"] == lc["replayed"] <= lc["launches"] == len(la)
    assert la[~la["replayed"]]["always_valid_confirmed"].isna().all()
    # "their daily record is incomplete"
    assert la[~la["replayed"]]["cautions"].str.contains("first week not recorded|gap in snapshots").all()
    assert (la["primary_p_arms"] < D.ALPHA).all() and (la["primary_lift"] > 0).all()
    assert lc["under_margin"] == int((la["primary_lift"] < 0.01).sum())


def test_sensitivity_statements(b, f):
    bp = f["by_primary"]
    assert bp["do not launch"].nunique() == 1                   # "The 'do not launch' count does not move"
    assert f["launch_every_primary"] <= min(bp["launch"]) and f["launch_any_primary"] >= max(bp["launch"])
    assert f["same_any_primary_decided"] <= f["decided_arms"] == 99 - f["up"]["no decision"] - f["up"]["check the data"]
    missing = b["table"][b["table"]["reading"] == "not testable"].groupby(KEY).ngroups
    assert bp["no decision"].iloc[1] - bp["no decision"].iloc[0] == missing == 5
    assert (bp["no decision"].iloc[1:] == bp["no decision"].iloc[1]).all()
    bm = f["by_margin"].set_index("margin")
    # the margin only moves arms between "no change" and "run longer" (one launch aside)
    for col in ("do not launch", "check the data", "no decision"):
        assert bm[col].nunique() == 1
    assert bm["launch"].max() - bm["launch"].min() <= 1
    assert bm["no change"].is_monotonic_increasing and bm["run longer"].is_monotonic_decreasing


def test_the_cases_are_what_the_memo_says_they_are(b, f):
    c, up = f["cases"], b["up"]
    assert [c[k]["row"].verdict for k in Q.CASES] == ["launch", "launch", "do not launch", "check the data",
                                                      "no change", "run longer", "check the data"]
    assert len({(c[k]["row"].experiment_id, c[k]["row"].variant_id) for k in Q.CASES}) == 7

    clear = c["clear"]                      # "all three guard rails moved the same way. The split is clean"
    assert clear["arm"].cautions == "" and (clear["metrics"]["lift"] > 0).all()
    assert clear["replayed"] and clear["av"].always_confirmed_stops and not clear["av"].always_confirmed_on_blip
    assert clear["row"].primary_p == up[up["verdict"] == "launch"]["primary_p"].min()

    thin = c["thin"]                        # "The guard rails are cleared. The always-valid rule never called"
    assert thin["replayed"] and not thin["av"].always_stops and not thin["av"].always_confirmed_stops
    assert [thin["states"][m] for m in (2, 3, 4)] == ["cleared"] * 3
    assert thin["row"].primary_p_arms == up[up["verdict"] == "launch"]["primary_p_arms"].max()
    assert thin["metrics"].loc[1, "mde"] > abs(thin["row"].primary_lift)

    guard = c["guard"]                      # "interval inside +/-1%" ... "lowers two other metrics"
    assert guard["row"].primary_state == "no change" and guard["row"].primary_p >= D.ALPHA
    assert sorted(m for m, s in guard["states"].items() if s == "worse") == [3, 4]
    assert guard["down"].verdict == "no change"

    jump = c["jump"]                        # "the second treatment of the experiment in the case before"
    assert jump["row"].experiment_id == guard["row"].experiment_id
    assert (guard["row"].variant_id, jump["row"].variant_id) == (1, 2) and jump["arm"].arms_in_experiment == 2
    assert jump["row"].end_jump_metrics == "4" and jump["row"].primary_state == "no change"
    m4 = jump["metrics"].loc[4]
    assert m4.p < D.ALPHA and m4.lift < 0 and jump["states"][4] == "worse"
    # "If the figure holds, the verdict is 'do not launch'"
    assert D.verdict("read", jump["row"].primary_state, [jump["states"][m] for m in (2, 3, 4)], False)[0] == "do not launch"
    assert jump["blips"] > 0

    flat = c["flat"]                        # "Every guard rail is cleared"
    assert [flat["states"][m] for m in (2, 3, 4)] == ["cleared"] * 3
    assert -D.MARGIN < flat["row"].primary_lo and flat["row"].primary_hi < D.MARGIN

    wide = c["wide"]                        # "a gain of several percent and a loss of more than 1% both fit"
    assert wide["row"].primary_lo < -D.MARGIN and wide["row"].primary_hi > 0.03
    assert wide["row"].primary_mde > wide["largest_significant"]
    assert wide["days_needed"] > wide["longest_test"] and wide["users_needed"] > wide["users_had"]
    assert wide["users_had"] == min(wide["arm"].users_c, wide["arm"].users_t)
    sd = np.sqrt(wide["metrics"].loc[1, "variance_c"]) / wide["metrics"].loc[1, "mean_c"]
    assert wide["users_needed"] == R.users_per_group(sd, 0.01)

    split = c["split"]                      # "smaller than the control group ... drifts from week to week"
    assert {"sample-ratio mismatch", "split drifts"} <= set(split["arm"].cautions.split("; "))
    assert split["users_gap"] == split["arm"].users_c - split["arm"].users_t > 0
    gap_share = split["users_gap"] / split["arm"].users_t
    assert split["row"].primary_lift > 0 and gap_share >= split["row"].primary_lift   # "as much as the whole result"
    assert not split["replayed"]


def test_case_figures_in_the_memo_text(b, f):
    memo = Q.render(b, f)
    c = f["cases"]
    for k in Q.CASES:
        row = c[k]["row"]
        assert f"### {Q.CASES.index(k) + 1}. {Q.CASE_TITLE[k]}: `{row.experiment_id}` / {row.variant_id}" in memo
    pr = c["clear"]["metrics"].loc[1]
    assert f"The primary is up {100 * pr.lift:+.2f}% ({100 * pr.lift_lo:+.2f}% to {100 * pr.lift_hi:+.2f}%)" in memo
    w = c["wide"]
    assert f"about {w['users_needed']:,} users per group" in memo and f"the smaller group had {w['users_had']:,}" in memo
    s = c["split"]
    assert f"{s['users_gap']:,} users smaller than the control group" in memo
    assert "None " not in memo.replace("None is stopped by a guard rail", "")
    assert "nan" not in memo.lower().replace("financ", "")


def test_decisions_csv_matches_the_two_tables(b, f):
    dec = f["decisions"]
    assert len(dec) == 99
    assert (dec["verdict_if_higher_better"].to_numpy() == b["up"]["verdict"].to_numpy()).all()
    assert (dec["verdict_if_lower_better"].to_numpy() == b["down"]["verdict"].to_numpy()).all()
    rp = b["replay"][b["replay"]["metric_id"] == D.PRIMARY]
    assert dec["replayed"].sum() == len(rp) == 67
    assert dec["always_valid_confirmed"].notna().sum() == 67


def test_report_csvs_and_charts_regenerate_byte_for_byte(b, f, tmp_path):
    (tmp_path / "charts").mkdir()
    Q.write_outputs(b, f, results=tmp_path, charts=tmp_path / "charts")
    names = ["DECISION_MEMO.md", "decisions.csv", "decisions_by_primary.csv", "decisions_by_margin.csv",
             "charts/10_verdicts.png", "charts/11_memo_cases.png"]
    for name in names:
        assert (tmp_path / name).read_bytes() == (Q.RESULTS / name).read_bytes(), name
