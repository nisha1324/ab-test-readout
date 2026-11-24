"""Every figure the README quotes, tied to the generated results."""
import re
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
    pngs = {p.name for p in (root / "results" / "charts").glob("*.png")}
    linked = set(re.findall(r"\]\(results/charts/([^)]+\.png)\)", README))
    assert linked == pngs                                        # every chart shown, none missing
    assert README.count("](results/charts/") == len(pngs) + 1    # chart 10 is shown twice (summary + Part 4)


def test_status_line():
    assert "**Status: complete.**" in README and "Work in progress" not in README
    assert README.count("~~") == 10     # all five roadmap steps struck out
    assert "5. ~~A one-page summary of the whole project" in README


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


# ------------------------------------------------------------ Part 3
@pytest.fixture(scope="module")
def k():
    from scripts import peeking_report as K
    b3 = K.build()
    return K, b3, K.facts(b3)


def part3_statements(K, f):
    P = K.P
    r = f["rules"]
    pk, av, avc = r["peek"], r["always"], r["always_confirmed"]
    mn = f["minimum"].set_index("first_decision_day")
    return [
        f"replays {f['arms']} of the {f['arms_part2']} arms ({f['tests']} tests)",
        f"the other {f['arms_skipped']} arms have an incomplete daily record",
        f"No rule may stop a test before day {P.FIRST_DECISION_DAY:g}",
        f"**Calling a result at the first daily look with p < {P.ALPHA} gives one in {100 * pk['null_mean']:.1f}%**",
        f"({100 * pk['null_min']:.1f}% to {100 * pk['null_max']:.1f}% depending on the arm)",
        f"{100 * f['curve'][10]:.1f}% for 10 looks and {100 * f['curve'][100]:.1f}% for 100",
        f"is {1 / P.ALPHA:.0f} times the evidence",
        f"Its simulated false-alarm rate here is {100 * av['null_mean']:.1f}%, and {100 * avc['null_mean']:.1f}% when",
        f"**{f['sig_end']} of the {f['tests']} tests are significant at the end. Daily peeking would have called {pk['stops']}**, "
        f"and {pk['not_sig_end']} of those are not significant when the test ends",
        f"peeking would be expected to call about {pk['expected_null']:.0f}",
        f"{f['arms_any_peek']} of {f['arms']} arms show a \"significant\" metric on some day, against {f['arms_any_end']} at the end",
        f"For the {pk['sig_end']} tests that end significant, the change on the day peeking called it was larger than "
        f"the final one in {pk['larger_at_stop']}, by a median factor of {pk['overstatement']:.2f}",
        f"calls {avc['stops']} results:** {avc['sig_end']} of the {f['sig_end']} that end significant and {avc['not_sig_end']} that do not",
        f"For those {avc['sig_end']} it decides on a median of day {avc['stop_day_median']:g} of a test that ran "
        f"{avc['last_day_median']:g} days, {avc['days_saved_median']:g} days sooner",
        f"It misses {f['missed']} tests that end significant",
        f"{f['bound_min']:.2f} standard errors or more at every look in these tests, against {f['z_crit']:.2f} for a single test",
        f"**{f['blips']} blips**",
        f"jumps by more than {P.BLIP_Z:g} standard errors and jumps back within three looks",
        f"{f['blips_by_metric'][2]} of the {f['blips']} are in metric 2",
        f"the simulation has {f['sim_blip_looks']} such looks in {f['sim_looks'] / 1e6:.1f} million, "
        f"the real tests {f['blip_looks']} in {f['looks_total']:,}",
        f"{av['on_blip']} of the {av['stops']} always-valid results were called inside a blip",
        f"{avc['on_blip']} of the {avc['stops']} confirmed results were still called inside a longer one",
        f"**{avc['not_sig_end'] - f['fading_on_blip']} results passed the confirmed rule with no fault marked and are "
        f"not significant at the end**, where chance would give about {avc['expected_null']:.0f} in all {f['tests']} tests",
        f"Here it would have called {pk['stops']} results where {f['sig_end']} held to the end",
        f"It found {avc['sig_end']} of the {f['sig_end']} results that held, a median of {avc['days_saved_median']:g} days early, "
        f"with a simulated false-alarm rate of {100 * avc['null_mean']:.1f}%",
        f"With no decision before day 28 the confirmed rule calls {mn.loc[28.0, 'always_confirmed_stops']} results and "
        f"{mn.loc[28.0, 'always_confirmed_not_significant_at_end']} do not last, against {avc['stops']} and {avc['not_sig_end']} from day 7",
        f"(a median factor of {avc['overstatement']:.2f} for the confirmed rule)",
        f"(more than {P.BLIP_Z:g} standard errors there and back, within {P.MAX_BLIP_LOOKS} looks)",
        f"only run on the {f['tests']} tests replayed",
    ]


def test_every_part3_figure_matches_the_results(k):
    K, _, f = k
    missing = [s for s in part3_statements(K, f) if s not in README]
    assert not missing, missing


def test_part3_statements_in_words(k):
    K, b3, f = k
    jump = f["end_jumps"].iloc[0]
    assert len(f["end_jumps"]) == 1
    assert f"One Part 2 result (`{jump.experiment_id}` / {jump.variant_id}, metric {jump.metric_id})" in README
    assert K.P.MAX_BLIP_LOOKS == 3 and "within three looks" in README
    assert f["rules"]["always_confirmed"]["null_mean"] < f["rules"]["always"]["null_mean"] < K.P.ALPHA
    root = Path(__file__).resolve().parents[1]
    for rel in ("results/PEEKING.md", "results/peeking.csv", "results/charts/07_false_alarms_by_looks.png",
                "results/charts/08_three_tests.png", "results/charts/09_results_by_rule.png",
                "sql/daily_looks.sql", "abtest/peeking.py", "scripts/peeking_report.py"):
        assert rel.split("/")[-1] in README and (root / rel).exists(), rel
    ci = (root / ".github" / "workflows" / "ci.yml").read_text()
    for script in ("data_checks_report.py", "readout_report.py", "peeking_report.py"):
        assert f"python scripts/{script}" in ci and f"python scripts/{script}" in README


@pytest.fixture(scope="module")
def d():
    from scripts import decision_report as Q
    b4 = Q.build()
    return Q, b4, Q.facts(b4)


def part4_statements(Q, f):
    D = Q.D
    up, dn, fn, g, lc, e = f["up"], f["down"], f["funnel"], f["guards"], f["launch_check"], f["ends"]
    c = f["cases"]
    pr = {k: c[k]["metrics"].loc[D.PRIMARY] for k in Q.CASES}
    name = {k: f"`{c[k]['row'].experiment_id}` / {c[k]['row'].variant_id}" for k in Q.CASES}
    split = c["split"]
    return [
        f"(here metric {D.PRIMARY}, a working choice)",
        f"rules out a loss of {100 * D.MARGIN:g}% or more",
        f"**{up['launch']} of the {f['arms']} treatments pass every rule for a launch** if higher is better "
        f"({dn['launch']} if lower is better)",
        f"{up['do not launch']} are \"do not launch\", {up['no change']} \"no change\", {up['run longer']} \"run longer\", "
        f"{up['check the data']} \"check the data\" and {up['no decision']} \"no decision\"",
        f"{fn['naive']} arms end with the primary significantly higher and {fn['launch']} are launches",
        f"would pick {f['any_metric_up']}.",
        f"{g['flat_with_worse']} arms have a primary that is not significantly different and a guard rail that is "
        f"significantly worse. In {g['flat_with_worse_primary_no_change']} of them the primary alone reads \"no change\"",
        f"**{lc['confirmed']} of the {lc['replayed']} launches that have a complete daily record also pass",
        f"The other {lc['not_confirmed']} are launches only if",
        f"{f['same_both_directions']} of the {f['arms']} verdicts are the same either way; all {f['launch_flips']} launches "
        f"become \"do not launch\" if lower is better",
        f"Of the {f['decided_arms']} arms that get a verdict on the metrics, {f['same_any_primary_decided']} get the same one",
        f"{f['launch_any_primary']} arms are a launch under at least one choice and {f['launch_every_primary']} under every choice",
        f"run here on all {e['tests']} tests, finds {e['jumps']}:",
        f"| {name['clear']} | **Launch.** Primary up {100 * pr['clear'].lift:.2f}%",
        f"| {name['thin']} | **Launch only if the end date was planned.** Primary up {100 * pr['thin'].lift:.2f}% "
        f"with p = {pr['thin'].p:.3f}",
        f"| {name['guard']} | **Do not launch.** Primary flat, two guard rails significantly lower",
        f"| {name['jump']} | **Check the data.**",
        f"| {name['flat']} | **No change.** {c['flat']['arm'].users / 1e6:.0f} million users rule out a change of "
        f"{100 * D.MARGIN:g}% either way",
        f"| {name['wide']} | **Cannot tell.** Could only detect a {100 * pr['wide'].mde:.1f}% change; a "
        f"{100 * D.MARGIN:g}% change would need about {c['wide']['days_needed']:.0f} days",
        f"| {name['split']} | **Check the data.** Primary up {100 * pr['split'].lift:.2f}%, but the treatment group is "
        f"{100 * split['users_gap'] / split['arm'].users_t:.2f}% short of users",
        f"the direction changes {f['arms'] - f['same_both_directions']} of {f['arms']}, the choice of primary "
        f"{f['decided_arms'] - f['same_any_primary_decided']} of {f['decided_arms']}",
        f"Half of the {up['do not launch']} \"do not launch\" verdicts",
        f"{up['no change']} tests rule out a change of {100 * D.MARGIN:g}%, which closes the idea. {up['run longer']} cannot tell",
        f"{lc['not_confirmed']} of the launches here depend on which it was",
        f"Part 4's primary (metric {D.PRIMARY}), the {100 * D.MARGIN:g}% margin and the direction are working choices",
        f"uncorrected p < {D.ALPHA}, three per arm",
    ]


def test_every_part4_figure_matches_the_results(d):
    Q, _, f = d
    missing = [s for s in part4_statements(Q, f) if s not in README]
    assert not missing, missing


def test_part4_statements_in_words(d):
    Q, b4, f = d
    g, up, c = f["guards"], f["up"], f["cases"]
    assert 2 * g["flat_with_worse"] == up["do not launch"]                     # "Half of the 20"
    assert g["better_with_worse"] == 0 and f["funnel"]["guard"] == 0           # "never stopped a launch"
    assert sorted(m for m, s in c["guard"]["states"].items() if s == "worse").__len__() == 2
    assert c["guard"]["row"].primary_state == "no change"                      # "Primary flat"
    assert (c["clear"]["metrics"]["lift"].drop(index=Q.D.PRIMARY) > 0).all()   # "guard rails up too"
    assert c["clear"]["arm"].cautions == ""                                    # "clean test"
    assert not c["thin"]["av"].always_confirmed_stops                          # "never confirmed under daily looks"
    assert c["jump"]["row"].end_jump_metrics != ""
    assert len(Q.CASES) == 7 and "seven treatments" in README
    root = Path(__file__).resolve().parents[1]
    for rel in ("results/DECISION_MEMO.md", "results/decisions.csv", "results/charts/10_verdicts.png",
                "results/charts/11_memo_cases.png", "abtest/decision.py", "scripts/decision_report.py"):
        assert rel.split("/")[-1] in README and (root / rel).exists(), rel
    ci = (root / ".github" / "workflows" / "ci.yml").read_text()
    assert "python scripts/decision_report.py" in ci and "python scripts/decision_report.py" in README
    assert "all four reports" in README and "four reports" in ci


# ------------------------------------------------------------ Summary
def section(start, end):
    return README.split(start, 1)[1].split(end, 1)[0]


SUMMARY = section("## Summary", "## The data")
BODY = README.split("## The data", 1)[1]


def summary_statements(f1, f2, f3, f4, D):
    pk, avc = f3["rules"]["peek"], f3["rules"]["always_confirmed"]
    up = f4["up"]
    jump = f3["end_jumps"].iloc[0]
    return [
        f"{f1['experiments']} experiments, {f1['arms']} treatment arms and four metrics per arm",
        f"{f1['arms'] - f1['clean_arms']} of {f1['arms']} arms fail at least one check",
        f"raises {f1['naive_alarms']} alarms, but only {f1['small']} look like users going missing; the other {f1['unequal']}",
        f"{f1['coverage_caution_arms']} arms ended within a week",
        f"{f2['significant']} of {f2['tests']} tests are significant where about {f2['expected_by_chance']:.0f} would be by chance",
        f"the median significant change is {100 * f2['sig_abs_median']:.2f}% of the control mean",
        f"about {round(f2['need'][1], -3):,} users per group for metric 1 and {round(f2['need'][4], -3):,} for metric 4",
        f"Of the {f2['quiet']} quiet tests, {f2['quiet_inside']} rule out a change of 1% either way",
        f"gives a false alarm in {100 * pk['null_mean']:.1f}% of simulated tests",
        f"would have called {pk['stops']} results where {f3['sig_end']} held to the end",
        f"keeps the simulated rate at {100 * avc['null_mean']:.1f}% and found {avc['sig_end']} of the {f3['sig_end']}, "
        f"a median of {avc['days_saved_median']:g} days early",
        f"{f3['blips']} blips, where a result jumps by more than",
        f"{f3['blips_by_metric'][2]} of them are in metric 2",
        f"{up['launch']} of {f4['arms']} treatments are a launch if higher is better",
        f"would pick {f4['any_metric_up']}.",
        f"The direction changes {f4['arms'] - f4['same_both_directions']} of {f4['arms']} verdicts and the choice of primary "
        f"{f4['decided_arms'] - f4['same_any_primary_decided']} of {f4['decided_arms']}",
        f"{f4['guards']['flat_with_worse']} arms with a quiet primary are stopped by a guard rail",
        f"a {100 * D.MARGIN:g}% margin",
        # the recommendations table
        f"the direction changes {f4['arms'] - f4['same_both_directions']} of {f4['arms']} verdicts and the choice of primary "
        f"{f4['decided_arms'] - f4['same_any_primary_decided']} of {f4['decided_arms']}. Part 2: \"any of four metrics\" "
        f"calls {f2['arms_any']} of {f2['arms_read']} arms a result, {f2['arms_any_holm']} after correction",
        f"Part 1: {f1['naive_alarms']} alarms against 50/50, {f1['unequal']} of them",
        f"roughly {f2['need'][1] / 1e6:.1f} to {f2['need'][4] / 1e6:.1f} million users per group",
        f"Part 4: {up['no change']} treatments rule out a {100 * D.MARGIN:g}% change, {up['run longer']} cannot tell",
        f"{100 * pk['null_mean']:.1f}% false alarms for daily peeking against {100 * avc['null_mean']:.1f}% for the confirmed",
        f"still found {avc['sig_end']} of the {f3['sig_end']} results that held",
        f"half of the {up['do not launch']} \"do not launch\" verdicts",
        f"Part 3: {f3['blips']} blips, {f3['blips_by_metric'][2]} of them in metric 2",
        f"one final snapshot (`{jump.experiment_id}` / {jump.variant_id}) is suspect",
        f"median factor of {pk['overstatement']:.2f} under daily peeking and {avc['overstatement']:.2f} under the confirmed",
    ]


@pytest.fixture(scope="module")
def all_facts(b, f, q, k, d):
    return f, q[2], k[2], d[2], d[0].D


def test_every_summary_figure_matches_the_results(all_facts):
    missing = [s for s in summary_statements(*all_facts) if s not in SUMMARY]
    assert not missing, missing


def test_summary_statements_in_words(all_facts, b):
    f1, f2, f3, f4, D = all_facts
    assert (f1["arms"] - f1["clean_arms"]) / f1["arms"] > 1 / 3                  # "Over a third"
    assert 2 * f4["guards"]["flat_with_worse"] == f4["up"]["do not launch"]      # "half of the 20"
    assert f4["guards"]["flat_with_worse"] == 10
    small = b["arms"][b["arms"].split == "small imbalance"]
    worst = small.loc[small.users_short.idxmax()]                                # the largest shortfall
    assert f"(a treatment share of {100 * worst.treatment_share:.3f}%)" in SUMMARY
    assert ((small.treatment_share - 0.5).abs() < 0.001).all()                   # "within a tenth of a point"
    case = f4["cases"]["jump"]
    assert case["row"].verdict == "check the data"
    jump = f3["end_jumps"].iloc[0]
    assert f"`{jump.experiment_id}` / {jump.variant_id}" == f"`{case['row'].experiment_id}` / {case['row'].variant_id}"
    assert D.PRIMARY == 1 and D.MARGIN == 0.01


def test_no_summary_figure_is_new():
    """The summary reports Parts 1 to 4; every number in it must already appear below it."""
    numbers = set(re.findall(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?%?", SUMMARY))
    absent = sorted(n for n in numbers if n not in BODY)
    assert not absent, absent


def test_recommendations_table():
    rows = [r for r in section("### What a team should take from it", "What this analysis cannot do").splitlines()
            if re.match(r"\| \d+ \|", r)]
    assert [int(r.split("|")[1]) for r in rows] == list(range(1, 8))
    for r in rows:
        evidence = r.rstrip(" |").split("|")[-1]
        assert re.search(r"Part [1-4]", evidence), r
    assert "[Limits](#limits)" in README and "\n## Limits\n" in README
