"""Part 3: the cost of peeking.

    python scripts/peeking_report.py

Reads the committed CSV in data/raw/ (no download needed) and writes
results/PEEKING.md, four CSVs and charts 07 to 09. The simulation is seeded,
so every run gives the same figures. Takes about a minute.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abtest import checks as C, load, peeking as P, readout as R  # noqa: E402
from scripts.data_checks_report import (  # noqa: E402
    BLUE, CHARTS, GRID, INK, INK_2, METRIC_TYPE, MUTED, ORANGE, RESULTS, SAVE, SURFACE,
    arm_name, md_table, n, style)
from scripts.readout_report import pc, p_text  # noqa: E402

KEY = ["experiment_id", "variant_id"]
RULE_NAME = {"end": "End only (Part 2)", "peek": "Daily peeking",
             "peek_confirmed": "Daily peeking, confirmed", "always": "Always-valid",
             "always_confirmed": "Always-valid, confirmed"}
EQUAL_LOOKS = (1, 2, 3, 5, 7, 10, 14, 20, 30, 50, 75, 100)
MINIMUM_RUNS = (7.0, 14.0, 21.0, 28.0)
Z_CLIP = 7.0   # chart 08 shows z between -7 and +7


# ------------------------------------------------------------------ build
def simulate(res: pd.DataFrame, paths: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """No-change tests on every arm's own look schedule. Returns the replay
    table with each test's simulated false-alarm rates added (`null_<rule>`),
    one row per arm, and the blip count of the simulation."""
    per_test, per_arm, blip_looks, simulated = {}, [], 0, 0
    for (e, v), g in res.groupby(KEY, sort=True):
        base = paths[(e, v, 1)]                      # metric 1 is recorded at every look
        t = P.info_fractions(base["users_c"], base["users_t"])
        z = P.null_z(t, P.PATHS, np.random.default_rng(P.SEED))
        blip_looks += int(P.blip_mask(z).sum())
        simulated += int(z.size)
        col = {day: i for i, day in enumerate(base["day"])}
        arm = {"experiment_id": e, "variant_id": v, "looks": len(base),
               "decision_looks": int(P.decision_looks(base["day"]).sum())}
        for r in g.itertuples(index=False):
            s = paths[(e, v, r.metric_id)]
            cols = [col[day] for day in s["day"]]    # a metric with no variance at a look skips it
            o = P.null_rates(z[:, cols], t[cols], P.decision_looks(s["day"]),
                             {r.metric_id: P.TAU / r.lift_se_end})
            per_test[(e, v, r.metric_id)] = {
                "null_end": o["end"], "null_peek": o["peek"], "null_peek_confirmed": o["peek_confirmed"],
                "null_always": o["always"][r.metric_id],
                "null_always_confirmed": o["always_confirmed"][r.metric_id]}
            if r.metric_id == 1:
                arm.update(end=o["end"], peek=o["peek"], peek_confirmed=o["peek_confirmed"])
            arm[f"always_m{r.metric_id}"] = o["always"][r.metric_id]
            arm[f"always_confirmed_m{r.metric_id}"] = o["always_confirmed"][r.metric_id]
        per_arm.append(arm)
    null = pd.DataFrame([per_test[k] for k in zip(res["experiment_id"], res["variant_id"], res["metric_id"])])
    cols = (["experiment_id", "variant_id", "looks", "decision_looks", "end", "peek", "peek_confirmed"]
            + [f"always_m{m}" for m in (1, 2, 3, 4)] + [f"always_confirmed_m{m}" for m in (1, 2, 3, 4)])
    return (pd.concat([res.reset_index(drop=True), null], axis=1),
            pd.DataFrame(per_arm, columns=cols), {"blip_looks": blip_looks, "looks": simulated})


def by_minimum_run(paths: dict) -> pd.DataFrame:
    """Every rule replayed with a longer minimum run before it may stop a test."""
    rows = []
    for first in MINIMUM_RUNS:
        d = pd.DataFrame([P.decide(s, first_day=first) for s in paths.values()])
        row = {"first_decision_day": first, "tests": len(d), "significant_at_end": int(d["sig_end"].sum())}
        for rule in P.RULES:
            stops = d[f"{rule}_stops"]
            row[f"{rule}_stops"] = int(stops.sum())
            row[f"{rule}_not_significant_at_end"] = int((stops & ~d["sig_end"]).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build() -> dict:
    raw = load.read_raw()
    con = load.connect(raw)
    arms = C.check_arms(raw, load.run_sql(con, "arm_summary"))
    table = R.read_arms(load.run_sql(con, "final_snapshot"), arms)
    res, paths = P.replay(load.run_sql(con, "daily_looks"), table, arms)
    res, null_arms, sim = simulate(res, paths)
    curve = pd.DataFrame({"looks": EQUAL_LOOKS,
                          "false_alarm_rate": [P.equal_looks_rate(k) for k in EQUAL_LOOKS]})
    return {"arms": arms, "table": table, "res": res, "paths": paths, "null_arms": null_arms,
            "sim": sim, "curve": curve, "minimum": by_minimum_run(paths)}


def rule_facts(res: pd.DataFrame, rule: str) -> dict:
    stops = res[f"{rule}_stops"]
    s = res[stops]
    both = s[s["sig_end"]]
    return {
        "stops": int(stops.sum()),
        "sig_end": len(both), "not_sig_end": int((~s["sig_end"]).sum()),
        "on_blip": int(s[f"{rule}_on_blip"].sum()),
        "on_blip_not_sig_end": int((s[f"{rule}_on_blip"] & ~s["sig_end"]).sum()),
        "expected_null": float(res[f"null_{rule}"].sum()),
        "null_mean": float(res[f"null_{rule}"].mean()),
        "null_min": float(res[f"null_{rule}"].min()), "null_max": float(res[f"null_{rule}"].max()),
        "arms": s.groupby(KEY).ngroups,
        "opposite_sign": int((np.sign(s[f"{rule}_lift"]) != np.sign(s["lift_end"])).sum()),
        "opposite_sign_sig_end": int((np.sign(both[f"{rule}_lift"]) != np.sign(both["lift_end"])).sum()),
        "overstatement": float((both[f"{rule}_lift"].abs() / both["lift_end"].abs()).median()),
        "larger_at_stop": int((both[f"{rule}_lift"].abs() > both["lift_end"].abs()).sum()),
        "stop_day_median": float(both[f"{rule}_day"].median()),
        "last_day_median": float(both["last_day"].median()),
        "days_saved_median": float((both["last_day"] - both[f"{rule}_day"]).median()),
        "share_of_run_median": float((both[f"{rule}_day"] / both["last_day"]).median()),
        "holm": int((s["p_holm"] < P.ALPHA).sum()),
    }


def long_blips(mask) -> int:
    """Number of blips in a 1-D mask that last more than one look."""
    m = np.asarray(mask, dtype=bool)
    runs, length = 0, 0
    for v in list(m) + [False]:
        if v:
            length += 1
        else:
            runs += length > 1
            length = 0
    return runs


def examples(res: pd.DataFrame) -> dict:
    """Three tests for chart 08, each picked by a fixed rule. The first is
    taken from the tests whose statistic stays on the chart's scale."""
    ac = res["always_confirmed_stops"]
    holds = res[ac & res["sig_end"] & (res["blips"] == 0) & (res["max_abs_z"] <= Z_CLIP)]
    holds = holds.assign(saved=holds["last_day"] - holds["always_confirmed_day"]).sort_values(
        ["saved", "experiment_id", "variant_id", "metric_id"], ascending=[False, True, True, True]).iloc[0]
    fades = res[ac & ~res["sig_end"] & (res["blips"] == 0)].sort_values(
        ["p_end", "experiment_id", "variant_id", "metric_id"], ascending=[False, True, True, True]).iloc[0]
    glitch = res[res["always_stops"] & res["always_on_blip"] & ~res["sig_end"]].sort_values(
        ["p_end", "experiment_id", "variant_id", "metric_id"], ascending=[False, True, True, True]).iloc[0]
    return {"holds": holds, "fades": fades, "glitch": glitch}


def facts(b: dict) -> dict:
    """Every number the report and the README quote, computed once."""
    res, table, arms = b["res"], b["table"], b["arms"]
    read = table[table["reading"].isin(["read", "read with warning"])]
    replayed = P.replayed_arms(arms, table)
    read_arms = read[KEY].drop_duplicates().merge(arms, on=KEY)
    skipped = read_arms[~read_arms.set_index(KEY).index.isin(replayed.set_index(KEY).index)]
    blippy = res[res["blips"] > 0]
    bounds = [P.always_valid_bound(s["lift_se"][P.decision_looks(s["day"])]) for s in b["paths"].values()]
    jumps = res[res["end_jump"]]
    per_arm = res.groupby(KEY).agg(end=("sig_end", "any"), peek=("peek_stops", "any"),
                                   always_confirmed=("always_confirmed_stops", "any"))
    rules = {rule: rule_facts(res, rule) for rule in P.RULES}
    missed = res[res["sig_end"] & ~res["always_confirmed_stops"]]
    quiet = res[~res["sig_end"]]
    fading = res[res["always_confirmed_stops"] & ~res["sig_end"]]
    curve = b["curve"].set_index("looks")["false_alarm_rate"]
    return {
        "tests_part2": len(read), "arms_part2": len(read_arms),
        "arms": len(replayed), "arms_skipped": len(skipped),
        "skipped_late": int(skipped["cautions"].str.contains("first week not recorded").sum()),
        "skipped_gap": int(skipped["cautions"].str.contains("gap in snapshots").sum()),
        "experiments": res["experiment_id"].nunique(), "tests": len(res),
        "looks_total": int(res["looks"].sum()),
        "looks_median": float(res["looks"].median()), "looks_min": int(res["looks"].min()),
        "looks_max": int(res["looks"].max()),
        "decision_looks_median": float(res["decision_looks"].median()),
        "last_day_median": float(res["last_day"].median()),
        "sig_end": int(res["sig_end"].sum()), "quiet_end": len(quiet),
        "expected_end": P.ALPHA * len(res), "null_end_mean": float(res["null_end"].mean()),
        "rules": rules,
        "quiet_peek": int(quiet["peek_stops"].sum()),
        "missed": len(missed), "missed_p_min": float(missed["p_end"].min()),
        "missed_p_max": float(missed["p_end"].max()),
        "missed_holm": int((missed["p_holm"] < P.ALPHA).sum()),
        "blips": int(res["blips"].sum()), "blip_looks": int(res["blip_looks"].sum()),
        "blip_tests": len(blippy), "blip_arms": blippy.groupby(KEY).ngroups,
        "blips_by_metric": {int(m): int(v) for m, v in res.groupby("metric_id")["blips"].sum().items()},
        "blips_one_look": int(sum(P.episodes(s["blip"]) for s in b["paths"].values())
                              - sum(long_blips(s["blip"]) for s in b["paths"].values())),
        "sim_blip_looks": b["sim"]["blip_looks"], "sim_looks": b["sim"]["looks"],
        "end_jumps": jumps, "end_jump": len(jumps), "end_jump_sig": int(jumps["sig_end"].sum()),
        "bound_min": float(min(x.min() for x in bounds)),
        "bound_first_median": float(np.median([x[0] for x in bounds])),
        "peek_last_look_only": int((res["peek_stops"] & (res["peek_day"] == res["last_day"])).sum()),
        "missed_p_median": float(missed["p_end"].median()),
        "fading_on_blip": int(fading["always_confirmed_on_blip"].sum()),
        "arms_any_end": int(per_arm["end"].sum()), "arms_any_peek": int(per_arm["peek"].sum()),
        "arms_any_always_confirmed": int(per_arm["always_confirmed"].sum()),
        "fading": fading, "fading_arms": fading.groupby(KEY).ngroups,
        "curve": {int(k): float(v) for k, v in curve.items()},
        "minimum": b["minimum"],
        "examples": examples(res),
        "z_crit": float(stats.norm.isf(P.ALPHA / 2)),
    }


# ------------------------------------------------------------------ charts
def chart_looks(b: dict, f: dict, path: Path) -> None:
    """False-alarm rate against the number of looks, when nothing changed."""
    curve, na = b["curve"], b["null_arms"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(curve["looks"], 100 * curve["false_alarm_rate"], color=INK_2, linewidth=1.6, zorder=2)
    ax.scatter(na["decision_looks"], 100 * na["peek"], s=26, color=ORANGE, edgecolor=SURFACE,
               linewidth=0.7, zorder=3, label="Daily peeking: stop at the first p < 0.05")
    av = na[[f"always_confirmed_m{m}" for m in (1, 2, 3, 4)]].mean(axis=1)
    ax.scatter(na["decision_looks"], 100 * av, s=26, color=BLUE, marker="D", edgecolor=SURFACE,
               linewidth=0.7, zorder=3, label="Always-valid rule, confirmed on the next look")
    ax.axhline(100 * P.ALPHA, color=INK_2, linewidth=0.9, linestyle=(0, (3, 3)))
    ax.text(128, 100 * P.ALPHA + 0.6, "the 5% a single test promises", fontsize=9, color=INK, ha="right")
    ax.text(100, 100 * f["curve"][100] + 1.2, "equally spaced looks", fontsize=9, color=INK_2, ha="right")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlim(0.9, 135)
    ax.set_ylim(0, 42)
    ax.set_yticks([0, 10, 20, 30, 40], ["0%", "10%", "20%", "30%", "40%"])
    ax.set_xlabel("Number of looks at which the test could be stopped (log scale)")
    ax.set_ylabel("Tests called a result\nwhen nothing changed")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    style(ax, f"Checking a test every day turns a 5% false-alarm rate into about {100 * f['rules']['peek']['null_mean']:.0f}%",
          f"Simulated tests with no real change. One dot per replayed arm ({len(na)}), on that arm's own schedule of looks from day 7.")
    fig.tight_layout()
    fig.savefig(path, **SAVE)
    plt.close(fig)


def chart_paths(b: dict, f: dict, path: Path) -> None:
    """Three tests, look by look, against both boundaries."""
    ex = f["examples"]
    panels = (("holds", "A change that holds"), ("fades", "An early change that fades"),
              ("glitch", "A data fault"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), sharey=True)
    for ax, (name, title) in zip(axes, panels):
        r = ex[name]
        s = b["paths"][(r.experiment_id, r.variant_id, r.metric_id)]
        sign = 1 if s["z"].iloc[np.argmax(np.abs(s["z"].to_numpy()))] >= 0 else -1   # draw the peak upwards
        z = (sign * s["z"]).clip(-Z_CLIP, Z_CLIP)
        bound = np.minimum(P.always_valid_bound(s["lift_se"]), Z_CLIP + 1)
        ax.axvspan(0, P.FIRST_DECISION_DAY, color=GRID, alpha=0.5, linewidth=0)
        for k in (1, -1):
            ax.axhline(k * f["z_crit"], color=ORANGE, linewidth=1.0, linestyle=(0, (3, 3)),
                       label="p = 0.05 (a single test)" if k == 1 else None)
            ax.plot(s["day"], k * bound, color=BLUE, linewidth=1.2,
                    label="always-valid boundary" if k == 1 else None)
        ax.plot(s["day"], z, color=INK, linewidth=1.3, label="test statistic (z)")
        sp = s[s["blip"]]
        ax.scatter(sp["day"], (sign * sp["z"]).clip(-Z_CLIP, Z_CLIP), s=28, facecolor=SURFACE,
                   edgecolor=INK, linewidth=1.1, zorder=4, label="blip" if name == "glitch" else None)
        style(ax)
        ax.set_title(f"{title}\n{r.experiment_id} / {r.variant_id}, metric {r.metric_id}", loc="left",
                     fontsize=9.5, color=INK)
        ax.set_xlabel("Days since the test started", fontsize=9)
        ax.set_xlim(0, s["day"].max() + 1)
        ax.set_ylim(-Z_CLIP, Z_CLIP)
    axes[0].set_ylabel("Treatment against control,\nin standard errors (z)")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("The same three tests read every day: one result holds, one fades, one is a data fault",
                 x=0.012, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.012, 0.9, "A rule calls a result when the black line leaves its band. Grey: the first week, when no rule may stop. "
             "Each test is drawn so that its largest swing points up.", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(path, **SAVE)
    plt.close(fig)


def chart_rules(b: dict, f: dict, path: Path) -> None:
    """Tests called a result under each rule, split by the end-of-test verdict."""
    order = ("end", "peek", "peek_confirmed", "always", "always_confirmed")
    held = [f["sig_end"]] + [f["rules"][r]["sig_end"] for r in order[1:]]
    gone = [0] + [f["rules"][r]["not_sig_end"] for r in order[1:]]
    chance = [f["expected_end"]] + [f["rules"][r]["expected_null"] for r in order[1:]]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.barh(y, held, color=BLUE, height=0.56, label="significant at the end of the test")
    ax.barh(y, gone, left=held, color=ORANGE, height=0.56, label="not significant at the end")
    ax.scatter(chance, y, marker="|", s=420, color=INK, linewidth=1.8, zorder=4,
               label="expected if no treatment changed anything")
    for yi, h, g in zip(y, held, gone):
        ax.text(h + g + 2, yi, f"{h + g}" + (f"  ({g} not significant at the end)" if g else ""),
                va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [RULE_NAME[r] for r in order])
    ax.set_xlim(0, 205)
    ax.set_xlabel(f"Tests called a result, of {f['tests']} replayed")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title(f"Daily peeking calls {f['rules']['peek']['stops']} results where the end of the test supports {f['sig_end']}",
                 loc="left", fontsize=12, fontweight="bold", pad=46)
    ax.text(0, 1.125, "Every rule may stop from day 7. \"Confirmed\" needs the signal on two looks in a row.",
            transform=ax.transAxes, fontsize=9, color=INK_2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.995), ncol=3, frameon=False, fontsize=8.5,
              handlelength=1.2, columnspacing=1.2, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(path, **SAVE)
    plt.close(fig)


# ------------------------------------------------------------------ report
def render(b: dict, f: dict) -> str:
    res, rules, a = b["res"], f["rules"], P.ALPHA
    pk, pkc, av, avc = (rules[r] for r in P.RULES)
    ex = f["examples"]

    rule_t = pd.DataFrame([
        ("End only (Part 2)", "one test at the last snapshot", f["sig_end"], f["sig_end"], 0,
         f"{f['expected_end']:.0f}", f"{100 * f['null_end_mean']:.1f}%")] + [
        (RULE_NAME[r], how, rules[r]["stops"], rules[r]["sig_end"], rules[r]["not_sig_end"],
         f"{rules[r]['expected_null']:.0f}", f"{100 * rules[r]['null_mean']:.1f}%")
        for r, how in (("peek", f"the first daily look with p < {a}"),
                       ("peek_confirmed", f"p < {a} on two looks in a row"),
                       ("always", f"the first daily look where the evidence ratio reaches {1 / a:.0f}"),
                       ("always_confirmed", f"evidence ratio of {1 / a:.0f} or more on two looks in a row"))],
        columns=["Rule", "Calls a result at", "Tests called a result", "Significant at the end",
                 "Not significant at the end", "Expected if nothing changed", "Simulated false-alarm rate"])

    curve_t = pd.DataFrame({"Looks": list(f["curve"]),
                            "False-alarm rate": [f"{100 * v:.1f}%" for v in f["curve"].values()]})

    met = res.groupby("metric_id").agg(tests=("sig_end", "size"), end=("sig_end", "sum"),
                                       peek=("peek_stops", "sum"), always=("always_stops", "sum"),
                                       avc=("always_confirmed_stops", "sum"), blips=("blips", "sum")).reset_index()
    met_t = pd.DataFrame({
        "Metric": [f"{m}: {METRIC_TYPE[m]}" for m in met["metric_id"]],
        "Tests": met["tests"], "Significant at the end": met["end"], "Daily peeking": met["peek"],
        "Always-valid": met["always"], "Always-valid, confirmed": met["avc"], "Blips": met["blips"],
    })

    blippy = res[res["blips"] > 0].groupby(KEY).agg(
        blips=("blips", "sum"), looks=("blip_looks", "sum"),
        metrics=("metric_id", lambda s: ", ".join(str(m) for m in sorted(s))),
        last_day=("last_day", "first")).reset_index().sort_values(["blips", "experiment_id", "variant_id"],
                                                                  ascending=[False, True, True])
    blip_t = pd.DataFrame({
        "Arm": [arm_name(e, v) for e, v in zip(blippy["experiment_id"], blippy["variant_id"])],
        "Blips": blippy["blips"], "Looks in a blip": blippy["looks"], "Metrics affected": blippy["metrics"],
        "Days the test ran": [f"{d:g}" for d in blippy["last_day"]],
    })

    ej = f["end_jumps"]
    jump_text = "; ".join(
        f"{arm_name(r.experiment_id, r.variant_id)}, metric {r.metric_id} (final p = {p_text(r.p_end)})"
        for r in ej.itertuples(index=False))

    fd = f["fading"].sort_values(["experiment_id", "variant_id", "metric_id"])
    fade_t = pd.DataFrame({
        "Arm": [arm_name(e, v) for e, v in zip(fd["experiment_id"], fd["variant_id"])],
        "Metric": fd["metric_id"],
        "Called on day": [f"{d:g}" for d in fd["always_confirmed_day"]],
        "Change then": [pc(v) for v in fd["always_confirmed_lift"]],
        "Test ended on day": [f"{d:g}" for d in fd["last_day"]],
        "Change at the end": [pc(v) for v in fd["lift_end"]],
        "p at the end": [p_text(v) for v in fd["p_end"]],
        "Called during a blip": ["yes" if v else "no" for v in fd["always_confirmed_on_blip"]],
    })

    mn = f["minimum"]
    min_t = pd.DataFrame({
        "No decision before day": [f"{d:g}" for d in mn["first_decision_day"]],
        "Daily peeking": [f"{s} ({g})" for s, g in zip(mn["peek_stops"], mn["peek_not_significant_at_end"])],
        "Daily peeking, confirmed": [f"{s} ({g})" for s, g in zip(mn["peek_confirmed_stops"], mn["peek_confirmed_not_significant_at_end"])],
        "Always-valid": [f"{s} ({g})" for s, g in zip(mn["always_stops"], mn["always_not_significant_at_end"])],
        "Always-valid, confirmed": [f"{s} ({g})" for s, g in zip(mn["always_confirmed_stops"], mn["always_confirmed_not_significant_at_end"])],
    })
    by_metric = ", ".join(f"{v} in metric {m}" for m, v in f["blips_by_metric"].items())
    clean_fades = avc["not_sig_end"] - f["fading_on_blip"]
    m28 = mn[mn["first_decision_day"] == MINIMUM_RUNS[-1]].iloc[0]

    return f"""# Part 3: the cost of peeking

*Generated by `scripts/peeking_report.py` from the committed source file. Every number below is computed by the script; nothing is typed in by hand. The simulation is seeded, so a rerun gives the same figures.*

Part 2 read each test once, at its last snapshot. Real teams do not wait: the dashboard updates every day and someone looks. This part asks what that does. It replays each test as a series of daily looks, simulates tests in which nothing changed on the same schedule of looks, and tries a stopping rule that is built to be checked every day.

**Headline:** read once at the end, {f['sig_end']} of {f['tests']} tests are significant. A team that looked every day from day 7 and called a result at the first p < {a} would have called {pk['stops']}, and {pk['not_sig_end']} of those are not significant when the test ends. In simulated tests where nothing changed, that habit calls a result {100 * pk['null_mean']:.0f}% of the time, where a single test promises {100 * a:.0f}%. An always-valid rule, confirmed on the next look, brings the simulated rate down to {100 * avc['null_mean']:.1f}% and calls {avc['stops']} results, {avc['sig_end']} of them significant at the end, a median of {avc['days_saved_median']:g} days before the test finished.

As in Part 2, the metrics are not named, so nothing here is called a win or a loss.

## 1. What is replayed

- **A daily look** is the latest snapshot in each 24-hour window since the test started ([`sql/daily_looks.sql`](../sql/daily_looks.sql)), so an arm recorded every 12 hours and one recorded every 24 are both looked at once a day. A snapshot whose cumulative user count is lower than an earlier one is not a look.
- **{f['arms']} of the {f['arms_part2']} arms that Part 2 read are replayed** ({f['experiments']} experiments, {f['tests']} of Part 2's {f['tests_part2']} tests). The other {f['arms_skipped']} are left out because their daily record is incomplete (Part 1): {f['skipped_late']} were first recorded after day 7 and {f['skipped_gap']} have a gap of three days or more.
- That gives {n(f['looks_total'])} looks. A test has a median of {f['looks_median']:g} looks ({f['looks_min']} to {f['looks_max']}) and ran for a median of {f['last_day_median']:g} days.
- **No rule may stop a test before day {P.FIRST_DECISION_DAY:g}.** Part 1's rule that a test must see every day of the week applies to every rule below. The last look always counts as a decision.
- At every look the test is the one Part 2 used (Welch's t-test; relative change with a delta-method standard error). At the last look the figures equal Part 2's.

## 2. What looking every day does when nothing changed

A p-value below {a} happens by chance in {100 * a:.0f}% of tests in which nothing changed. That is the promise of one test. Each extra look is another chance, and the looks are not independent, so the chances do not simply add; a simulation shows what they come to.

![False-alarm rate against the number of looks](charts/07_false_alarms_by_looks.png)

With equally spaced looks and no real change, calling a result at the first p < {a} gives ({n(P.PATHS_EQUAL)} simulated tests per row):

{md_table(curve_t)}

For the experiments here, the simulation uses each arm's own schedule: the day of every look and the share of its final users it had by then ({n(P.PATHS)} simulated tests per arm, [`peeking_null_by_arm.csv`](peeking_null_by_arm.csv)).

- **Daily peeking from day {P.FIRST_DECISION_DAY:g} calls a false result in {100 * pk['null_mean']:.1f}% of no-change tests on average** ({100 * pk['null_min']:.1f}% to {100 * pk['null_max']:.1f}% depending on the arm's schedule). That is {pk['null_mean'] / a:.1f} times the {100 * a:.0f}% a single test promises.
- **Waiting for a second look in a row helps a little:** {100 * pkc['null_mean']:.1f}%. Two looks a day apart share almost all their users, so the second mostly repeats the first.
- **The always-valid rule stays under {100 * a:.0f}%:** {100 * av['null_mean']:.1f}% on average (at most {100 * av['null_max']:.1f}%), and {100 * avc['null_mean']:.1f}% when confirmed on the next look. It is stricter than it needs to be on these schedules, which costs some sensitivity (section 5).

## 3. The rule that can be checked every day

The always-valid rule ([`abtest/peeking.py`](../abtest/peeking.py)) is the mixture sequential probability ratio test. At each look it computes one number, the *evidence ratio*: how much better the data fits "the treatment changed the metric by some amount" than "nothing changed", averaged over a spread of possible changes (a normal distribution with a standard deviation of {100 * P.TAU:.0f}% of the control mean, Part 2's reference change). It calls a result when the ratio reaches {1 / a:.0f} (1 ÷ {a}).

- The guarantee: if nothing changed, the chance that the ratio *ever* reaches {1 / a:.0f} is at most {100 * a:.0f}%, however many times it is checked. It needs no planned end date and no planned number of looks.
- The price: its bar is higher than a single test's. A single test needs the difference to be {f['z_crit']:.2f} standard errors; the always-valid rule needs at least {f['bound_min']:.2f} at every look in these tests (a median of {f['bound_first_median']:.2f} at the first look where a decision is allowed).
- The spread ({100 * P.TAU:.0f}%) changes how sensitive the rule is to changes of different sizes. It does not change the guarantee.

## 4. Snapshots that cannot be right

Replaying the looks turned up a data fault that Part 1's checks on user counts could not see.

- **{f['blips']} blips, covering {f['blip_looks']} looks.** A blip is a run of 1 to {P.MAX_BLIP_LOOKS} looks at which the test statistic has jumped by more than {P.BLIP_Z:g} standard errors and after which it jumps back by more than {P.BLIP_Z:g}. {f['blips_one_look']} of the {f['blips']} last a single look. They are in {f['blip_tests']} tests of {f['blip_arms']} arms: {by_metric}.
- **Chance almost never does this.** A few days of new users are a small share of a test, so the statistic moves slowly from look to look. The simulated no-change tests have {n(f['sim_blip_looks'])} looks in a blip out of {n(f['sim_looks'])}; the real tests have {f['blip_looks']} out of {n(f['looks_total'])}.
- The likely cause is a snapshot in which one group's metric had been updated and the other's had not. That is an inference from the pattern (mostly one metric, a few looks, then back to where it was); the source does not say.
- **A rule that acts on the first signal acts on these.** {pk['on_blip']} of the {pk['stops']} peeking results and {av['on_blip']} of the {av['stops']} always-valid results were called at a look inside a blip; {av['on_blip_not_sig_end']} of those {av['on_blip']} are not significant at the end. The always-valid rule's higher bar is no defence, because a fault can be any size.
- **Asking for the signal on two looks in a row removes the one-look blips and no more.** {avc['on_blip']} of the {avc['stops']} confirmed always-valid results were still called inside a blip, one that lasted two or three looks. Confirmation is a cheap first defence (it costs a day); the real fix is in the pipeline, which should hold back a snapshot whose result jumps away from the day before.
- **A fault in the last snapshot cannot be confirmed**, because no later look shows whether it jumps back. In {f['end_jump']} of the {f['tests']} tests the last look is more than {P.BLIP_Z:g} standard errors away from the last look before it that is not in a blip: {jump_text}. Part 2 reads that snapshot, so its figure for that test deserves a second look before anyone relies on it.

{md_table(blip_t, left=("Arm", "Metrics affected"))}

## 5. The real tests, read every day

![Tests called a result under each rule](charts/09_results_by_rule.png)

{md_table(rule_t, left=("Rule", "Calls a result at"))}

"Expected if nothing changed" is the sum, over the {f['tests']} tests, of each test's simulated false-alarm rate. The comparison with the end of the test is a yardstick, not the truth: the final reading can be wrong as well.

- **Daily peeking calls {pk['stops']} results where the end of the test supports {f['sig_end']}.** It cannot miss a test that ends significant, because the last look is one of its looks ({f['peek_last_look_only']} of the {pk['stops']} are called only there); it adds {pk['not_sig_end']} results that do not last. Of the {f['quiet_end']} tests that end without a significant change, {f['quiet_peek']} ({100 * f['quiet_peek'] / f['quiet_end']:.0f}%) showed p < {a} on at least one day.
- **By arm it is worse.** With four metrics and daily looks, {f['arms_any_peek']} of {f['arms']} arms show a "significant" metric at some point, against {f['arms_any_end']} at the end.
- **A result called early is overstated.** For the {pk['sig_end']} tests that peeking calls and that end significant, the change on the day it was called was larger than the final one in {pk['larger_at_stop']}, by a median factor of {pk['overstatement']:.2f}. In {pk['opposite_sign']} of the {pk['stops']} peeking results the change even has the opposite sign to the final estimate ({pk['opposite_sign_sig_end']} of them in tests that end significant).
- **The confirmed always-valid rule calls {avc['stops']} results:** {avc['sig_end']} of the {f['sig_end']} that are significant at the end, and {avc['not_sig_end']} that are not. If nothing had changed in any test it would be expected to call about {avc['expected_null']:.0f}.
- **It decides early.** For those {avc['sig_end']}, the result came on a median of day {avc['stop_day_median']:g} of a test that ran {avc['last_day_median']:g} days: {avc['days_saved_median']:g} days sooner. The estimate on that day is still inflated (median factor {avc['overstatement']:.2f} against the final one), so the size should be re-read at the end even when the decision is made early.
- **It misses {f['missed']} tests that end significant.** That is the price of the higher bar. Their final p-values have a median of {p_text(f['missed_p_median'])}, and {f['missed_holm']} of them survive Part 2's correction for multiple metrics, so they are not all borderline. Of the {avc['stops']} results the rule does call, {avc['holm']} survive that correction.

By metric:

{md_table(met_t)}

![Three tests look by look](charts/08_three_tests.png)

The three tests are picked by rule, not by eye: the confirmed result with the most days saved among those whose statistic stays within ±{Z_CLIP:g} ({arm_name(ex['holds'].experiment_id, ex['holds'].variant_id)}, metric {ex['holds'].metric_id}), the confirmed result that ended furthest from significance ({arm_name(ex['fades'].experiment_id, ex['fades'].variant_id)}, metric {ex['fades'].metric_id}), and the always-valid result called inside a blip that ended furthest from significance ({arm_name(ex['glitch'].experiment_id, ex['glitch'].variant_id)}, metric {ex['glitch'].metric_id}).

## 6. Results that do not last

{avc['not_sig_end']} results in {f['fading_arms']} arms pass the confirmed always-valid rule and are not significant when the test ends, where chance would give about {avc['expected_null']:.0f} across all {f['tests']} tests:

{md_table(fade_t)}

{f['fading_on_blip']} of the {avc['not_sig_end']} were called inside a blip that lasted more than one look: data faults (section 4). For the other {clean_fades} no fault is marked: the signal was there on at least two looks in a row and the change then shrank. Two explanations fit those, and this data cannot separate them:

1. **The change was real and wore off**, or it was concentrated in the first users. A new feature often gets attention that fades. Then the early call was right about the first weeks and wrong about the long run, and no statistical rule about chance protects against that.
2. **The simulation is too kind.** It assumes each user's value is fixed once counted. These metrics are cumulative: a user counted in week one keeps adding to them, so two looks are less alike than the simulation assumes, and every rule's true false-alarm rate is somewhat higher than the simulated one. Without user-level data the size of that effect cannot be measured.

A longer minimum run removes some of them: with no decision before day {m28.first_decision_day:g}, the confirmed always-valid rule calls {int(m28.always_confirmed_stops)} results and {int(m28.always_confirmed_not_significant_at_end)} are not significant at the end. Tests called a result (in brackets: not significant at the end) when no rule may stop before the given day:

{md_table(min_t)}

## 7. So what, for a team that runs experiments

1. **Do not stop a test on the first p < {a}.** Looking daily and stopping on the first significant day calls a result in about {100 * pk['null_mean']:.0f}% of tests where nothing changed, and here it would have called {pk['stops']} results where {f['sig_end']} held to the end. Either fix the end date before the test and read it once, or use a rule built for daily looks.
2. **If the team wants to look every day, give it an always-valid rule.** It kept the simulated false-alarm rate at {100 * avc['null_mean']:.1f}%, found {avc['sig_end']} of the {f['sig_end']} results that held, and found them a median of {avc['days_saved_median']:g} days early. That is test capacity freed for the next idea.
3. **Never act on a single look, and check the snapshot before the statistics.** This dataset has {f['blips']} blips where a result jumps and jumps back, and {av['on_blip']} always-valid results were called inside one. Requiring the signal on two looks in a row costs a day and removes the one-look faults; {avc['on_blip']} longer ones still got through, so the pipeline should flag any snapshot that jumps away from the day before.
4. **Set a minimum run for the business, not only for the statistics.** {clean_fades} results passed the strict rule with no data fault marked and were not significant at the end. If the decision is about a permanent change, the test has to run long enough to see whether the change lasts; the table in section 6 shows what two, three and four weeks would have done here.
5. **Re-read the size at the end.** A result called early is overstated (a median factor of {avc['overstatement']:.2f} here), so a forecast built on the day-of-decision figure will disappoint.

## 8. Limits

- The end-of-test reading is used as the yardstick, and it is not the truth. A result that "does not last" may be a real early change (section 6), and a result that lasts may still be a false alarm (Part 2, section 5).
- The simulation treats users as independent and each user's value as fixed once counted. With cumulative metrics the second is not true, so the simulated false-alarm rates are best read as lower ends for every rule, the always-valid rule included.
- The replay knows how each test ended and assumes it would have run the same way under any rule. A team that stops early also frees traffic and starts other tests, which is not modelled.
- Nobody knows whether these tests were in fact stopped on their results. If they were, Part 2's p-values are too small. Section 2 shows how far the false-alarm rate can grow; nothing here can measure how far it did.
- The blip rule (more than {P.BLIP_Z:g} standard errors there and back, within {P.MAX_BLIP_LOOKS} looks) is a working choice made after looking at the data. It cannot judge the first or the last look of a test, it misses smaller and longer faults, and it was only run on the tests replayed here.
- The always-valid rule is one of several. A group-sequential design (planned looks with a spending rule) is more sensitive when the end date is fixed in advance; it needs that plan, which this dataset does not publish.
- {f['arms_skipped']} arms that Part 2 read are not replayed, so the counts here are for {f['tests']} tests and are not comparable with Part 2's {f['tests_part2']} without that in mind.
- The 5% level, the day-{P.FIRST_DECISION_DAY:g} minimum and the {100 * P.TAU:.0f}% spread are set in `abtest/peeking.py`.
"""


def write_outputs(b: dict, f: dict, results: Path = RESULTS, charts: Path = CHARTS) -> None:
    kw = {"index": False, "float_format": "%.6g", "lineterminator": "\n"}
    b["res"].to_csv(results / "peeking.csv", **kw)
    b["null_arms"].to_csv(results / "peeking_null_by_arm.csv", **kw)
    b["curve"].to_csv(results / "peeking_equal_looks.csv", **kw)
    b["minimum"].to_csv(results / "peeking_by_minimum_run.csv", **kw)
    chart_looks(b, f, charts / "07_false_alarms_by_looks.png")
    chart_paths(b, f, charts / "08_three_tests.png")
    chart_rules(b, f, charts / "09_results_by_rule.png")
    (results / "PEEKING.md").write_text(render(b, f))


def main() -> int:
    b = build()
    f = facts(b)
    write_outputs(b, f)
    r = f["rules"]
    print(f"{f['tests']} tests replayed in {f['arms']} arms; significant at the end {f['sig_end']}, "
          f"daily peeking {r['peek']['stops']}, always-valid confirmed {r['always_confirmed']['stops']}; "
          f"{f['blips']} blips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
