"""Part 4: decision rules and the go / no-go memo.

    python scripts/decision_report.py

Reads the committed CSV in data/raw/ (no download needed) and writes
results/DECISION_MEMO.md, three CSVs and charts 10 and 11. No simulation, so
it runs in a few seconds.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abtest import checks as C, decision as D, load, peeking as P, readout as R  # noqa: E402
from scripts.data_checks_report import (  # noqa: E402
    BLUE, CHARTS, GRID, INK, INK_2, METRIC_TYPE, MUTED, ORANGE, RESULTS, SAVE, SURFACE,
    arm_name, md_table, n)
from scripts.readout_report import pc, p_text  # noqa: E402

KEY = D.KEY
MARGINS = (0.005, 0.01, 0.02)
DIRECTION_NAME = {1: "higher is better", -1: "lower is better"}
CASES = ("clear", "thin", "guard", "jump", "flat", "wide", "split")
CASE_TITLE = {
    "clear": "A clear result",
    "thin": "A result on thin evidence",
    "guard": "A flat primary that hides a loss",
    "jump": "A result that rests on one snapshot",
    "flat": "A test that found nothing, and can say so",
    "wide": "A test too small to say anything",
    "split": "A result from a test with a broken split",
}


# ------------------------------------------------------------------ build
def build() -> dict:
    raw = load.read_raw()
    con = load.connect(raw)
    arms = C.check_arms(raw, load.run_sql(con, "arm_summary"))
    table = R.read_arms(load.run_sql(con, "final_snapshot"), arms)
    looks = load.run_sql(con, "daily_looks")
    ends = D.end_checks(looks, table)
    replay, _ = P.replay(looks, table, arms)
    up = D.decide_arms(table, ends, direction=1)
    down = D.decide_arms(table, ends, direction=-1)
    by_primary = pd.DataFrame([
        {"primary_metric": m, **counts(D.decide_arms(table, ends, primary=m))} for m in (1, 2, 3, 4)])
    by_margin = pd.DataFrame([
        {"margin": mg, **counts(D.decide_arms(table, ends, margin=mg))} for mg in MARGINS])
    return {"arms": arms, "table": table, "ends": ends, "replay": replay, "up": up, "down": down,
            "by_primary": by_primary, "by_margin": by_margin,
            "per_primary": {m: D.decide_arms(table, ends, primary=m) for m in (1, 2, 3, 4)}}


def counts(dec: pd.DataFrame) -> dict:
    c = dec["verdict"].value_counts()
    return {v: int(c.get(v, 0)) for v in D.VERDICTS}


def decisions_table(b: dict) -> pd.DataFrame:
    """One row per arm with the verdict under both directions and, for the
    primary metric, what the always-valid rule of Part 3 said."""
    up, down = b["up"], b["down"]
    out = up.drop(columns=["direction", "verdict", "reason", "primary_state", "guards_worse",
                           "guards_not_cleared"]).copy()
    for name, d in (("higher_better", up), ("lower_better", down)):
        out[f"verdict_if_{name}"] = d["verdict"].to_numpy()
        out[f"reason_if_{name}"] = d["reason"].to_numpy()
        out[f"guards_worse_if_{name}"] = d["guards_worse"].to_numpy()
        out[f"guards_not_cleared_if_{name}"] = d["guards_not_cleared"].to_numpy()
    rp = b["replay"][b["replay"]["metric_id"] == D.PRIMARY].set_index(KEY)
    idx = pd.MultiIndex.from_frame(out[KEY])
    out["replayed"] = idx.isin(rp.index)
    out["always_valid_confirmed"] = rp["always_confirmed_stops"].reindex(idx).to_numpy()
    out["always_valid_day"] = rp["always_confirmed_day"].reindex(idx).to_numpy()
    front = KEY + ["primary_metric", "verdict_if_higher_better", "reason_if_higher_better",
                   "verdict_if_lower_better", "reason_if_lower_better"]
    return out[front + [c for c in out.columns if c not in front]]


def pick_cases(b: dict) -> dict:
    """The seven arms the memo writes up, picked by rule from the
    "higher is better" verdicts (ties broken by experiment and arm id)."""
    up, table = b["up"].sort_values(KEY), b["table"]
    launch = up[up["verdict"] == "launch"]
    guard = up[up["reason"] == "guard rail worse"].copy()
    tested = table[table["reading"].isin(D.READ) & (table["metric_id"] != D.PRIMARY)]
    worst = tested[tested["lift"] < 0].groupby(KEY)["p"].min()
    guard["guard_p"] = [worst[(e, v)] for e, v in zip(guard["experiment_id"], guard["variant_id"])]
    flat = up[up["verdict"] == "no change"].assign(width=lambda d: d["primary_hi"] - d["primary_lo"])
    wide = up[up["reason"] == "primary too wide to tell"]
    return {
        "clear": launch.loc[launch["primary_p"].idxmin()],            # the strongest launch
        "thin": launch.loc[launch["primary_p_arms"].idxmax()],        # the weakest launch
        "guard": guard.loc[guard["guard_p"].idxmin()],                # flat primary, strongest guard-rail loss
        "jump": up[up["reason"] == "last snapshot jumps away from the one before"].iloc[0],
        "flat": flat.loc[flat["width"].idxmin()],                     # the tightest "no change"
        "wide": wide.loc[wide["primary_mde"].idxmax()],               # the least sensitive test
        "split": up[up["reason"] == "Part 1 warning on the split"].sort_values("primary_p").iloc[0],
    }


def case_facts(b: dict, row: pd.Series) -> dict:
    e, v = row.experiment_id, row.variant_id
    arm = b["arms"].set_index(KEY).loc[(e, v)]
    t = b["table"][(b["table"]["experiment_id"] == e) & (b["table"]["variant_id"] == v)].set_index("metric_id")
    rp = b["replay"][(b["replay"]["experiment_id"] == e) & (b["replay"]["variant_id"] == v)].set_index("metric_id")
    down = b["down"].set_index(KEY).loc[(e, v)]
    pr = t.loc[D.PRIMARY]
    sd_over_mean = math.sqrt(pr.variance_c) / pr.mean_c
    needed = R.users_per_group(sd_over_mean, D.MARGIN)
    had = int(min(pr.users_c, pr.users_t))
    states = {D.PRIMARY: row.primary_state}
    for m, r in t.drop(index=D.PRIMARY).iterrows():
        states[m] = ("not testable" if r.reading == "not testable"
                     else D.guard_state(r.lift, r.lift_lo, r.lift_hi, r.p))
    sig = b["table"][b["table"]["reading"].isin(D.READ) & (b["table"]["p"] < D.ALPHA)]
    return {"row": row, "arm": arm, "metrics": t, "states": states, "down": down,
            "largest_significant": float(sig["lift"].abs().max()),
            "longest_test": float(b["arms"]["last_day"].max()),
            "users_gap": int(arm.users_c - arm.users_t),
            "replayed": D.PRIMARY in rp.index,
            "av": rp.loc[D.PRIMARY] if D.PRIMARY in rp.index else None,
            "blips": int(rp["blips"].sum()) if len(rp) else 0,
            "users_needed": needed, "users_had": had,
            "days_needed": float(arm.last_day * needed / had)}


def facts(b: dict) -> dict:
    up, down, table = b["up"], b["down"], b["table"]
    dec = decisions_table(b)
    f = {"arms": len(up), "up": counts(up), "down": counts(down),
         "up_reasons": up["reason"].value_counts().to_dict(),
         "down_reasons": down["reason"].value_counts().to_dict(),
         "same_both_directions": int((up["verdict"] == down["verdict"]).sum()),
         "decisions": dec}
    f["decided"] = {k: sum(c[v] for v in ("launch", "do not launch", "no change"))
                    for k, c in (("up", f["up"]), ("down", f["down"]))}
    # launches that turn into "do not launch" when the direction is turned
    f["launch_flips"] = int(((up["verdict"] == "launch") & (down["verdict"] == "do not launch")).sum())

    # the funnel: from "primary p < 0.05 and higher" to "launch"
    prim = table[table["metric_id"] == D.PRIMARY].set_index(KEY)
    naive = (prim["p"] < D.ALPHA) & (prim["lift"] > 0)
    u = up.set_index(KEY)
    in_naive = u[naive.reindex(u.index).to_numpy()]
    f["funnel"] = {
        "naive": int(naive.sum()),
        "no_decision": int((in_naive["verdict"] == "no decision").sum()),
        "check": int((in_naive["verdict"] == "check the data").sum()),
        "unconfirmed": int((in_naive["primary_state"] == "unconfirmed").sum()),
        "guard": int(((in_naive["primary_state"] == "better")
                      & ~in_naive["verdict"].isin(["launch", "check the data"])).sum()),
        "launch": int((in_naive["verdict"] == "launch").sum()),
    }
    # the "any metric moved" rule Part 2 warned about, on the arms Part 2 read
    read = table[table["reading"].isin(D.READ)]
    any_up = read[(read["p"] < D.ALPHA) & (read["lift"] > 0)].groupby(KEY).ngroups
    f["any_metric_up"] = any_up
    f["read_arms"] = read.groupby(KEY).ngroups

    # guard rails
    better = up[up["primary_state"] == "better"]
    f["guards"] = {
        "better_arms": len(better),
        "better_with_worse": int((better["guards_worse"] != "").sum()),
        "better_not_cleared": int((better["guards_not_cleared"] != "").sum()),
        "flat_with_worse": int((up["reason"] == "guard rail worse").sum()),
        "flat_with_worse_down": int((down["reason"] == "guard rail worse").sum()),
        "primary_worse": int((up["reason"] == "primary worse").sum()),
        "primary_worse_down": int((down["reason"] == "primary worse").sum()),
    }
    gw = up[up["reason"] == "guard rail worse"]
    f["guards"]["flat_with_worse_primary_no_change"] = int((gw["primary_state"] == "no change").sum())

    # launches against Part 3's always-valid rule
    la = dec[dec["verdict_if_higher_better"] == "launch"]
    f["launch_check"] = {
        "launches": len(la), "replayed": int(la["replayed"].sum()),
        "confirmed": int((la["always_valid_confirmed"] == True).sum()),      # noqa: E712
        "not_confirmed": int((la["always_valid_confirmed"] == False).sum()),  # noqa: E712
        "p_max": float(la["primary_p_arms"].max()),
        "lift_min": float(la["primary_lift"].min()), "lift_max": float(la["primary_lift"].max()),
        "lift_median": float(la["primary_lift"].median()),
        "under_margin": int((la["primary_lift"] < D.MARGIN).sum()),
    }
    conf = la[la["always_valid_confirmed"] == True]                           # noqa: E712
    f["launch_check"]["confirmed_day_median"] = float(conf["always_valid_day"].median())

    # the end-of-test snapshot check, now on every test Part 2 read
    ends = b["ends"]
    f["ends"] = {"tests": len(ends), "jumps": int(ends["end_jump"].sum()),
                 "replayed_tests": len(b["replay"]),
                 "new_tests": len(ends) - len(b["replay"]),
                 "jump_rows": ends[ends["end_jump"]]}

    # sensitivity
    per = b["per_primary"]
    verdicts = pd.DataFrame({m: d.set_index(KEY)["verdict"] for m, d in per.items()})
    f["same_any_primary"] = int((verdicts.nunique(axis=1) == 1).sum())
    decided = ~verdicts[1].isin(["no decision", "check the data"])
    f["decided_arms"] = int(decided.sum())
    f["same_any_primary_decided"] = int((verdicts[decided].nunique(axis=1) == 1).sum())
    f["launch_any_primary"] = int((verdicts == "launch").any(axis=1).sum())
    f["launch_every_primary"] = int((verdicts == "launch").all(axis=1).sum())
    f["by_primary"] = b["by_primary"]
    f["by_margin"] = b["by_margin"]

    f["cases"] = {k: case_facts(b, r) for k, r in pick_cases(b).items()}
    return f


# ------------------------------------------------------------------ charts
def chart_verdicts(f: dict, path: Path) -> None:
    """How the 99 arms are decided under each direction."""
    order = list(D.VERDICTS)
    y = np.arange(len(order))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for off, key, color, label in ((h / 2, "up", BLUE, "if higher is better"),
                                   (-h / 2, "down", ORANGE, "if lower is better")):
        vals = [f[key][v] for v in order]
        ax.barh(y + off, vals, height=h, color=color, label=label)
        for yi, val in zip(y + off, vals):
            ax.text(val + 0.6, yi, str(val), va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [v[0].upper() + v[1:] for v in order])
    ax.set_xlim(0, max(max(f["up"].values()), max(f["down"].values())) + 6)
    ax.set_xlabel(f"Treatment arms, of {f['arms']}")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title(f"A launch for {f['up']['launch']} of {f['arms']} arms if higher is better, {f['down']['launch']} if lower is",
                 loc="left", fontsize=12, fontweight="bold", pad=40)
    ax.text(0, 1.085, f"Primary: metric {D.PRIMARY}; metrics 2 to 4 are guard rails. The source does not say which "
            "direction is good, so both are shown.", transform=ax.transAxes, fontsize=9, color=INK_2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.99), ncol=2, frameon=False, fontsize=8.5,
              handlelength=1.2, columnspacing=1.4, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(path, **SAVE)
    plt.close(fig)


def chart_cases(f: dict, path: Path) -> None:
    """The seven memo cases: change and 95% interval for each metric."""
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 5.9))
    flat_axes = axes.ravel()
    for ax, key in zip(flat_axes, CASES):
        c = f["cases"][key]
        t, row = c["metrics"], c["row"]
        lo_all, hi_all = [], []
        for m in (1, 2, 3, 4):
            if m not in t.index or pd.isna(t.loc[m, "lift_lo"]):
                if m in t.index:
                    ax.text(0, m, "no variance", va="center", ha="center", fontsize=8, color=MUTED)
                continue
            r = t.loc[m]
            state = c["states"][m]
            color = ORANGE if state == "worse" else (BLUE if m == D.PRIMARY else INK_2)
            ax.plot([100 * r.lift_lo, 100 * r.lift_hi], [m, m], color=color, linewidth=2.2,
                    solid_capstyle="butt")
            ax.scatter([100 * r.lift], [m], s=34, color=color, zorder=3,
                       marker="D" if m == D.PRIMARY else "o")
            lo_all.append(100 * r.lift_lo)
            hi_all.append(100 * r.lift_hi)
        span = max(1.35, 1.12 * max(abs(min(lo_all)), abs(max(hi_all))))
        ax.axvspan(-100 * D.MARGIN, 100 * D.MARGIN, color=GRID, alpha=0.6, linewidth=0)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_xlim(-span, span)
        ax.set_ylim(4.6, 0.4)
        ax.set_yticks([1, 2, 3, 4], ["Metric 1\n(primary)", "Metric 2", "Metric 3", "Metric 4"], fontsize=8.5)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:+g}%" if x else "0")
        ax.tick_params(length=0, labelsize=8.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title(f"{row.experiment_id} / {row.variant_id}\n{row.verdict[0].upper() + row.verdict[1:]}",
                     loc="left", fontsize=10, fontweight="bold")
    legend = flat_axes[-1]
    legend.axis("off")
    items = [(BLUE, "D", "Primary metric"), (INK_2, "o", "Guard rail"),
             (ORANGE, "o", "Significantly lower (p < 0.05)")]
    for i, (color, marker, label) in enumerate(items):
        yy = 0.86 - 0.13 * i
        legend.plot([0.02, 0.16], [yy, yy], color=color, linewidth=2.2, transform=legend.transAxes,
                    solid_capstyle="butt")
        legend.scatter([0.09], [yy], s=34, color=color, marker=marker, transform=legend.transAxes, zorder=3)
        legend.text(0.22, yy, label, transform=legend.transAxes, va="center", fontsize=9, color=INK)
    legend.add_patch(plt.Rectangle((0.02, 0.42), 0.14, 0.07, transform=legend.transAxes, color=GRID, alpha=0.6,
                                   linewidth=0))
    legend.text(0.22, 0.455, f"Within ±{100 * D.MARGIN:g}% (the margin)", transform=legend.transAxes,
                va="center", fontsize=9, color=INK)
    legend.text(0.02, 0.2, "Point: relative change at the\nlast snapshot. Line: 95% interval.\n"
                "Verdicts assume higher is better.\nThe scale differs between panels.",
                transform=legend.transAxes, va="center", fontsize=8.5, color=INK_2, linespacing=1.4)
    fig.suptitle("Seven experiments, seven different answers", x=0.012, ha="left", fontsize=12.5,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.985), w_pad=1.6, h_pad=1.8)
    fig.savefig(path, **SAVE)
    plt.close(fig)


# ------------------------------------------------------------------ report
def interval(lo: float, hi: float) -> str:
    return f"{pc(lo)} to {pc(hi)}"


def metric_table(c: dict) -> str:
    rows = []
    for m, r in c["metrics"].iterrows():
        role = "primary" if m == D.PRIMARY else "guard rail"
        if pd.isna(r.lift_lo):
            rows.append((f"{m}: {METRIC_TYPE[m]}", role, pc(r.lift), "no variance", "n/a", "not testable"))
            continue
        rows.append((f"{m}: {METRIC_TYPE[m]}", role, pc(r.lift), interval(r.lift_lo, r.lift_hi),
                     p_text(r.p), c["states"][m]))
    return md_table(pd.DataFrame(rows, columns=["Metric", "Role", "Change", "95% interval", "p", "Reads as"]),
                    left=("Metric", "Role", "Reads as"))


def evidence_lines(c: dict) -> str:
    arm, row = c["arm"], c["row"]
    lines = [f"- Users: {n(arm.users_c)} control, {n(arm.users_t)} treatment "
             f"(treatment share {100 * arm.treatment_share:.2f}%). Ran {arm.last_day:g} days. "
             f"Treatment arms in this experiment: {arm.arms_in_experiment}.",
             f"- Part 1 cautions: {arm.cautions if arm.cautions else 'none'}."]
    if c["replayed"]:
        av = c["av"]
        if av.always_confirmed_stops:
            lines.append(f"- Daily looks (Part 3): the always-valid rule, confirmed on the next look, called the "
                         f"primary on day {av.always_confirmed_day:g}, when the change read {pc(av.always_confirmed_lift)}.")
        else:
            lines.append("- Daily looks (Part 3): the always-valid rule never called the primary.")
        if c["blips"]:
            lines.append(f"- Blips in the daily record (all metrics): {c['blips']}.")
    else:
        lines.append("- Daily looks (Part 3): not replayed, the daily record is incomplete.")
    if row.end_jump_metrics:
        lines.append(f"- Last snapshot: metric {row.end_jump_metrics} jumps by more than {P.BLIP_Z:g} standard "
                     "errors from the look before it.")
    return "\n".join(lines)


def other_direction(c: dict) -> str:
    d = c["down"]
    if d.verdict == c["row"].verdict:
        return f"**If lower is better:** the same verdict ({d.verdict})."
    return f"**If lower is better:** {d.verdict} ({d.reason})."


def case_text(key: str, c: dict) -> str:
    """The recommendation paragraph for one memo case. Every figure comes
    from the case's own rows; the wording is fixed per kind of case."""
    row, t, arm = c["row"], c["metrics"], c["arm"]
    pr = t.loc[D.PRIMARY]
    name = arm_name(row.experiment_id, row.variant_id)
    mg = f"{100 * D.MARGIN:g}%"
    if key == "clear":
        av = c["av"]
        return (f"**Recommendation: launch.** The primary is up {pc(pr.lift)} ({interval(pr.lift_lo, pr.lift_hi)}), "
                f"and all three guard rails moved the same way. The split is clean and the test ran "
                f"{arm.last_day:g} days. A team looking every day under the always-valid rule would have had this "
                f"answer on day {av.always_confirmed_day:g}.\n\n"
                f"**Plan on** the interval, not the point: a gain between {pc(pr.lift_lo)} and {pc(pr.lift_hi)}.\n\n"
                f"**What would change this:** nothing in the data. The open question is the one the data cannot "
                f"answer: what metric {D.PRIMARY} is.")
    if key == "thin":
        return (f"**Recommendation: launch only if the end date was fixed before the test started.** The primary is up "
                f"{pc(pr.lift)} with p = {p_text(pr.p)}: it passes the rule, narrowly, and the lower end of the "
                f"interval ({pc(pr.lift_lo)}) is close to no gain at all. The guard rails are cleared. "
                f"The always-valid rule never called this result in {arm.last_day:g} days of daily looks, so if "
                f"the test ended when it did because someone liked the number that day, p = {p_text(pr.p)} is not "
                f"evidence (Part 3).\n\n"
                f"**Plan on** a gain that may be close to zero. This test could reliably detect a change of "
                f"{100 * pr.mde:.2f}%; the one it reports is {100 * abs(pr.lift):.2f}%.\n\n"
                f"**What would change this:** a planned end date on record turns this into a launch. Without one, "
                f"repeat the test.")
    if key == "guard":
        worse = [m for m, s in c["states"].items() if s == "worse"]
        parts = ", ".join(f"metric {m} is down {pc(t.loc[m, 'lift'])[1:]} (p = {p_text(t.loc[m, 'p'])})" for m in worse)
        return (f"**Recommendation: do not launch.** On the primary alone this is a quiet test: {pc(pr.lift)}, "
                f"p = {p_text(pr.p)}, interval inside ±{mg}. A readout that stopped there would file it as "
                f"\"no change\". The guard rails say otherwise: {parts}. "
                f"A change that leaves the primary where it was and lowers "
                f"{'two other metrics' if len(worse) == 2 else 'another metric' if len(worse) == 1 else 'three other metrics'} "
                f"is a loss.\n\n"
                f"**What would change this:** learning that lower is better for those metrics (see below), or "
                f"that they do not matter to this product area.")
    if key == "jump":
        m = int(str(row.end_jump_metrics).split(", ")[0])
        r = t.loc[m]
        return (f"**Recommendation: do not decide yet. Ask the data team about the last snapshot.** This is the second "
                f"treatment of the experiment in the case before. Its primary is flat ({pc(pr.lift)}, "
                f"p = {p_text(pr.p)}), and metric {m} ends {pc(r.lift)} with p = {p_text(r.p)}, which on its own "
                f"would be a clear \"do not launch\". That figure comes from a final snapshot that moved by more "
                f"than {P.BLIP_Z:g} standard errors from the day before. One more day of users almost never moves "
                f"a test of this size that far (Part 3's simulation); Part 3 found {c['blips']} blip{'s' if c['blips'] != 1 else ''} elsewhere "
                f"in this arm's record. No later snapshot exists to show whether the figure jumps back.\n\n"
                f"**What would change this:** one more snapshot, or the pipeline's log for the last day. If the "
                f"figure holds, the verdict is \"do not launch\", as for the other treatment. Either way the "
                f"treatment is not launched today.")
    if key == "flat":
        return (f"**Recommendation: do not launch for the metrics; the test found no change.** The primary moved "
                f"{pc(pr.lift)} and its interval ({interval(pr.lift_lo, pr.lift_hi)}) rules out a change of {mg} "
                f"in either direction. Every guard rail is cleared. With {n(arm.users)} users this is a finding, not a "
                f"shrug: the idea does not move metric {D.PRIMARY} by an amount worth having.\n\n"
                f"**What would change this:** a reason to launch that is not a metric (lower cost, simpler code, "
                f"a legal requirement). The test says such a launch would be safe to within {mg}.")
    if key == "wide":
        return (f"**Recommendation: the test cannot tell, and running it longer is not realistic.** The primary reads {pc(pr.lift)}, "
                f"but the interval runs from {pc(pr.lift_lo)} to {pc(pr.lift_hi)}: a gain of several percent "
                f"and a loss of more than {mg} both fit. The smallest change this test could reliably detect is "
                f"{100 * pr.mde:.1f}%, more than the largest significant change in any test here "
                f"({100 * c['largest_significant']:.2f}%, Part 2).\n\n"
                f"**What it would take:** about {n(c['users_needed'])} users per group to detect a {mg} change; "
                f"the smaller group had {n(c['users_had'])}. At the rate this test ran, that is roughly "
                f"{c['days_needed']:,.0f} days; the longest test in this dataset ran {c['longest_test']:g}. So either the change being "
                f"tested has to be one that could plausibly move the metric by {100 * pr.mde:.0f}% or more, or the "
                f"decision has to be made on other grounds, and said to be.\n\n"
                f"**What would change this:** only more users.")
    if key == "split":
        return (f"**Recommendation: do not decide. Find out why the split is off, then rerun.** The primary is up "
                f"{pc(pr.lift)} with p = {p_text(pr.p)}, which would be good news in a sound test. Part 1 found "
                f"the treatment group {n(c['users_gap'])} users smaller than the control group (treatment share "
                f"{100 * arm.treatment_share:.3f}%, p = {p_text(arm.srm_p)} against an even split) and a split "
                f"that drifts from week to week. That gap is {100 * c['users_gap'] / arm.users_t:.2f}% of the "
                f"treatment group. If the missing users were all ones with a \"no\" on metric {D.PRIMARY}, "
                f"their absence alone would raise the treatment's figure by {100 * c['users_gap'] / arm.users_t:.2f}%: "
                f"as much as the whole result.\n\n"
                f"**What would change this:** an explanation for the missing users that has nothing to do with "
                f"how they behaved. Without one, the result is not evidence.")
    raise KeyError(key)


def count_of(k: int, one: str, many: str) -> str:
    """'1 arm is' / '3 arms are' / 'None is'."""
    return f"None {one}" if k == 0 else f"{k} {one if k == 1 else many}"


def render(b: dict, f: dict) -> str:
    up, dn, a, mg = f["up"], f["down"], D.ALPHA, f"{100 * D.MARGIN:g}%"
    fn, g, lc, e = f["funnel"], f["guards"], f["launch_check"], f["ends"]
    cases = f["cases"]

    count_t = pd.DataFrame({
        "Verdict": [v[0].upper() + v[1:] for v in D.VERDICTS],
        "What it means": ["Primary better, every guard rail cleared, data sound",
                          "Primary worse, or a guard rail worse",
                          f"Primary inside ±{mg} and no guard rail worse",
                          "The test cannot tell: too small, or a guard rail not cleared",
                          "A split warning (Part 1) or a suspect last snapshot (Part 3)",
                          "Unexplained split or under a week of data (Part 1)"],
        "If higher is better": [up[v] for v in D.VERDICTS],
        "If lower is better": [dn[v] for v in D.VERDICTS],
    })
    case_t = pd.DataFrame([
        (i + 1, CASE_TITLE[k], arm_name(cases[k]["row"].experiment_id, cases[k]["row"].variant_id),
         cases[k]["row"].verdict, cases[k]["down"].verdict)
        for i, k in enumerate(CASES)],
        columns=["#", "Case", "Arm", "If higher is better", "If lower is better"])
    prim_t = pd.DataFrame({"Primary metric": [f"{m}: {METRIC_TYPE[m]}" for m in f["by_primary"]["primary_metric"]],
                           **{v[0].upper() + v[1:]: f["by_primary"][v] for v in D.VERDICTS}})
    marg_t = pd.DataFrame({"Margin": [f"±{100 * m:g}%" for m in f["by_margin"]["margin"]],
                           **{v[0].upper() + v[1:]: f["by_margin"][v] for v in D.VERDICTS}})
    jr = e["jump_rows"].iloc[0]

    sections = []
    for i, k in enumerate(CASES):
        c = cases[k]
        row = c["row"]
        sections.append(f"""### {i + 1}. {CASE_TITLE[k]}: {arm_name(row.experiment_id, row.variant_id)}

{case_text(k, c)}

{other_direction(c)}

{metric_table(c)}

{evidence_lines(c)}
""")
    picked = (
        f"the launch with the smallest primary p-value; the launch with the largest; the \"guard rail worse\" arm "
        f"with the smallest guard-rail p-value; the arm whose last snapshot jumps; the \"no change\" arm with the "
        f"narrowest primary interval; the \"too wide\" arm with the largest detectable change; and the arm with a "
        f"split warning and the smallest primary p-value")

    return f"""# Part 4: the decision memo

*Generated by `scripts/decision_report.py` from the committed source file. Every number below is computed by the script; the wording of each recommendation is fixed per kind of case and filled in from that case's own figures.*

**To:** the product team. **Question:** which of these treatments do we launch?

**Short answer:** of {f['arms']} treatments tested, **{up['launch']} pass every rule for a launch** if a higher metric {D.PRIMARY} is better ({dn['launch']} if lower is better). {up['do not launch']} should not be launched, {up['no change']} changed nothing worth having, {up['run longer']} could not tell, and {up['check the data'] + up['no decision']} are not decided: {up['no decision']} for an unexplained split or too short a run, {up['check the data']} held for a data check. The source does not say what the metrics are or which direction is good, so every verdict below is given both ways; {f['same_both_directions']} of the {f['arms']} verdicts are the same either way.

![Verdicts under both directions](charts/10_verdicts.png)

## 1. The rules

One metric is the **primary**: the one the test was run to move, named before the test. The others are **guard rails**: nobody expects them to improve, but a launch must not make them worse. Here the primary is metric {D.PRIMARY} ({METRIC_TYPE[D.PRIMARY]}) and metrics 2 to 4 are guard rails. That is a working choice, since the source names nothing; section 5 shows what changes with another choice.

Applied to each treatment arm in this order ([`abtest/decision.py`](../abtest/decision.py)):

1. **No decision** if Part 1 could not explain the split or the test ran under a week.
2. **Check the data** if Part 1 warned about the split, or the last snapshot of any metric jumps by more than {P.BLIP_Z:g} standard errors from the look before it (Part 3).
3. **Do not launch** if the primary is significantly worse (p < {a}), or any guard rail is.
4. **Launch** if the primary is significantly better, also after Holm's correction across the treatment arms of the experiment, and every guard rail's 95% interval rules out a loss of {mg} or more.
5. **Run longer** if the primary is better but a guard rail is not cleared or cannot be tested.
6. **No change** if the primary's interval lies inside ±{mg}.
7. **Run longer** otherwise: the test could not tell.

Three choices in those rules are deliberate:

- **Good news has a higher bar than bad news.** A launch needs the primary to survive a correction for the number of arms; an uncorrected p < {a} in the wrong direction on any metric is enough to stop one. A false alarm on a guard rail costs a delayed launch. A missed one costs a loss that ships.
- **A guard rail is cleared by its interval, not by "not significant".** "Not significant" is also what a test too small to see anything reports. A guard rail is cleared only when its interval rules out a loss of {mg}.
- **The primary is one metric, so it needs no correction across metrics.** That is what naming it in advance buys (Part 2).

## 2. What the rules decide

{md_table(count_t, left=("Verdict", "What it means"))}

Every arm and its verdict both ways: [`decisions.csv`](decisions.csv).

**From "significant" to "launch".** Taking higher as better:

- {fn['naive']} of {f['arms']} arms end with the primary higher at p < {a}. That is the list a dashboard would show.
- {count_of(fn['no_decision'], 'is in an arm', 'are in arms')} Part 1 would not decide on (an unexplained split or under a week of data), and {count_of(fn['check'], 'is', 'are').lower()} held for a data check.
- {count_of(fn['unconfirmed'], 'is one of several treatments in its experiment and does', 'are one of several treatments in their experiment and do')} not survive the correction for that.
- {count_of(fn['guard'], 'is', 'are')} stopped by a guard rail.
- **{fn['launch']} are left.** Their gains run from {pc(lc['lift_min'])} to {pc(lc['lift_max'])}, with a median of {pc(lc['lift_median'])}; {lc['under_margin']} of the {lc['launches']} are under {mg}.

The rule Part 2 warned about, "launch if any of the four metrics is significantly higher", would pick {f['any_metric_up']} of the {f['read_arms']} arms Part 2 read.

**What the guard rails did.** They never stopped a launch: in all {g['better_arms']} arms where the primary is better (the {up['launch']} launches and {g['better_arms'] - up['launch']} held for a data check), no guard rail is worse and every one is cleared. These four metrics move together (Part 2), so a treatment that lifts one rarely lowers another. The guard rails earned their place among the quiet tests instead:

- **{g['flat_with_worse']} arms have a primary that is not significantly different and a guard rail that is significantly worse.** In {g['flat_with_worse_primary_no_change']} of them the primary's interval is inside ±{mg}, so a readout of the primary alone would say "no change, safe to launch if there is another reason to". The guard rails say it is not safe.
- Together with the {g['primary_worse']} arms whose primary is worse, that makes {up['do not launch']} "do not launch" verdicts. A team reading only the primary would have seen {g['primary_worse']} of them.
- If lower is better the picture is the same with different arms: {g['primary_worse_down']} with a worse primary, {g['flat_with_worse_down']} with a worse guard rail.

**How solid are the {lc['launches']} launches?** The rules above read each test once, at its end. That is sound if the end date was fixed before the test; the source does not say whether it was. Part 3's always-valid rule is the check that does not depend on that:

- {lc['replayed']} of the {lc['launches']} have a complete daily record. The always-valid rule, confirmed on the next look, called **{lc['confirmed']}** of them, on a median of day {lc['confirmed_day_median']:g}.
- The confirmed rule did not call the other {lc['not_confirmed']}. Those {lc['not_confirmed']} are launches only if their end date was planned (case 2 below is one).
- The remaining {lc['launches'] - lc['replayed']} could not be replayed (their daily record is incomplete).

**The last-snapshot check, on every test.** Part 3 ran its check for a suspect final snapshot only on the {e['replayed_tests']} tests it replayed. A decision needs it everywhere, so it is run here on all {e['tests']} tests Part 2 read ({e['new_tests']} more). It finds {e['jumps']}: {arm_name(jr.experiment_id, jr.variant_id)}, metric {jr.metric_id}, the one Part 3 had already found. No other final snapshot is suspect by this rule.

## 3. Seven decisions, written out

The seven arms are picked by rule, not by eye, from the "higher is better" verdicts: {picked}.

{md_table(case_t, left=("Case", "Arm", "If higher is better", "If lower is better"))}

![The seven cases](charts/11_memo_cases.png)

{chr(10).join(sections)}
## 4. Which direction is good?

The source does not say, and it decides the answer.

- **{f['same_both_directions']} of {f['arms']} verdicts do not depend on it:** every "no decision" and "check the data", and the tests in which no metric is significantly different.
- **All {f['launch_flips']} launches become "do not launch"** if lower is better, because the metric that rose is then the metric that got worse.
- The verdicts apply one direction to all four metrics. If the primary were better higher and a guard rail better lower (a cost, a return rate), a treatment that raises everything would be a launch with a guard-rail loss. Since these metrics move together, that would happen often here. Only the metric definitions can settle it.

## 5. Does the choice of primary or margin matter?

Verdicts if each metric in turn were the primary (higher is better; the other three are guard rails):

{md_table(prim_t)}

- Of the {f['decided_arms']} arms that get a verdict on the metrics with metric 1 as primary, {f['same_any_primary_decided']} get the same verdict whichever metric is the primary.
- {f['launch_any_primary']} arms would be a launch under at least one choice of primary, {f['launch_every_primary']} under every choice. A team that picks the primary after seeing the results has the larger number to choose from.
- The "do not launch" count does not move, because a metric that is significantly worse stops a launch whether it is the primary or a guard rail.
- With metrics 2 to 4 as primary, {int(f['by_primary']['no decision'].iloc[1] - f['by_primary']['no decision'].iloc[0])} more arms get no decision: the source has no variance for those metrics in those arms.

Verdicts for different margins (primary metric {D.PRIMARY}, higher is better):

{md_table(marg_t)}

The margin moves arms between "no change" and "run longer" and leaves the rest almost untouched. It is a business number, the largest loss a launch may risk and the smallest gain worth having, and it should be set before the test, like the primary.

## 6. So what, for a team that runs experiments

1. **Write the decision rule before the test.** Primary metric, direction, guard rails, margin, end date. Every one of those is missing from this dataset, and each one changes verdicts: the direction changes {f['arms'] - f['same_both_directions']} of {f['arms']}, the choice of primary changes {f['decided_arms'] - f['same_any_primary_decided']} of {f['decided_arms']}.
2. **"Significant" is where the decision starts.** {fn['naive']} arms show a significantly higher primary; {fn['launch']} are launches. The rest fall at the steps in between (section 2).
3. **Guard rails matter most when the primary is quiet.** {g['flat_with_worse']} arms looked like "nothing happened" on the primary and had a significantly worse guard rail. Read every guard rail on every test, including the ones nobody is excited about.
4. **Separate "no change" from "cannot tell".** {up['no change']} tests rule out a change of {mg}: a finding, and the idea can be closed. {up['run longer']} cannot tell, and for the smallest of them no realistic amount of extra time would help (case 6). Check the detectable change before starting a test.
5. **A verdict is only as good as the last snapshot.** One strong result here rests on a final figure that jumped (case 4). Hold the decision one day when the last look disagrees with the one before.
6. **A launch decided on a fixed end date and a launch decided by daily looks are different claims.** {lc['confirmed']} of the {lc['replayed']} launches with a daily record hold under a rule built for daily looks; {lc['not_confirmed']} depend on the end date having been planned. Record which it was.

## 7. Limits

- The primary (metric {D.PRIMARY}), the direction and the {mg} margin are working choices made for this exercise. The source publishes none of them, so no verdict here is a statement about what ASOS did or should have done.
- One direction is applied to all four metrics; mixed directions are not modelled (section 4).
- The verdicts read each test once, at its last snapshot, as if the end date had been fixed in advance. Section 2 says which launches also hold under daily looks.
- Guard rails are checked at an uncorrected p < {a}, three per arm, so some "guard rail worse" verdicts are false alarms: up to {100 * R.any_false_alarm(3):.0f}% of arms would show one if nothing changed and the metrics were independent. That is the price of a low bar for bad news.
- "Run longer" assumes the test could be extended. The source gives no dates, traffic plans or costs.
- The last-snapshot check cannot judge an arm's first look, misses smaller faults, and was tuned on this data (Part 3).
- The metrics are cumulative and unnamed, so no change can be given a money value, and a gain at the end of a test may not be the gain after a launch.
- The seven cases are chosen to show seven kinds of answer. They are not a sample of the {f['arms']} arms.
"""


def write_outputs(b: dict, f: dict, results: Path = RESULTS, charts: Path = CHARTS) -> None:
    kw = {"index": False, "float_format": "%.6g", "lineterminator": "\n"}
    f["decisions"].to_csv(results / "decisions.csv", **kw)
    b["by_primary"].to_csv(results / "decisions_by_primary.csv", **kw)
    b["by_margin"].to_csv(results / "decisions_by_margin.csv", **kw)
    chart_verdicts(f, charts / "10_verdicts.png")
    chart_cases(f, charts / "11_memo_cases.png")
    (results / "DECISION_MEMO.md").write_text(render(b, f))


def main() -> int:
    b = build()
    f = facts(b)
    write_outputs(b, f)
    print(f"{f['arms']} arms; if higher is better: " + ", ".join(f"{v} {f['up'][v]}" for v in D.VERDICTS)
          + f"; same verdict both ways: {f['same_both_directions']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
