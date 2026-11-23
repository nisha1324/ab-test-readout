"""Part 2: the readout at each arm's last snapshot.

    python scripts/readout_report.py

Reads the committed CSV in data/raw/ (no download needed) and writes
results/READOUT.md, results/readout.csv, results/readout_by_metric.csv and
charts 04 to 06.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abtest import checks as C, load, readout as R  # noqa: E402
from scripts.data_checks_report import (  # noqa: E402
    BLUE, CHARTS, GRID, INK, INK_2, METRIC_TYPE, MUTED, ORANGE, RESULTS, SAVE, SURFACE,
    arm_name, md_table, n, style)

READ = ("read", "read with warning")
CLIP = 0.05   # chart 04 shows relative changes between -5% and +5%
BINS = 20     # chart 06


def pc(x: float, nd: int = 2) -> str:
    """A relative change as a signed percentage."""
    return f"{100 * x:+.{nd}f}%"


def p_text(p: float) -> str:
    if p < 1e-15:
        return "< 1e-15"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ build
def build() -> dict:
    raw = load.read_raw()
    con = load.connect(raw)
    arms = C.check_arms(raw, load.run_sql(con, "arm_summary"))
    metrics = load.run_sql(con, "metric_profile")
    table = R.read_arms(load.run_sql(con, "final_snapshot"), arms)
    return {"arms": arms, "metrics": metrics, "table": table, "by_metric": by_metric(table)}


def read_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["reading"].isin(READ)]


def by_metric(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in read_rows(table).groupby("metric_id"):
        sig = g["p"] < R.ALPHA
        rows.append({
            "metric_id": m, "tests": len(g),
            "significant": int(sig.sum()),
            "up": int((sig & (g["lift"] > 0)).sum()), "down": int((sig & (g["lift"] < 0)).sum()),
            "significant_holm": int((g["p_holm"] < R.ALPHA).sum()),
            "significant_bh": int((g["p_bh"] < R.ALPHA).sum()),
            "median_abs_lift": float(g["lift"].abs().median()),
            "median_abs_lift_significant": float(g.loc[sig, "lift"].abs().median()),
            "mde_min": float(g["mde"].min()), "mde_median": float(g["mde"].median()),
            "mde_max": float(g["mde"].max()),
            "can_detect_reference": int((g["mde"] <= R.REFERENCE_LIFT).sum()),
        })
    return pd.DataFrame(rows)


def z_correlation(rd: pd.DataFrame, m1: int, m2: int) -> float:
    """Rank correlation, across arms, of the test statistics of two metrics
    (rank, so the few very large effects do not set the figure)."""
    z = rd.assign(z=rd["diff"] / rd["se_diff"]).pivot(
        index=["experiment_id", "variant_id"], columns="metric_id", values="z")[[m1, m2]].dropna()
    return float(z[m1].corr(z[m2], method="spearman"))


def facts(b: dict) -> dict:
    """Every number the report and the README quote, computed once."""
    t, arms = b["table"], b["arms"]
    rd = read_rows(t)
    key = ["experiment_id", "variant_id"]
    sig = rd[rd["p"] < R.ALPHA]
    quiet = rd[rd["p"] >= R.ALPHA]
    per_arm = rd.assign(s=rd["p"] < R.ALPHA, h=rd["p_holm"] < R.ALPHA, q=rd["p_bh"] < R.ALPHA,
                        up=(rd["p"] < R.ALPHA) & (rd["lift"] > 0),
                        down=(rd["p"] < R.ALPHA) & (rd["lift"] < 0)
                        ).groupby(key).agg(tests=("s", "size"), s=("s", "sum"), h=("h", "sum"),
                                           q=("q", "sum"), up=("up", "sum"), down=("down", "sum"))
    ci_excludes_zero = (rd["lift_lo"] > 0) | (rd["lift_hi"] < 0)
    inside = (quiet["lift_lo"] > -R.REFERENCE_LIFT) & (quiet["lift_hi"] < R.REFERENCE_LIFT)
    warned = rd[rd["reading"] == "read with warning"]
    cv = b["metrics"].set_index("metric_id")["median_sd_over_mean"]
    arm_reading = t.groupby(key)["reading"].agg(lambda s: "not decided" if (s == "not decided").all()
                                                else ("read with warning" if (s == "read with warning").any() else "read"))
    not_decided = arms[arms["cautions"].str.split("; ").apply(lambda i: any(c in i for c in R.NOT_DECIDED))]
    big = sig.loc[sig["lift"].abs().idxmax()]
    return {
        "rows": len(t), "arms": len(arms),
        "reading": t["reading"].value_counts().to_dict(),
        "arms_read": len(per_arm), "experiments_read": rd["experiment_id"].nunique(),
        "arms_not_decided": int((arm_reading == "not decided").sum()),
        "arms_unequal": int(not_decided["cautions"].str.contains("unequal split").sum()),
        "arms_short": int(not_decided["cautions"].str.contains("under a week").sum()),
        "arms_warned": warned.groupby(key).ngroups,
        "arms_untestable": t[t["reading"] == "not testable"].groupby(key).ngroups,
        "tests": len(rd), "significant": len(sig), "quiet": len(quiet),
        "expected_by_chance": R.ALPHA * len(rd),
        "up": int((sig["lift"] > 0).sum()), "down": int((sig["lift"] < 0).sum()),
        "holm": int((rd["p_holm"] < R.ALPHA).sum()), "bh": int((rd["p_bh"] < R.ALPHA).sum()),
        "sig_abs_median": float(sig["lift"].abs().median()),
        "sig_abs_min": float(sig["lift"].abs().min()), "sig_abs_max": float(sig["lift"].abs().max()),
        "sig_under_reference": int((sig["lift"].abs() < R.REFERENCE_LIFT).sum()),
        "biggest": big,
        "ci_test_disagree": int(((rd["p"] < R.ALPHA) != ci_excludes_zero).sum()),
        "arms_any": int((per_arm["s"] > 0).sum()), "arms_any_holm": int((per_arm["h"] > 0).sum()),
        "arms_any_bh": int((per_arm["q"] > 0).sum()),
        "arms_all": int((per_arm["s"] == per_arm["tests"]).sum()),
        "arms_by_count": per_arm["s"].value_counts().sort_index().to_dict(),
        "arms_mixed_direction": int(((per_arm["up"] > 0) & (per_arm["down"] > 0)).sum()),
        "quiet_inside": int(inside.sum()), "quiet_open": int((~inside).sum()),
        "mde_min": float(rd["mde"].min()), "mde_max": float(rd["mde"].max()),
        "can_detect": int((rd["mde"] <= R.REFERENCE_LIFT).sum()),
        "warned_significant": int((warned["p"] < R.ALPHA).sum()), "warned_tests": len(warned),
        "false_alarm_4": R.any_false_alarm(4),
        "false_alarm_max_family": R.any_false_alarm(int(rd.groupby("experiment_id").size().max())),
        "max_family": int(rd.groupby("experiment_id").size().max()),
        "cv": {int(k): float(v) for k, v in cv.items()},
        "need": {int(k): R.users_per_group(float(v), R.REFERENCE_LIFT) for k, v in cv.items()},
        "need_half": {int(k): R.users_per_group(float(v), R.REFERENCE_LIFT / 2) for k, v in cv.items()},
        "multiplier": R.power_multiplier(),
        "per_arm": per_arm,
        "min_group": int(rd[["users_c", "users_t"]].min().min()),
        "z_corr_1_4": z_correlation(rd, 1, 4),
        "z_corr_min": min(z_correlation(rd, i, j) for i in (1, 2, 3) for j in range(i + 1, 5)),
    }


# ------------------------------------------------------------------ charts
def chart_effects(table: pd.DataFrame, path: Path) -> None:
    """Every test read, sorted by relative change, with its confidence interval."""
    rd = read_rows(table)
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 4.4), sharey=True)
    groups = (("Not significant", MUTED, lambda g: g["p"] >= R.ALPHA),
              ("Significant only before correction", ORANGE, lambda g: (g["p"] < R.ALPHA) & (g["p_holm"] >= R.ALPHA)),
              ("Significant after correction", BLUE, lambda g: g["p_holm"] < R.ALPHA))
    clipped = 0
    for ax, m in zip(axes, (1, 2, 3, 4)):
        g = rd[rd["metric_id"] == m].sort_values("lift").reset_index(drop=True)
        clipped += int(((g["lift_lo"] < -CLIP) | (g["lift_hi"] > CLIP)).sum())
        for label, colour, pick in groups:
            d = g[pick(g)]
            ax.vlines(d.index, 100 * d["lift_lo"].clip(-CLIP, CLIP), 100 * d["lift_hi"].clip(-CLIP, CLIP),
                      color=colour, linewidth=1.3, label=label)
            ax.scatter(d.index, 100 * d["lift"], s=7, color=colour, zorder=3)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        style(ax)
        ax.set_title(f"Metric {m}\n{METRIC_TYPE[m]}", loc="left", fontsize=9.5, color=INK)
        ax.set_xticks([])
        ax.set_xlabel(f"{len(g)} tests, sorted", fontsize=9)
        ax.set_ylim(-100 * CLIP, 100 * CLIP)
    axes[0].set_ylabel("Treatment against control,\nrelative change (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    n_sig = int((rd["p"] < R.ALPHA).sum())
    n_holm = int((rd["p_holm"] < R.ALPHA).sum())
    fig.suptitle(f"{n_sig} of {len(rd)} tests show a significant change, {n_holm} after correction; most are small",
                 x=0.012, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.012, 0.9, "One line per treatment arm: the relative change at the last snapshot and its 95% confidence interval. "
             f"{clipped} intervals run past ±{100 * CLIP:.0f}% and are cut off.", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(path, **SAVE)
    plt.close(fig)


def chart_mde(table: pd.DataFrame, path: Path) -> None:
    """Smallest detectable change against the size of the test, metrics 1 and 4."""
    rd = read_rows(table)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for m, colour, marker in ((1, BLUE, "o"), (4, ORANGE, "D")):
        g = rd[rd["metric_id"] == m]
        ax.scatter(g["users_c"] + g["users_t"], 100 * g["mde"], s=30, color=colour, marker=marker,
                   edgecolor=SURFACE, linewidth=0.8, label=f"Metric {m}: {METRIC_TYPE[m]}", zorder=3)
    ax.axhline(100 * R.REFERENCE_LIFT, color=INK_2, linewidth=0.9, linestyle=(0, (3, 3)))
    ax.text(rd["users_c"].add(rd["users_t"]).max() * 1.2, 100 * R.REFERENCE_LIFT * 1.08,
            "a 1% change", fontsize=9, color=INK, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.5, 1, 2, 5], ["0.2%", "0.5%", "1%", "2%", "5%"])
    ax.minorticks_off()
    ax.set_xlabel("Users in the comparison (control + treatment, log scale)")
    ax.set_ylabel("Smallest relative change the test\nwould detect 80% of the time (log scale)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    cant = {m: int((rd.loc[rd["metric_id"] == m, "mde"] > R.REFERENCE_LIFT).sum()) for m in (1, 4)}
    tot = {m: int((rd["metric_id"] == m).sum()) for m in (1, 4)}
    style(ax, f"A 1% change was out of reach for {cant[4]} of {tot[4]} arms on metric 4, and {cant[1]} of {tot[1]} on metric 1",
          "One dot per arm and metric at the last snapshot. Dots above the dashed line could not reliably detect a 1% change.")
    fig.tight_layout()
    fig.savefig(path, **SAVE)
    plt.close(fig)


def p_histogram(table: pd.DataFrame) -> np.ndarray:
    return np.histogram(read_rows(table)["p"], bins=BINS, range=(0, 1))[0]


def chart_pvalues(table: pd.DataFrame, path: Path) -> None:
    counts = p_histogram(table)
    total = int(counts.sum())
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    edges = np.linspace(0, 1, BINS + 1)
    ax.bar(edges[:-1] + 0.5 / BINS, counts, width=1 / BINS * 0.9, color=BLUE)
    flat = total / BINS
    ax.axhline(flat, color=INK_2, linewidth=0.9, linestyle=(0, (3, 3)))
    ax.text(0.99, flat + 1.5, f"if no treatment changed anything: about {flat:.0f} per bar",
            fontsize=9, color=INK, ha="right", va="bottom")
    ax.text(edges[1] + 0.012, counts[0], f"{counts[0]} tests\nbelow 0.05", fontsize=9, color=INK, va="top")
    ax.set_xlabel("p-value of the test")
    ax.set_ylabel("Number of tests")
    ax.set_xlim(0, 1)
    style(ax, f"{counts[0]} of {total} tests fall below 0.05, where chance alone would put about {flat:.0f}",
          "Every arm and metric that was read, at its last snapshot. Bars are 0.05 wide.")
    fig.tight_layout()
    fig.savefig(path, **SAVE)
    plt.close(fig)


# ------------------------------------------------------------------ report
def render(b: dict, f: dict) -> str:
    t, bm = b["table"], b["by_metric"]
    rd = read_rows(t)
    a, z = R.ALPHA, R.REFERENCE_LIFT

    reading = pd.DataFrame([
        ("read", "nothing in Part 1 stands in the way", f["reading"].get("read", 0)),
        ("read with warning", "the arm has a sample-ratio mismatch or a drifting split; the result is shown and marked",
         f["reading"].get("read with warning", 0)),
        ("not testable", "the source has no variance for this metric at the last snapshot", f["reading"].get("not testable", 0)),
        ("not decided", "the arm's split is far from even with no published design, or it ran under a week",
         f["reading"].get("not decided", 0)),
    ], columns=["Reading", "When", "Arm × metric rows"])

    res = pd.DataFrame({
        "Metric": [f"{m}: {METRIC_TYPE[m]}" for m in bm["metric_id"]],
        "Tests read": bm["tests"],
        f"Significant (p < {a})": bm["significant"],
        "Treatment higher": bm["up"], "Treatment lower": bm["down"],
        "Median size of a significant change": [f"{100 * v:.2f}%" for v in bm["median_abs_lift_significant"]],
        "Significant after correction": bm["significant_holm"],
    })

    top = rd[rd["p_holm"] < a].assign(size=lambda d: d["lift"].abs()).sort_values("size", ascending=False).head(10)
    top_t = pd.DataFrame({
        "Arm": [arm_name(e, v) for e, v in zip(top["experiment_id"], top["variant_id"])],
        "Metric": top["metric_id"],
        "Users": [n(c + u) for c, u in zip(top["users_c"], top["users_t"])],
        "Relative change": [pc(v) for v in top["lift"]],
        "95% interval": [f"{pc(lo)} to {pc(hi)}" for lo, hi in zip(top["lift_lo"], top["lift_hi"])],
        "p": [p_text(v) for v in top["p"]],
        "p after correction": [p_text(v) for v in top["p_holm"]],
        "Reading": top["reading"],
    })

    size = pd.DataFrame({
        "Metric": [f"{m}: {METRIC_TYPE[m]}" for m in bm["metric_id"]],
        "Smallest": [f"{100 * v:.2f}%" for v in bm["mde_min"]],
        "Median": [f"{100 * v:.2f}%" for v in bm["mde_median"]],
        "Largest": [f"{100 * v:.2f}%" for v in bm["mde_max"]],
        f"Could detect a {100 * z:.0f}% change": [f"{c} of {k}" for c, k in zip(bm["can_detect_reference"], bm["tests"])],
        "Median SD ÷ mean": [f"{f['cv'][m]:.2f}" for m in bm["metric_id"]],
        f"Users per group for {100 * z:.0f}%": [n(f["need"][m]) for m in bm["metric_id"]],
        f"Users per group for {100 * z / 2:.1f}%": [n(f["need_half"][m]) for m in bm["metric_id"]],
    })

    counts = pd.DataFrame({
        "Significant metrics in the arm": list(f["arms_by_count"].keys()),
        "Arms": list(f["arms_by_count"].values()),
    })

    warned = rd[rd["reading"] == "read with warning"]
    w = warned.groupby(["experiment_id", "variant_id"]).agg(
        cautions=("cautions", "first"), tests=("p", "size"),
        sig=("p", lambda s: int((s < a).sum())), holm=("p_holm", lambda s: int((s < a).sum()))).reset_index()
    warn_t = pd.DataFrame({
        "Arm": [arm_name(e, v) for e, v in zip(w["experiment_id"], w["variant_id"])],
        "Part 1 cautions": w["cautions"], "Tests": w["tests"],
        f"Significant (p < {a})": w["sig"], "After correction": w["holm"],
    })
    big = f["biggest"]

    return f"""# Part 2: the readout

*Generated by `scripts/readout_report.py` from the committed source file. Every number below is computed by the script; nothing is typed in by hand.*

Part 1 decided which experiments can be read. This part reads them: for each treatment arm and each of the four metrics, how big the change is, how sure the data is about it, whether the test was big enough to see a change worth having, and what reading four metrics at once does to the chance of a false alarm. Everything uses each arm's last snapshot.

**Headline:** of {f['tests']} tests read in {f['arms_read']} arms, {f['significant']} are significant at the usual 5% level, where chance alone would give about {f['expected_by_chance']:.0f}. {f['holm']} are still significant once each experiment's tests are corrected for being read together. The changes are small: the median significant change is {100 * f['sig_abs_median']:.2f}% of the control mean, and the largest is {100 * f['sig_abs_max']:.1f}%.

**One thing this readout cannot say is which changes are good.** The source does not name the metrics, so a higher value may be a gain (more orders) or a loss (more returns). The report says "higher" and "lower", never "won" and "lost".

## 1. What is read

The dataset has {f['arms']} treatment arms and four metrics, so {f['rows']} arm × metric rows ([`sql/final_snapshot.sql`](../sql/final_snapshot.sql)). Part 1's cautions ([`arm_checks.csv`](arm_checks.csv)) decide how each is treated:

{md_table(reading, left=("Reading", "When"))}

- **{f['arms_not_decided']} arms are not decided on:** {f['arms_unequal']} whose split is far from even and {f['arms_short']} that ran under a week. Their figures are in [`readout.csv`](readout.csv) for completeness and are left out of every count below.
- **{f['arms_read']} arms in {f['experiments_read']} experiments are read**, {f['arms_warned']} of them with a warning. {f['arms_untestable']} of the arms read have no variance for metrics 2 to 4, so only their metric 1 can be tested. That leaves **{f['tests']} tests**.

## 2. How each number is calculated

All from the group-level figures in the source (users, mean, variance per group), in [`abtest/readout.py`](../abtest/readout.py):

- **Relative change** = treatment mean ÷ control mean − 1. A relative figure lets tests with different baselines be compared.
- **95% confidence interval** for the relative change, by the delta method (the standard way to put an interval on a ratio of two means).
- **p-value** from Welch's t-test on the difference in means, which does not assume the two groups have equal variance or equal size. With this many users the test and the interval agree: they disagree on whether a change is significant in {f['ci_test_disagree']} of {f['tests']} tests.
- **Smallest detectable change** = {f['multiplier']:.2f} × the standard error of the difference ÷ the control mean: the true change this test would have flagged {100 * R.POWER:.0f}% of the time at the {100 * a:.0f}% level.
- **Correction for multiple tests:** Holm's method over all the tests of one experiment (every arm and metric read together to make one decision). Section 5 explains why.

## 3. What the tests found

![Relative change and confidence interval for every test](charts/04_effects.png)

{md_table(res)}

- **{f['significant']} of {f['tests']} tests ({100 * f['significant'] / f['tests']:.0f}%) are significant at p < {a}.** If no treatment changed anything, about {f['expected_by_chance']:.0f} would be. So many of these changes are real, and a meaningful share may still be false alarms: section 5.
- **The changes are small.** Among the significant ones the size runs from {100 * f['sig_abs_min']:.2f}% to {100 * f['sig_abs_max']:.2f}% of the control mean, with a median of {100 * f['sig_abs_median']:.2f}%; {f['sig_under_reference']} of the {f['significant']} are under {100 * z:.0f}%. The largest is arm {arm_name(big.experiment_id, big.variant_id)}, metric {big.metric_id}, at {pc(big.lift)}.
- **Direction is split almost evenly:** treatment is higher in {f['up']} of the significant tests and lower in {f['down']}. Without metric names that is all that can be said.
- **Within an arm the metrics agree.** {f['arms_mixed_direction']} arms have one metric significantly higher and another significantly lower. The four metrics move together: across arms, the test statistics of any two metrics have a rank correlation of at least {f['z_corr_min']:.2f}. That matters for section 5.

The ten largest changes that are still significant after correction:

{md_table(top_t, left=("Arm", "Reading"))}

## 4. Was each test big enough?

A test that finds nothing has two possible meanings: there is nothing to find, or the test was too small to see it. The smallest detectable change tells them apart.

![Smallest detectable change against test size](charts/05_detectable_change.png)

Smallest relative change each test would detect {100 * R.POWER:.0f}% of the time, and the users an even two-group test would need for a metric with the median spread:

{md_table(size)}

- **Most of these tests are very sensitive.** {f['can_detect']} of the {f['tests']} tests could detect a {100 * z:.0f}% change, and the best could detect {100 * f['mde_min']:.2f}%. The least sensitive could only detect {100 * f['mde_max']:.1f}%.
- **The noisier the metric, the more users it takes.** Detecting a {100 * z:.0f}% change needs about {n(round(f['need'][1], -3))} users per group for metric 1 and about {n(round(f['need'][4], -3))} for metric 4, {f['need'][4] / f['need'][1]:.1f} times as many. Halving the change to detect quadruples the users needed.
- **What "not significant" means here.** {f['quiet']} tests are not significant. In {f['quiet_inside']} of them the whole confidence interval lies between −{100 * z:.0f}% and +{100 * z:.0f}%: those tests rule out a change of {100 * z:.0f}% or more in either direction, which is a finding. In the other {f['quiet_open']} the interval reaches past {100 * z:.0f}%, so the test is inconclusive and "no effect" would be the wrong reading.

The {100 * z:.0f}% reference is a working choice. A real team would set it from what the change costs to build and run, before the test starts.

## 5. Four metrics at once

![Distribution of p-values](charts/06_p_values.png)

Each test has a {100 * a:.0f}% chance of a false alarm when nothing changed. Reading four metrics gives four chances: up to {100 * f['false_alarm_4']:.1f}% for at least one false alarm in an arm if the metrics were independent. An experiment with several arms has more: the largest family here has {f['max_family']} tests, and up to {100 * f['false_alarm_max_family']:.0f}%. The metrics here move together (section 3), so the true figures are lower, but they are well above 5%.

How many metrics are significant in each of the {f['arms_read']} arms read:

{md_table(counts)}

- **{f['arms_any']} of {f['arms_read']} arms ({100 * f['arms_any'] / f['arms_read']:.0f}%) have at least one significant metric.** A team that calls a test a result whenever any metric moves would report a change for almost half of everything it runs.
- **After Holm's correction within each experiment, {f['arms_any_holm']} arms still have one**, and {f['holm']} of the {f['significant']} significant tests survive. The other {f['significant'] - f['holm']} are the ones to treat as unconfirmed: each could be real, but not on this evidence.
- **A second view gives a similar answer.** The Benjamini-Hochberg method, applied across all {f['tests']} tests to keep false alarms to about {100 * a:.0f}% of the results called significant, keeps {f['bh']} tests in {f['arms_any_bh']} arms.
- **{f['arms_all']} arms are significant on every metric they report.** Those are the clear results.

## 6. Arms read with a warning

{f['arms_warned']} arms have a sample-ratio mismatch or a drifting split (Part 1). Their {f['warned_tests']} tests are included above; {f['warned_significant']} are significant, and {int((warned['p_holm'] < a).sum())} after correction.

{md_table(warn_t, left=("Arm", "Part 1 cautions"))}

A significant result in one of these arms needs the cause of the caution found before anyone acts on it: if the users who went missing differ from those who stayed, the two groups are no longer comparable, and the test cannot show how large that bias is.

## 7. So what, for a team that runs experiments

1. **Expect small changes and size tests for them.** The median significant change here is {100 * f["sig_abs_median"]:.2f}%. The largest is {100 * f['sig_abs_max']:.1f}%, so a test sized to detect only a 5% change is sized for something that did not happen here. Detecting {100 * z:.0f}% takes roughly {n(f['need'][1] / 1e6, 1)} to {n(f['need'][4] / 1e6, 1)} million users per group, depending on the metric.
2. **Pick the deciding metric before the test.** With four metrics, {f['arms_any']} of {f['arms_read']} arms show "something"; after correction {f['arms_any_holm']} do. One primary metric named in advance needs no correction, and the others become guard rails (Part 4).
3. **Report the interval, not just "significant" or "not".** {f['quiet_inside']} of the {f['quiet']} quiet results are tight enough to say the change is under {100 * z:.0f}% either way, which can settle a decision. The other {f['quiet_open']} say only that the test was too small.
4. **Record what each metric is and which direction is good.** This dataset shows what is left without that: {f['up']} changes up and {f['down']} down, and no way to call any of them a win.

## 8. Limits

- Each arm is read once, at its last snapshot. If a team stopped a test because of what the results showed, the p-values here are too small. Part 3 measures that.
- The 5% level, 80% power and the {100 * z:.0f}% reference change are conventions and working choices, set in `abtest/readout.py`.
- Users are treated as independent and the metrics as close to normal on average. The smallest group read has {n(f['min_group'])} users, so the second is safe even for skewed metrics; the first cannot be checked without user-level data.
- Arms of one experiment share a control group, so their results are not independent. Holm's correction is valid under any dependence; the "by chance" figures in section 5 assume independence and are upper ends.
- A significant change found by a test that could barely detect it tends to be overstated. The sizes in section 3 are best read as upper estimates for the less sensitive tests.
- Not being named, the metrics cannot be given a direction, a money value or a guard-rail role. Part 4 sets out decision rules that would apply once they are.
"""


def main() -> int:
    b = build()
    f = facts(b)
    b["table"].to_csv(RESULTS / "readout.csv", index=False, float_format="%.6g", lineterminator="\n")
    b["by_metric"].to_csv(RESULTS / "readout_by_metric.csv", index=False, float_format="%.6g", lineterminator="\n")
    chart_effects(b["table"], CHARTS / "04_effects.png")
    chart_mde(b["table"], CHARTS / "05_detectable_change.png")
    chart_pvalues(b["table"], CHARTS / "06_p_values.png")
    (RESULTS / "READOUT.md").write_text(render(b, f))
    print(f"{f['tests']} tests read in {f['arms_read']} arms; {f['significant']} significant at {R.ALPHA}, "
          f"{f['holm']} after Holm; {f['arms_not_decided']} arms not decided")
    return 0


if __name__ == "__main__":
    sys.exit(main())
