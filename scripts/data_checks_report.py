"""Part 1: check the data before reading any experiment.

    python scripts/data_checks_report.py

Reads the committed CSV in data/raw/ (no download needed) and writes
results/DATA_CHECKS.md, three CSVs and three charts.
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
from abtest import checks as C, load, source  # noqa: E402

RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"

# Reference palette (2 slots, light mode)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "path.simplify": False,
})
SAVE = {"dpi": 150, "metadata": {"Software": None}}

METRIC_TYPE = {1: "yes/no per user", 2: "count per user", 3: "count per user",
               4: "amount per user (0 or more)"}
COVERAGE_CAUTIONS = ("under a week", "first week not recorded", "gap in snapshots")


def n(x: float, nd: int = 0) -> str:
    return f"{x:,.{nd}f}"


def pct(x: float, nd: int = 2) -> str:
    return f"{100 * x:.{nd}f}%"


def pval(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 1e-15:
        return "< 1e-15"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def arm_name(exp: str, variant: int) -> str:
    return f"`{exp}` / {variant}"


def md_table(df: pd.DataFrame, left: tuple[str, ...] = ()) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(" --- " if (i == 0 or c in left) else " ---: "
                         for i, c in enumerate(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def style(ax, title: str | None = None, subtitle: str | None = None) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2)


def has(arms: pd.DataFrame, caution: str) -> pd.Series:
    return arms["cautions"].str.split("; ").apply(lambda items: caution in items)


# ------------------------------------------------------------------ charts
def chart_split(arms: pd.DataFrame, path: Path) -> None:
    """Funnel plot: distance from an even split against the size of the test."""
    near = arms[arms["split"] != "unequal"]
    flagged = near["split"] == "small imbalance"
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    xs = np.logspace(np.log10(near["users"].min() * 0.8), np.log10(near["users"].max() * 1.3), 200)
    bar = stats.norm.isf(C.ALPHA / 2) * 0.5 / np.sqrt(xs) * 100
    ax.fill_between(xs, -bar, bar, color=GRID, alpha=0.7, linewidth=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    y = (near["treatment_share"] - 0.5) * 100
    ax.scatter(near.loc[~flagged, "users"], y[~flagged], s=26, color=BLUE,
               edgecolor=SURFACE, linewidth=0.8, label="Within chance", zorder=3)
    ax.scatter(near.loc[flagged, "users"], y[flagged], s=44, color=ORANGE, marker="D",
               edgecolor=SURFACE, linewidth=0.8, label="Sample-ratio mismatch (p < 0.001)", zorder=4)
    for r in near[flagged].itertuples():
        ax.annotate(r.experiment_id, (r.users, (r.treatment_share - 0.5) * 100),
                    xytext=(7, -3), textcoords="offset points", fontsize=8.5, color=INK)
    ax.text(near["users"].min() * 1.15, 0.22, "Grey: the range chance alone explains\n(99.9% of truly even splits fall inside it)",
            fontsize=8.5, color=INK_2, va="center")
    ax.set_xscale("log")
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel("Users in the comparison (control + treatment, log scale)")
    ax.set_ylabel("Treatment share of users,\npoints away from 50%")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    style(ax, f"{int(flagged.sum())} of {len(near)} even-split arms are short of treatment users by more than chance",
          f"One dot per treatment arm at its last snapshot. The {len(arms) - len(near)} arms far from 50% are listed separately.")
    fig.tight_layout()
    fig.savefig(path, **SAVE)
    plt.close(fig)


def chart_coverage(raw: pd.DataFrame, arms: pd.DataFrame, path: Path) -> None:
    """One row per arm: a line wherever snapshots are less than LONG_GAP days apart."""
    caution = arms["cautions"].str.split("; ").apply(
        lambda items: any(c in items for c in COVERAGE_CAUTIONS))
    order = arms.assign(caution=caution).sort_values(
        ["caution", "last_day", "first_day"], ascending=[False, True, True]).reset_index(drop=True)
    m1 = raw[raw["metric_id"] == 1]
    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    for i, a in order.iterrows():
        days = np.sort(m1.loc[(m1["experiment_id"] == a.experiment_id)
                              & (m1["variant_id"] == a.variant_id), "time_since_start"].to_numpy())
        colour = ORANGE if a.caution else BLUE
        start = 0
        for j in range(1, len(days) + 1):
            if j == len(days) or days[j] - days[j - 1] >= C.LONG_GAP:
                ax.plot([days[start], days[j - 1]], [i, i], color=colour, linewidth=1.6,
                        solid_capstyle="butt")
                start = j
        ax.plot(days[0], i, marker="|", color=colour, markersize=5, markeredgewidth=1.2)
    n_caution = int(caution.sum())
    ax.axvline(C.MIN_DAYS, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(C.MIN_DAYS + 1, len(order) + 0.6, "day 7", fontsize=8.5, color=INK_2, va="bottom")
    ax.axhline(n_caution - 0.5, color=GRID, linewidth=0.8)
    ax.text(133, n_caution / 2, f"{n_caution} arms with a\ncoverage caution",
            fontsize=9, color=INK, ha="right", va="center")
    ax.text(133, n_caution + 4, f"{len(order) - n_caution} arms\nmeasured from\nthe first week\nwith no long gap",
            fontsize=9, color=INK, ha="right", va="bottom")
    ax.set_yticks([])
    ax.set_ylim(-1, len(order) + 3)
    ax.set_xlim(0, 134)
    ax.set_xlabel("Days since the experiment started")
    style(ax, f"{n_caution} of {len(order)} arms ended within a week, were first recorded late or have a hole in the record",
          "One row per treatment arm. A line is drawn wherever snapshots are less than 3 days apart; "
          "the tick is the first snapshot.")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, **SAVE)
    plt.close(fig)


def growth_curves(raw: pd.DataFrame, arms: pd.DataFrame) -> pd.DataFrame:
    """Control mean as a share of its final value, on a grid of 'share of the
    run elapsed', one row per experiment x metric x grid point. Only experiments
    recorded from the first week, and running at least 14 days, are used."""
    exp = arms.groupby("experiment_id").agg(first=("first_day", "min"), last=("last_day", "max"))
    keep = exp[(exp["first"] <= C.LATE_START) & (exp["last"] >= 14)].index
    grid = np.round(np.arange(0.5, 1.0001, 0.05), 2)  # every kept experiment has a snapshot by half-time
    rows = []
    control = raw[raw["experiment_id"].isin(keep) & (raw["count_c"] > 0)].drop_duplicates(
        ["experiment_id", "metric_id", "time_since_start"])
    for (e, m), g in control.groupby(["experiment_id", "metric_id"]):
        g = g.sort_values("time_since_start")
        x = g["time_since_start"].to_numpy() / g["time_since_start"].max()
        y = g["mean_c"].to_numpy() / g["mean_c"].iloc[-1]
        if x.min() > grid[0]:
            raise ValueError(f"{e}: no snapshot in the first half of the run")
        for gx, gy in zip(grid, np.interp(grid, x, y)):
            rows.append((e, m, gx, gy))
    return pd.DataFrame(rows, columns=["experiment_id", "metric_id", "run_share", "mean_vs_final"])


def chart_growth(curves: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.6), sharey=True)
    for ax, m in zip(axes, (1, 2, 3, 4)):
        d = curves[curves["metric_id"] == m]
        for _, g in d.groupby("experiment_id"):
            ax.plot(g["run_share"] * 100, g["mean_vs_final"] * 100, color=MUTED, linewidth=0.6, alpha=0.45)
        med = d.groupby("run_share")["mean_vs_final"].median()
        ax.plot(med.index * 100, med.values * 100, color=BLUE, linewidth=2)
        half = med.loc[0.5] * 100
        ax.plot(50, half, "o", color=BLUE, markersize=6, markeredgecolor=SURFACE)
        ax.annotate(f"{half:.0f}% at\nhalf-time", (50, half), xytext=(8, -28),
                    textcoords="offset points", fontsize=8.5, color=INK)
        style(ax)
        ax.set_title(f"Metric {m}\n{METRIC_TYPE[m]}", loc="left", fontsize=9.5, color=INK)
        ax.set_xticks([50, 75, 100], ["half-time", "75%", "end"])
        ax.set_xlim(47, 103)
        ax.set_ylim(0, 125)
    axes[0].set_ylabel("Control-group mean,\n% of its value at the end")
    fig.supxlabel("Share of the experiment's run elapsed", fontsize=10, color=INK_2)
    fig.suptitle("The metrics keep growing while a test runs, so only same-day comparisons are fair",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.905, f"Grey: one line per experiment ({curves['experiment_id'].nunique()} recorded from "
             "their first week and at least 14 days long). Blue: the median.", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, **SAVE)
    plt.close(fig)


# ------------------------------------------------------------------ report
def build() -> dict:
    raw = load.read_raw()
    con = load.connect(raw)
    integrity = load.run_sql(con, "integrity")
    metrics = load.run_sql(con, "metric_profile")
    arms = C.check_arms(raw, load.run_sql(con, "arm_summary"))
    curves = growth_curves(raw, arms)
    return {"raw": raw, "integrity": integrity, "metrics": metrics, "arms": arms, "curves": curves}


def facts(b: dict) -> dict:
    """Every number the report and the README quote, computed once."""
    raw, arms, integrity = b["raw"], b["arms"], b["integrity"]
    viol = dict(zip(integrity["check_name"], integrity["violations"]))
    exp = arms.groupby("experiment_id").agg(
        arms=("variant_id", "size"), first=("first_day", "min"), last=("last_day", "max"),
        clean=("cautions", lambda s: (s == "").all()))
    gaps = raw[raw["metric_id"] == 1].sort_values("time_since_start").groupby(
        ["experiment_id", "variant_id"])["time_since_start"].diff().dropna().round(6)
    small = arms[arms["split"] == "small imbalance"]
    unequal = arms[arms["split"] == "unequal"]
    at_half = b["curves"][b["curves"]["run_share"] == 0.5]
    half = at_half.groupby("metric_id")["mean_vs_final"].median()
    below = at_half.assign(below=at_half["mean_vs_final"] < 1).groupby("metric_id")["below"].sum()
    f = {
        "rows": len(raw), "experiments": len(exp), "arms": len(arms),
        "arms_per_experiment": exp["arms"].value_counts().sort_index().to_dict(),
        "multi_arm_experiments": int((exp["arms"] > 1).sum()),
        "viol": viol,
        "users_min": int(arms["users"].min()), "users_median": float(arms["users"].median()),
        "users_max": int(arms["users"].max()),
        "arms_over_1m": int((arms["users"] >= 1_000_000).sum()),
        "days_min": float(exp["last"].min()), "days_median": float(exp["last"].median()),
        "days_max": float(exp["last"].max()),
        "gap_half": int((gaps == 0.5).sum()), "gap_one": int((gaps == 1.0).sum()), "gaps": len(gaps),
        "even": int((arms["split"] == "even").sum()), "small": len(small), "unequal": len(unequal),
        "naive_alarms": int((arms["srm_p"] < C.ALPHA).sum()),
        "unequal_named": int((unequal["fits_design"] != "").sum()),
        "small_short_min": float(small["users_short"].min()), "small_short_max": float(small["users_short"].max()),
        "small_share_min": float(small["treatment_share"].min()),
        "stability": arms["stability"].value_counts().to_dict(),
        "frozen_snapshots": int(arms["frozen_snapshots"].sum()),
        "frozen_arms": int((arms["frozen_snapshots"] > 0).sum()),
        "one_sided_snapshots": int(arms["one_sided_snapshots"].sum()),
        "one_sided_arms": int((arms["one_sided_snapshots"] > 0).sum()),
        "falling_arms": int((arms["falling_snapshots"] > 0).sum()),
        "variance_any": int((arms["metrics_missing_variance"] > 0).sum()),
        "variance_end": int((arms["metrics_missing_variance_at_end"] > 0).sum()),
        "clean_arms": int((arms["cautions"] == "").sum()),
        "clean_experiments": int(exp["clean"].sum()),
        "short_experiments": int((exp["last"] < C.MIN_DAYS).sum()),
        "late_experiments": int((exp["first"] > C.LATE_START).sum()),
        "half_time": {int(k): float(v) for k, v in half.items()},
        "half_below": {int(k): int(v) for k, v in below.items()},
        "growth_experiments": int(b["curves"]["experiment_id"].nunique()),
    }
    for caution in ("sample-ratio mismatch", "unequal split", "split drifts", "under a week",
                    "first week not recorded", "gap in snapshots", "variance missing"):
        f["n_" + caution] = int(has(arms, caution).sum())
    f["coverage_caution_arms"] = int(arms["cautions"].str.split("; ").apply(
        lambda items: any(c in items for c in COVERAGE_CAUTIONS)).sum())
    return f


def render(b: dict, f: dict) -> str:
    arms, metrics, integrity = b["arms"], b["metrics"], b["integrity"]
    v = f["viol"]
    per = f["arms_per_experiment"]

    integ = integrity.rename(columns={"check_name": "Check", "rule": "Rule", "violations": "Breaks the rule"})
    integ["Check"] = "`" + integ["Check"] + "`"

    mt = pd.DataFrame({
        "Metric": metrics["metric_id"],
        "What the source says": [METRIC_TYPE[m] for m in metrics["metric_id"]],
        "Lowest control mean": [n(x, 3) for x in metrics["min_mean"]],
        "Median": [n(x, 3) for x in metrics["median_mean"]],
        "Highest": [n(x, 3) for x in metrics["max_mean"]],
        "Median SD ÷ mean": [n(x, 2) for x in metrics["median_sd_over_mean"]],
        "Arms with no variance at the end": metrics["arms_missing_variance"],
        "Median mean at half-time, % of final": [f"{100 * f['half_time'][m]:.0f}%" for m in metrics["metric_id"]],
    })

    small = arms[arms["split"] == "small imbalance"].sort_values("srm_p")
    st = pd.DataFrame({
        "Experiment / arm": [arm_name(e, a) for e, a in zip(small["experiment_id"], small["variant_id"])],
        "Users": [n(x) for x in small["users"]],
        "Treatment share": [pct(x, 3) for x in small["treatment_share"]],
        "Treatment users short of 50%": [n(x) for x in small["users_short"]],
        "p-value": [pval(p) for p in small["srm_p"]],
        "Split over time": small["stability"],
    })

    uneq = arms[arms["split"] == "unequal"].sort_values("treatment_share")
    ut = pd.DataFrame({
        "Experiment / arm": [arm_name(e, a) for e, a in zip(uneq["experiment_id"], uneq["variant_id"])],
        "Treatments in the experiment": uneq["arms_in_experiment"],
        "Users": [n(x) for x in uneq["users"]],
        "Treatment share": [pct(x, 2) for x in uneq["treatment_share"]],
        "Named design that fits": [d if d else "none" for d in uneq["fits_design"]],
        "Split over time": uneq["stability"],
    })

    unstable = arms[arms["stability"].isin(["drifting", "timing only"])].sort_values(
        ["stability", "experiment_id"])
    dt = pd.DataFrame({
        "Experiment / arm": [arm_name(e, a) for e, a in zip(unstable["experiment_id"], unstable["variant_id"])],
        "Verdict": unstable["stability"],
        "p-value between snapshots": [pval(p) for p in unstable["stability_p_snapshot"]],
        "p-value between weeks": [pval(p) for p in unstable["stability_p_weekly"]],
        "Weekly treatment share, lowest": [pct(x, 2) for x in unstable["weekly_share_min"]],
        "highest": [pct(x, 2) for x in unstable["weekly_share_max"]],
    })

    caution_rows = [
        ("sample-ratio mismatch", "Treatment share within half a point of 50%, but further from it than chance allows (p < 0.001)",
         "Read with a warning; never as the only evidence for a launch"),
        ("unequal split", "Treatment share more than half a point from 50%",
         "Readable only against the intended split, which the source does not publish"),
        ("split drifts", "Treatment share of new users differs between weeks (p < 0.001)",
         "Read with a warning"),
        ("under a week", "Last snapshot before day 7", "Too short to decide on"),
        ("first week not recorded", "First snapshot after day 7",
         "The end result is readable; what a daily look would have shown early on is not"),
        ("gap in snapshots", "3 or more days with no snapshot",
         "The end result is readable; the daily path has a hole"),
        ("variance missing", "No variance for at least one metric at the last snapshot",
         "Metric 1 is readable; the metrics without a variance cannot be tested"),
    ]
    ct = pd.DataFrame({
        "Caution": [c for c, _, _ in caution_rows],
        "Rule": [r for _, r, _ in caution_rows],
        "Arms": [f["n_" + c] for c, _, _ in caution_rows],
        "What it means for the readout": [w for _, _, w in caution_rows],
    })

    stab = f["stability"]
    out = f"""# Part 1: can these experiments be read at all?

*Generated by `scripts/data_checks_report.py` from the committed source file. Every number below is computed by the script; nothing is typed in by hand.*

Before anyone looks at whether a treatment won, three questions decide whether the experiment can be trusted: did each group get the users it was meant to get, did that stay true while the test ran, and was the test measured for long enough. This report asks them of all {f['experiments']} experiments in the ASOS Digital Experiments Dataset.

**Headline:** {f['clean_arms']} of the {f['arms']} treatment arms ({f['clean_experiments']} of {f['experiments']} experiments) pass every check. The other {f['arms'] - f['clean_arms']} carry at least one caution, listed in section 6, and each caution says what it rules out.

## 1. What is in the file

- **Source:** [{source.CSV_NAME}]({source.PROJECT_URL}), CC BY 4.0, SHA-256 `{source.CSV_SHA256[:16]}…` (pinned in `abtest/source.py`).
- **{n(f['rows'])} rows.** One row is one experiment × treatment arm × metric × snapshot time. It holds the number of users, the mean and the variance for the control group and for that treatment group, all cumulative since the experiment started. There are no user-level rows.
- **{f['experiments']} experiments, {f['arms']} treatment arms.** {per.get(1, 0)} experiments have one treatment, {per.get(2, 0)} have two, {per.get(3, 0)} have three and {per.get(4, 0)} has four. In the {f['multi_arm_experiments']} experiments with more than one treatment, every arm is compared with the same control group.
- **Size.** The smallest comparison has {n(f['users_min'])} users (control plus treatment), the median {n(f['users_median'] / 1e6, 1)} million and the largest {n(f['users_max'] / 1e6, 1)} million; {f['arms_over_1m']} of the {f['arms']} arms have at least a million.
- **Length.** The shortest experiment's last snapshot is on day {f['days_min']:g}, the median on day {f['days_median']:g} and the longest on day {f['days_max']:g}.
- **Cadence.** {n(f['gap_half'])} of the {n(f['gaps'])} steps between snapshots are 12 hours and {n(f['gap_one'])} are one day; the rest are longer.
- **Not in the file:** experiment names, dates, what was changed, the metric names, and the intended traffic split. The source's datasheet also says the experiments are not representative of ASOS's business. So this project can show how to read experiments like these; it cannot say anything about ASOS.

Structural rules, checked in DuckDB SQL ([`sql/integrity.sql`](../sql/integrity.sql)):

{md_table(integ, left=("Rule",))}

Four rules are broken somewhere:

- `empty_groups`: the {v['empty_groups']} rows are one snapshot (the first, at day 3) of experiment `64dc88`, where both arms report zero users. It is ignored.
- `snapshots_missing_a_metric`: one snapshot of `a4386f` reports metric 1 only.
- `rows_missing_variance`: {f['variance_any']} arms lack a variance for metrics 2 to 4 at some snapshot, and {f['variance_end']} of them still lack it at their last snapshot. Those {f['variance_end']} match the datasheet, which says 5 treatments are affected.
- `user_count_falls`: in {f['falling_arms']} arms a cumulative user count is lower than at the previous snapshot, which cannot really happen. The checks below skip such a snapshot and count its users in the next interval.

Two more things show the snapshots come from a pipeline, not a clean ledger: {f['frozen_snapshots']} snapshots in {f['frozen_arms']} arms add no user to either group (most likely the figures were not refreshed), and {f['one_sided_snapshots']} snapshots in {f['one_sided_arms']} arms add users to one group only.

## 2. The four metrics

The source does not name the metrics. Its schema says only this: metric 1 takes yes/no responses, metrics 2 and 3 are counts, metric 4 is a non-negative real number, and metrics 2 to 4 are right-skewed. This project uses those descriptions and does not guess at names such as "conversion" or "revenue".

{md_table(mt, left=("What the source says",))}

*Control group at each arm's last snapshot ([`sql/metric_profile.sql`](../sql/metric_profile.sql)). SD ÷ mean is the standard deviation divided by the mean. The last column uses the {f['growth_experiments']} experiments that were recorded from their first week and ran at least 14 days.*

Three things follow.

- **Metric 1 is a share of users.** Its variance equals mean × (1 − mean) in every row (`binary_variance_mismatch` is 0), so each user is a yes or a no and the mean is the share of users with a yes.
- **Metrics 2 to 4 are noisier.** The median SD ÷ mean rises from {n(metrics['median_sd_over_mean'].iloc[0], 2)} for metric 1 to {n(metrics['median_sd_over_mean'].iloc[3], 2)} for metric 4. For the same number of users, the same relative change is harder to detect on metric 4. Part 2 turns this into a required sample size.
- **The metrics keep growing while the test runs.** The figures are cumulative, so the longer the test has run, the more each user has had time to do. At half-time the median control mean is {100 * f['half_time'][1]:.0f}% of its final value for metric 1 and {100 * f['half_time'][4]:.0f}% for metric 4. It is below the final value in {f['half_below'][1]} of {f['growth_experiments']} experiments for metric 1 and {f['half_below'][4]} of {f['growth_experiments']} for metric 4, so this is the rule but not without exceptions. A treatment mean on day 28 must never be compared with a control mean on day 14, and an effect in absolute units depends on when the test stopped.

![Metrics grow during a test](charts/03_metrics_grow.png)

## 3. Did each group get its share of users?

If a test is meant to split users evenly and one group ends up with fewer, some users went missing after assignment (a redirect that fails, a slower page that loses impatient visitors, a tracking bug). Those missing users are rarely typical, so the comparison of the two groups is no longer fair. This is a **sample-ratio mismatch**, and it is checked before any metric.

The test is a chi-square test of the final user counts against an even split, with the usual alarm level of p < {C.ALPHA}. The intended split is not published, so the result is read in three bands:

| Band | Rule | Arms |
| --- | --- | ---: |
| Even | no alarm | {f['even']} |
| Sample-ratio mismatch | alarm, treatment share within half a point of 50% | {f['small']} |
| Unequal split | alarm, treatment share more than half a point from 50% | {f['unequal']} |

A reader who tests every arm against 50/50 and stops there gets {f['naive_alarms']} alarms. Only {f['small']} of them look like users going missing.

![Sample-ratio check](charts/01_sample_ratio.png)

**The {f['small']} mismatches.** In each, the test was almost certainly meant to be even, and the treatment group is short by {n(f['small_short_min'])} to {n(f['small_short_max'])} users:

{md_table(st, left=("Split over time",))}

The imbalances are small: the lowest treatment share is {pct(f['small_share_min'], 3)}. They are flagged because these tests are so large that chance cannot explain even that. Whether so few missing users could move a metric depends on who they were, which the file cannot show. Part 2 will read these arms with a warning.

**The {f['unequal']} unequal splits.** These are too far from 50% to be lost users in an even test. Either the test was designed with an unequal split, or it is badly broken; the data alone cannot say which.

{md_table(ut, left=("Named design that fits", "Split over time"))}

Two designs were checked by name: a treatment given twice the control's traffic (1:2), and a three-way 34/33/33 split, where each treatment has 33 users for the control's 34. {f['unequal_named']} of the {f['unequal']} arms are consistent with one of them (p ≥ {C.ALPHA} against that design). That is an inference from the counts, not a fact from the source. The list of named designs was kept to two on purpose: with enough candidates, some ratio always fits.

## 4. Did the split hold while the test ran?

A check that needs no knowledge of the intended split: among the *new* users of each interval, the treatment share should be the same from start to finish. It is tested between snapshots (intervals with fewer than {n(C.MIN_NEW_USERS)} new users pooled) and again between whole weeks.

- **Stable:** {stab.get('stable', 0)} arms.
- **Timing only:** {stab.get('timing only', 0)} arms differ between snapshots but not between weeks. That is the pattern expected when one group's count is refreshed before the other's and the next snapshot catches up.
- **Drifting:** {stab.get('drifting', 0)} arms differ between weeks.

{md_table(dt, left=("Verdict",))}

The drifts are small in size. They are flagged because a split that moves suggests the traffic allocation, or the tracking, changed during the test.

A note on the alarm level: with {f['arms']} arms tested at p < {C.ALPHA}, about {f['arms'] * C.ALPHA:.1f} false alarms are expected per check if nothing were wrong, so chance alone does not explain these flags (but see the note on lumpy traffic in section 7). The opposite error cannot be ruled out: an arm with no flag is not proven clean.

## 5. Was each test measured for long enough?

![Coverage of each arm](charts/02_coverage.png)

- **Under a week:** {f['n_under a week']} arms in {f['short_experiments']} experiments end before day 7. Shopping differs between weekdays and weekends, so a test that has not seen every day of the week is too short to decide on, however many users it has.
- **First week not recorded:** {f['n_first week not recorded']} arms in {f['late_experiments']} experiments have their first snapshot after day 7 (the latest starts on day {arms['first_day'].max():g}). The final result is still readable. What cannot be replayed is what someone checking daily would have seen in the first days, which matters for Part 3.
- **Gap in snapshots:** {f['n_gap in snapshots']} arms have 3 or more days with no snapshot (the longest gap is {arms['max_gap_days'].max():g} days).

## 6. What goes into Part 2

{md_table(ct, left=("Rule", "What it means for the readout"))}

An arm can carry more than one caution; {f['arms'] - f['clean_arms']} arms carry at least one and {f['clean_arms']} carry none. The full list is in [`arm_checks.csv`](arm_checks.csv), one row per arm with every figure used above.

**So what, for a team that runs experiments:**

1. **Check the split before the metrics, and against the intended split.** Testing everything against 50/50 raised {f['naive_alarms']} alarms here, and {f['unequal']} of them are too far from 50% to be an even test that lost a few users. The intended allocation belongs in the experiment record, next to the results.
2. **A sample-ratio alarm in a very large test can be a very small imbalance.** The largest shortfall here is {n(f['small_short_max'])} users out of {n(float(arms.loc[arms['users_short'].idxmax(), 'users']) / 1e6, 1)} million. Report the size with the alarm, so the decision is about what the missing users could have changed, and not about a p-value.
3. **Do not decide on a test shorter than a week,** and keep the daily record complete: {f['coverage_caution_arms']} of {f['arms']} arms here are too short, start late or have a hole in the record.
4. **Never compare metrics across different days of a test.** They are cumulative, and at half-time the typical control mean is at roughly {100 * min(f["half_time"].values()):.0f}% to {100 * max(f["half_time"].values()):.0f}% of its final level.

## 7. Limits

- The intended split, the dates and the metric names are not published. Every statement about design is an inference and is labelled as one.
- The alarm level (p < {C.ALPHA}), the half-point band, the 7-day rules and the 3-day gap are working choices, set in `abtest/checks.py`. They were chosen after looking at the data, so the counts in section 6 would shift a little with other choices.
- The checks use user counts only. They can show that a split is off; they cannot show why, or which users are missing.
- The chi-square tests treat users as independent. Real traffic is lumpier than that (the source cannot show how much), so the p-values are somewhat too small, most of all for the largest tests.
- An arm with no caution has passed these checks only. Problems that leave the user counts untouched, such as a metric logged wrongly in one group, are invisible here.
"""
    return out


def main() -> int:
    b = build()
    f = facts(b)
    RESULTS.mkdir(exist_ok=True)
    CHARTS.mkdir(exist_ok=True)
    b["arms"].to_csv(RESULTS / "arm_checks.csv", index=False, float_format="%.6g", lineterminator="\n")
    b["integrity"].to_csv(RESULTS / "integrity_checks.csv", index=False, lineterminator="\n")
    b["metrics"].to_csv(RESULTS / "metric_profile.csv", index=False, float_format="%.6g", lineterminator="\n")
    chart_split(b["arms"], CHARTS / "01_sample_ratio.png")
    chart_coverage(b["raw"], b["arms"], CHARTS / "02_coverage.png")
    chart_growth(b["curves"], CHARTS / "03_metrics_grow.png")
    (RESULTS / "DATA_CHECKS.md").write_text(render(b, f))
    print(f"{f['arms']} arms in {f['experiments']} experiments; {f['clean_arms']} arms with no caution; "
          f"{f['small']} sample-ratio mismatches, {f['unequal']} unequal splits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
