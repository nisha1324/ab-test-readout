"""Part 1 sanity checks: is each experiment fit to be read out at all?

Three questions per treatment arm, asked before any metric is looked at:

1. Sample-ratio mismatch: did the control and treatment groups get the share
   of users they were meant to get?
2. Split stability: did that share stay the same while the test ran?
3. Coverage: did the test run long enough, and was it measured throughout?

The dataset does not publish the intended split of any experiment. The checks
therefore test against an even split and say so when another design fits.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.001          # the usual alarm level for a sample-ratio mismatch
NEAR_EVEN = 0.005      # within half a percentage point of 50% counts as "meant to be even"
MIN_NEW_USERS = 1_000  # intervals with fewer new users are pooled before testing stability
WEEK = 7.0
MIN_DAYS = 7.0         # shorter than this and the test has not seen every day of the week
LATE_START = 7.0       # first snapshot later than this: the first week cannot be replayed
LONG_GAP = 3.0         # days without a snapshot

# Unequal designs worth naming when an arm is far from 50%. Kept short on
# purpose: with enough candidates some ratio always fits.
#   2/3   -> treatment gets twice the control's traffic (1:2)
#   33/67 -> a three-way 34/33/33 split, each treatment compared with the control
DESIGNS = {"1:2": 2 / 3, "34:33": 33 / 67}


def srm_test(users_c: float, users_t: float, expected_share: float = 0.5) -> tuple[float, float]:
    """Chi-square test (1 degree of freedom) that the treatment group's share
    of users equals `expected_share`. Returns (chi-square, p-value)."""
    n = users_c + users_t
    if n <= 0 or not 0 < expected_share < 1:
        raise ValueError("need users in the test and a share strictly between 0 and 1")
    exp_t = n * expected_share
    exp_c = n - exp_t
    chi2 = (users_t - exp_t) ** 2 / exp_t + (users_c - exp_c) ** 2 / exp_c
    return chi2, math.erfc(math.sqrt(chi2 / 2))


def users_short(users_c: float, users_t: float, expected_share: float = 0.5) -> float:
    """How many users the treatment group is short of (negative: over) its
    expected share of the users actually observed."""
    return (users_c + users_t) * expected_share - users_t


def new_users(days, cum_c, cum_t) -> pd.DataFrame:
    """Turn cumulative counts into new users per interval.

    A snapshot where either cumulative count is lower than at the previous kept
    snapshot is skipped (a cumulative count cannot really fall), so its users
    are counted in the next interval instead.
    """
    rows, last_c, last_t = [], 0.0, 0.0
    for day, c, t in sorted(zip(days, cum_c, cum_t)):
        if c < last_c or t < last_t:
            continue
        rows.append((day, c - last_c, t - last_t))
        last_c, last_t = c, t
    return pd.DataFrame(rows, columns=["day", "new_c", "new_t"])


def weekly(inc: pd.DataFrame) -> pd.DataFrame:
    """Sum new users into 7-day blocks counted from the start of the experiment
    (days 0-7, 7-14, ...). A snapshot on day 7.0 closes the first block."""
    block = np.ceil(inc["day"] / WEEK).clip(lower=1).astype(int)
    out = inc.groupby(block)[["new_c", "new_t"]].sum().reset_index(names="week")
    return out


def stability_test(inc: pd.DataFrame, min_new: int = MIN_NEW_USERS) -> tuple[float, int, float]:
    """Chi-square test that the treatment share of *new* users is the same in
    every interval. It needs no knowledge of the intended split.

    Intervals with fewer than `min_new` new users are pooled into one.
    Returns (chi-square, degrees of freedom, p-value); (nan, 0, nan) when
    fewer than two intervals are left.
    """
    total = inc["new_c"] + inc["new_t"]
    big = inc.loc[total >= min_new, ["new_c", "new_t"]].to_numpy(dtype=float)
    small = inc.loc[total < min_new, ["new_c", "new_t"]].to_numpy(dtype=float).sum(axis=0)
    table = np.vstack([big, small]) if small.sum() > 0 else big
    table = table[table.sum(axis=1) > 0]
    if len(table) < 2 or (table.sum(axis=0) == 0).any():
        return float("nan"), 0, float("nan")
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return float(chi2), int(dof), float(p)


def split_verdict(share: float, p_even: float) -> str:
    """Read the even-split test. 'even': no alarm. 'small imbalance': alarm,
    but within half a point of 50%, so the design was almost certainly even and
    users went missing. 'unequal': so far from 50% that it is either a designed
    unequal split or a broken test; the data alone cannot say which."""
    if p_even >= ALPHA:
        return "even"
    return "small imbalance" if abs(share - 0.5) < NEAR_EVEN else "unequal"


def fitting_design(users_c: float, users_t: float) -> str:
    """The named unequal design the counts are consistent with, or ''."""
    fits = [name for name, share in DESIGNS.items()
            if srm_test(users_c, users_t, share)[1] >= ALPHA]
    return fits[0] if len(fits) == 1 else ""


def stability_verdict(p_snapshot: float, p_weekly: float) -> str:
    """'drifting': the split differs between whole weeks. 'timing only':
    snapshots disagree but whole weeks agree (one group's count was updated
    before the other's). 'stable': no alarm. 'not testable': too few intervals."""
    if math.isnan(p_snapshot) and math.isnan(p_weekly):
        return "not testable"
    if not math.isnan(p_weekly) and p_weekly < ALPHA:
        return "drifting"
    if not math.isnan(p_snapshot) and p_snapshot < ALPHA:
        return "timing only"
    return "stable"


def weekly_share_range(week: pd.DataFrame, min_new: int = MIN_NEW_USERS) -> tuple[float, float]:
    """Lowest and highest treatment share of new users among the weekly blocks
    with at least `min_new` new users; (nan, nan) if there is none."""
    total = week["new_c"] + week["new_t"]
    share = (week["new_t"] / total)[total >= min_new]
    if share.empty:
        return float("nan"), float("nan")
    return float(share.min()), float(share.max())


def check_arms(raw: pd.DataFrame, arms: pd.DataFrame) -> pd.DataFrame:
    """Add the split, stability and coverage checks to the arm summary
    (the output of sql/arm_summary.sql)."""
    m1 = raw[raw["metric_id"] == 1]
    series = {key: g for key, g in m1.groupby(["experiment_id", "variant_id"])}
    rows = []
    for a in arms.itertuples(index=False):
        g = series[(a.experiment_id, a.variant_id)]
        inc = new_users(g["time_since_start"], g["count_c"], g["count_t"])
        chi2, p_even = srm_test(a.users_c, a.users_t)
        split = split_verdict(a.treatment_share, p_even)
        _, _, p_snap = stability_test(inc)
        week = weekly(inc)
        _, _, p_week = stability_test(week)
        lo, hi = weekly_share_range(week)
        stability = stability_verdict(p_snap, p_week)
        flags = []
        if split == "small imbalance":
            flags.append("sample-ratio mismatch")
        if split == "unequal":
            flags.append("unequal split")
        if stability == "drifting":
            flags.append("split drifts")
        if a.last_day < MIN_DAYS:
            flags.append("under a week")
        if a.first_day > LATE_START:
            flags.append("first week not recorded")
        if a.max_gap_days >= LONG_GAP:
            flags.append("gap in snapshots")
        if a.metrics_missing_variance_at_end > 0:
            flags.append("variance missing")
        rows.append({
            "srm_chi2": chi2, "srm_p": p_even, "split": split,
            "users_short": users_short(a.users_c, a.users_t) if split != "unequal" else float("nan"),
            "fits_design": fitting_design(a.users_c, a.users_t) if split == "unequal" else "",
            "stability_p_snapshot": p_snap, "stability_p_weekly": p_week, "stability": stability,
            "weekly_share_min": lo, "weekly_share_max": hi,
            "cautions": "; ".join(flags),
        })
    return pd.concat([arms.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
