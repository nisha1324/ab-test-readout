"""Part 2: read each experiment at its last snapshot.

For every treatment arm and metric:

* the difference between the groups and a Welch test on it,
* the relative change (treatment mean / control mean - 1) with a confidence
  interval from the delta method,
* the smallest relative change the test could reliably have detected,
* the p-value after allowing for the number of tests read together.

Everything is computed from group-level figures (users, mean, variance); the
source has no user-level rows. Users are treated as independent.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05           # two-sided false-alarm rate for a single test
POWER = 0.80           # chance of detecting an effect of the stated size
REFERENCE_LIFT = 0.01  # "a change worth having": a 1% relative change (a working choice)

# Part 1 cautions that decide how an arm is read (see results/arm_checks.csv).
NOT_DECIDED = ("unequal split", "under a week")
WARNED = ("sample-ratio mismatch", "split drifts")


def welch(mean_c: float, mean_t: float, var_c: float, var_t: float,
          n_c: float, n_t: float) -> tuple[float, float, float, float]:
    """Welch's unequal-variance t-test from group summaries.
    Returns (difference t - c, its standard error, degrees of freedom, two-sided p)."""
    if n_c < 2 or n_t < 2:
        raise ValueError("each group needs at least two users")
    a, b = var_c / n_c, var_t / n_t
    se = math.sqrt(a + b)
    if se == 0:
        raise ValueError("both variances are zero")
    df = (a + b) ** 2 / (a ** 2 / (n_c - 1) + b ** 2 / (n_t - 1))
    diff = mean_t - mean_c
    return diff, se, df, float(2 * stats.t.sf(abs(diff / se), df))


def relative_lift(mean_c: float, mean_t: float, var_c: float, var_t: float,
                  n_c: float, n_t: float, alpha: float = ALPHA) -> tuple[float, float, float, float]:
    """Relative change mean_t / mean_c - 1 with a delta-method confidence interval.

    The groups are independent, so for the ratio r = mean_t / mean_c
        Var(r) ~ Var(mean_t) / mean_c^2 + mean_t^2 * Var(mean_c) / mean_c^4.
    Returns (lift, standard error, lower, upper)."""
    if mean_c <= 0:
        raise ValueError("a relative change needs a positive control mean")
    lift = mean_t / mean_c - 1
    se = math.sqrt(var_t / n_t / mean_c ** 2 + mean_t ** 2 * var_c / n_c / mean_c ** 4)
    z = stats.norm.isf(alpha / 2)
    return lift, se, lift - z * se, lift + z * se


def power_multiplier(alpha: float = ALPHA, power: float = POWER) -> float:
    """z(1 - alpha/2) + z(power): 2.80 for a 5% two-sided test at 80% power."""
    return float(stats.norm.isf(alpha / 2) + stats.norm.ppf(power))


def mde_relative(mean_c: float, se_diff: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """Minimum detectable effect, as a share of the control mean: the true
    relative change this test would have flagged `power` of the time."""
    return power_multiplier(alpha, power) * se_diff / mean_c


def users_per_group(sd_over_mean: float, lift: float, alpha: float = ALPHA, power: float = POWER) -> int:
    """Users needed in each group of an even two-group test to detect a
    relative change `lift`, for a metric with the given SD / mean."""
    if lift == 0:
        raise ValueError("the change to detect cannot be zero")
    return math.ceil(2 * (power_multiplier(alpha, power) * sd_over_mean / lift) ** 2)


def any_false_alarm(tests: int, alpha: float = ALPHA) -> float:
    """Chance of at least one false alarm among `tests` independent tests when
    nothing changed. For metrics that move together the true figure is lower,
    so this is the upper end."""
    return 1 - (1 - alpha) ** tests


def holm(p) -> np.ndarray:
    """Holm's step-down adjusted p-values (controls the chance of any false
    alarm in the family, whatever the dependence between the tests)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    stepped = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(stepped, 1.0)
    return out


def benjamini_hochberg(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (controls the expected share of
    false alarms among the results called significant)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    stepped = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(stepped, 1.0)
    return np.maximum(out, p)   # p * m / m can land one rounding step below p


def reading(cautions: str, has_variance: bool) -> str:
    """How one arm x metric is read, given the arm's Part 1 cautions.

    'not decided': the arm's split is unexplained or it ran under a week.
    'not testable': the source has no variance for this metric.
    'read with warning': a sample-ratio mismatch or a drifting split.
    'read': nothing in Part 1 stands in the way."""
    items = [c for c in cautions.split("; ") if c]
    if any(c in NOT_DECIDED for c in items):
        return "not decided"
    if not has_variance:
        return "not testable"
    return "read with warning" if any(c in WARNED for c in items) else "read"


def read_arms(final: pd.DataFrame, arm_checks: pd.DataFrame) -> pd.DataFrame:
    """The readout table: one row per arm x metric (the output of
    sql/final_snapshot.sql), with effect, interval, test, detectable effect
    and corrected p-values.

    Corrections cover the rows that are read (with or without a warning):
    `p_holm` within one experiment (all its arms and metrics), `p_bh` across
    every row read in the dataset."""
    cautions = arm_checks.set_index(["experiment_id", "variant_id"])["cautions"].fillna("")
    rows = []
    for r in final.itertuples(index=False):
        has_var = not (pd.isna(r.variance_c) or pd.isna(r.variance_t))
        row = {"reading": reading(cautions[(r.experiment_id, r.variant_id)], has_var),
               "cautions": cautions[(r.experiment_id, r.variant_id)],
               "diff": r.mean_t - r.mean_c, "lift": r.mean_t / r.mean_c - 1}
        if has_var:
            _, se, df, p = welch(r.mean_c, r.mean_t, r.variance_c, r.variance_t, r.users_c, r.users_t)
            _, lift_se, lo, hi = relative_lift(r.mean_c, r.mean_t, r.variance_c, r.variance_t,
                                               r.users_c, r.users_t)
            row.update(se_diff=se, welch_df=df, p=p, lift_se=lift_se, lift_lo=lo, lift_hi=hi,
                       mde=mde_relative(r.mean_c, se))
        rows.append(row)
    cols = ["reading", "cautions", "diff", "lift", "se_diff", "welch_df", "p",
            "lift_se", "lift_lo", "lift_hi", "mde"]
    out = pd.concat([final.reset_index(drop=True), pd.DataFrame(rows, columns=cols)], axis=1)
    read = out["reading"].isin(["read", "read with warning"])
    out["p_holm"] = np.nan
    out["p_bh"] = np.nan
    for _, idx in out[read].groupby("experiment_id").groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"])
    out.loc[read, "p_bh"] = benjamini_hochberg(out.loc[read, "p"])
    return out
