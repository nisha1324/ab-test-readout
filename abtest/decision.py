"""Part 4: from a readout to a decision.

One metric is the *primary*: the one the test was run to move, named before
the test. The others are *guard rails*: they are not expected to improve, but
a launch must not make them worse.

The source does not name its metrics or say which direction is good, so every
rule here takes a `direction`: +1 if a higher value is better, -1 if a lower
value is better. The same direction is applied to all four metrics.

The rules, in the order they are applied to one treatment arm:

1. Part 1 said the arm cannot be decided on              -> "no decision"
2. the primary metric has no variance in the source      -> "no decision"
3. Part 1 warned about the split, or the arm's last
   snapshot jumps away from the one before it (Part 3)   -> "check the data"
4. the primary is significantly worse                    -> "do not launch"
5. a guard rail is significantly worse                   -> "do not launch"
6. the primary is significantly better (after allowing
   for the number of arms in the experiment) and every
   guard rail rules out a loss of MARGIN or more         -> "launch"
7. the primary is better but a guard rail is too wide to
   rule that loss out, or cannot be tested               -> "run longer"
8. the primary's interval lies inside +/- MARGIN         -> "no change"
9. anything else (the test could not tell)               -> "run longer"

The bar for good news is higher than the bar for bad news on purpose: a
launch needs the primary to survive Holm's correction across the arms of its
experiment, while an uncorrected p < ALPHA in the wrong direction is enough
to stop one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import peeking as P
from . import readout as R

ALPHA = R.ALPHA             # the same 5% level as Parts 2 and 3
MARGIN = R.REFERENCE_LIFT   # a loss of 1% of the control mean is the largest a launch may risk
PRIMARY = 1                 # working choice: metric 1, the yes/no per user

VERDICTS = ("launch", "do not launch", "no change", "run longer", "check the data", "no decision")
READ = ("read", "read with warning")
KEY = ["experiment_id", "variant_id"]


def worst_case(lo: float, hi: float, direction: int) -> float:
    """The end of the confidence interval that is worst for the business, as a
    signed gain: negative means a loss. For "higher is better" that is the
    lower end; for "lower is better" it is the upper end, with its sign turned."""
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    return lo if direction == 1 else -hi


def primary_state(lift: float, lo: float, hi: float, p: float, p_arms: float,
                  direction: int = 1, margin: float = MARGIN, alpha: float = ALPHA) -> str:
    """How the primary metric reads.

    'worse':       significantly in the bad direction (uncorrected p).
    'better':      significantly in the good direction, also after the
                   correction for the number of arms (`p_arms`).
    'unconfirmed': in the good direction at p < alpha, but not after that correction.
    'no change':   not significant, interval inside +/- margin.
    'too wide':    not significant, interval reaches beyond +/- margin."""
    worst_case(lo, hi, direction)   # validates direction
    if p < alpha:
        if direction * lift < 0:
            return "worse"
        return "better" if p_arms < alpha else "unconfirmed"
    return "no change" if (lo > -margin and hi < margin) else "too wide"


def guard_state(lift: float, lo: float, hi: float, p: float,
                direction: int = 1, margin: float = MARGIN, alpha: float = ALPHA) -> str:
    """How one guard-rail metric reads.

    'worse':       significantly in the bad direction (uncorrected p).
    'cleared':     the interval rules out a loss of `margin` or more.
    'not cleared': it does not."""
    if p < alpha and direction * lift < 0:
        return "worse"
    return "cleared" if worst_case(lo, hi, direction) > -margin else "not cleared"


def verdict(reading: str, primary: str | None, guards: list[str], end_jump: bool = False) -> tuple[str, str]:
    """The decision for one arm and the reason for it.

    `reading`: the arm's Part 2 reading on the primary metric.
    `primary`: output of `primary_state`, or None if the primary cannot be tested.
    `guards`:  output of `guard_state` for each guard rail, 'not testable' for
               one that has no variance.
    `end_jump`: the last snapshot of some metric jumps away from the look before it."""
    if reading == "not decided":
        return "no decision", "Part 1: split unexplained or under a week of data"
    if primary is None:
        return "no decision", "the primary metric cannot be tested"
    if reading == "read with warning":
        return "check the data", "Part 1 warning on the split"
    if end_jump:
        return "check the data", "last snapshot jumps away from the one before"
    if primary == "worse":
        return "do not launch", "primary worse"
    if "worse" in guards:
        return "do not launch", ("guard rail worse although the primary is better"
                                 if primary == "better" else "guard rail worse")
    if primary == "better":
        if "not testable" in guards:
            return "run longer", "primary better, a guard rail cannot be tested"
        if "not cleared" in guards:
            return "run longer", "primary better, a guard rail not cleared"
        return "launch", "primary better, guard rails cleared"
    if primary == "unconfirmed":
        return "run longer", "primary better only before allowing for the other arms"
    if primary == "no change":
        return "no change", "primary inside the margin, no guard rail worse"
    return "run longer", "primary too wide to tell"


def end_checks(looks: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """For every arm x metric that Part 2 read: does its last daily look jump
    away from the last earlier look that is not in a blip (Part 3's
    `end_jump`)? Part 3 only ran this on the arms it replayed; a decision
    needs it for every arm, so it is run here on all of them."""
    by_test = {k: g for k, g in looks.groupby(KEY + ["metric_id"])}
    rows = []
    for r in table[table["reading"].isin(READ)].itertuples(index=False):
        s = P.look_stats(by_test[(r.experiment_id, r.variant_id, r.metric_id)])
        d = P.decide(s)
        rows.append({"experiment_id": r.experiment_id, "variant_id": r.variant_id,
                     "metric_id": r.metric_id, "looks": d["looks"], "end_jump": d["end_jump"]})
    return pd.DataFrame(rows, columns=KEY + ["metric_id", "looks", "end_jump"])


def primary_p_across_arms(table: pd.DataFrame, primary: int = PRIMARY) -> pd.Series:
    """Holm-adjusted p-value of the primary metric across the arms of one
    experiment that are read. A test with three treatments has three chances
    of a false "better"; with one treatment the p-value is unchanged."""
    rows = table[(table["metric_id"] == primary) & table["reading"].isin(READ)]
    out = pd.Series(np.nan, index=pd.MultiIndex.from_frame(rows[KEY]), dtype=float)
    for _, g in rows.groupby("experiment_id"):
        out.loc[list(zip(g["experiment_id"], g["variant_id"]))] = R.holm(g["p"])
    return out


def decide_arms(table: pd.DataFrame, ends: pd.DataFrame, primary: int = PRIMARY,
                direction: int = 1, margin: float = MARGIN, alpha: float = ALPHA) -> pd.DataFrame:
    """One row per treatment arm: the state of the primary, of each guard
    rail, and the verdict. `table` is Part 2's readout (one row per arm x
    metric), `ends` the output of `end_checks`."""
    p_arms = primary_p_across_arms(table, primary)
    jump = ends[ends["end_jump"]].groupby(KEY)["metric_id"].apply(list).to_dict()
    rows = []
    for (e, v), g in table.groupby(KEY, sort=True):
        g = g.set_index("metric_id")
        row = {"experiment_id": e, "variant_id": v, "primary_metric": primary, "direction": direction,
               "cautions": g["cautions"].iloc[0], "end_jump_metrics": ", ".join(map(str, jump.get((e, v), [])))}
        # Part 1's cautions apply to the whole arm, so its readings agree
        # except where a metric has no variance ("not testable").
        if (g["reading"] == "not decided").any():
            reading = "not decided"
        else:
            reading = "read with warning" if (g["reading"] == "read with warning").any() else "read"
        state, guards = None, {}
        if reading != "not decided" and primary in g.index and g.loc[primary, "reading"] in READ:
            pr = g.loc[primary]
            state = primary_state(pr.lift, pr.lift_lo, pr.lift_hi, pr.p, p_arms[(e, v)], direction, margin, alpha)
            row.update(primary_lift=pr.lift, primary_lo=pr.lift_lo, primary_hi=pr.lift_hi,
                       primary_p=pr.p, primary_p_arms=p_arms[(e, v)], primary_mde=pr.mde)
            for m, r in g.drop(index=primary).iterrows():
                guards[m] = ("not testable" if r.reading == "not testable" else
                             guard_state(r.lift, r.lift_lo, r.lift_hi, r.p, direction, margin, alpha))
        row["primary_state"] = state
        row["guards_worse"] = ", ".join(str(m) for m, s in guards.items() if s == "worse")
        row["guards_not_cleared"] = ", ".join(str(m) for m, s in guards.items() if s == "not cleared")
        row["guards_not_testable"] = ", ".join(str(m) for m, s in guards.items() if s == "not testable")
        row["verdict"], row["reason"] = verdict(reading, state, list(guards.values()), bool(jump.get((e, v))))
        rows.append(row)
    cols = KEY + ["primary_metric", "direction", "verdict", "reason", "primary_state", "primary_lift",
                  "primary_lo", "primary_hi", "primary_p", "primary_p_arms", "primary_mde",
                  "guards_worse", "guards_not_cleared", "guards_not_testable", "end_jump_metrics", "cautions"]
    return pd.DataFrame(rows, columns=cols)
