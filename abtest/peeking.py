"""Part 3: what looking at a test every day does to its error rate.

Ways to decide the same test:

* "end": one test at the last snapshot (Part 2).
* "peek": test at every daily look and call a result the first time p < 0.05.
* "always": a rule built to be checked at every look. It uses the mixture
  sequential probability ratio test (Robbins 1970; Johari, Pekelis and Walsh
  2017): the evidence for "some change" against "no change", averaged over a
  normal spread of possible changes. Under "no change" the chance that this
  ratio ever reaches 1 / alpha is at most alpha, however often it is checked.
* "peek_confirmed" and "always_confirmed": the same two rules, but a result
  is only called when the signal is there on two looks in a row. One bad
  snapshot cannot trigger them.

The replay also marks "blips": looks at which the test statistic jumps and,
within three looks, jumps back (`blip_mask`). They are data faults, and a rule
that stops at the first signal acts on them.

Two things are done with them: the real experiments are replayed look by look
(`replay`), and tests where nothing changed are simulated on each arm's own
look schedule (`simulate_null`).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from . import readout as R
from .checks import MIN_DAYS

ALPHA = R.ALPHA                 # the same 5% level as Part 2
FIRST_DECISION_DAY = MIN_DAYS   # no rule may stop a test before it has run a full week (Part 1)
TAU = R.REFERENCE_LIFT          # spread of the changes the always-valid rule looks for: 1% of the control mean

# Part 1 cautions that make an arm's daily record unfit for a replay.
NOT_REPLAYED = ("first week not recorded", "gap in snapshots")

# A blip is a run of 1 to MAX_BLIP_LOOKS looks at which the test statistic has
# moved by more than BLIP_Z standard errors from the look before the run, and
# moves back by more than BLIP_Z at the look after it. A few days of new users
# cannot do that (the simulation in `null_rates` counts how often chance does).
BLIP_Z = 3.0
MAX_BLIP_LOOKS = 3

RULES = ("peek", "peek_confirmed", "always", "always_confirmed")

SEED = 20261123
PATHS = 20_000           # simulated no-change tests per arm
PATHS_EQUAL = 200_000    # simulated no-change tests per point of the equal-looks curve


# ------------------------------------------------------------ the rule
def mixture_ratio(lift: float, lift_se: float, tau: float = TAU) -> float:
    """Mixture likelihood ratio for a relative change estimated as `lift`
    with standard error `lift_se`.

    "No change" (true lift 0) is compared with true lifts drawn from a normal
    distribution with mean 0 and standard deviation `tau`. With V = lift_se^2:

        ratio = sqrt(V / (V + tau^2)) * exp(tau^2 * lift^2 / (2 * V * (V + tau^2)))
    """
    if lift_se <= 0 or tau <= 0:
        raise ValueError("the standard error and tau must be positive")
    v, t2 = lift_se ** 2, tau ** 2
    return math.sqrt(v / (v + t2)) * math.exp(t2 * lift ** 2 / (2 * v * (v + t2)))


def always_valid_bound(lift_se, tau: float = TAU, alpha: float = ALPHA) -> np.ndarray:
    """The |z| at which the mixture ratio reaches 1 / alpha, for a look with
    the given standard error. With r = tau^2 / V:

        z^2 = (2 * ln(1 / alpha) + ln(1 + r)) * (1 + r) / r
    """
    r = tau ** 2 / np.asarray(lift_se, dtype=float) ** 2
    return np.sqrt((2 * math.log(1 / alpha) + np.log1p(r)) * (1 + r) / r)


def _blips_1d(z: np.ndarray) -> np.ndarray:
    out = np.zeros(len(z), dtype=bool)
    n, j, ref = len(z), 1, 0            # ref: the last look that is not in a blip
    while j < n - 1:
        found = 0
        up = z[j] - z[ref]
        if abs(up) > BLIP_Z:
            for length in range(1, MAX_BLIP_LOOKS + 1):
                end = j + length - 1    # last look of the run; end + 1 must exist
                if end + 1 > n - 1 or (length > 1 and abs(z[end] - z[end - 1]) > BLIP_Z):
                    break
                back = z[end] - z[end + 1]
                if abs(back) > BLIP_Z and up * back > 0:
                    found = length
                    break
        if found:
            out[j:j + found] = True
            j += found
        else:
            ref = j
            j += 1
    return out


def blip_mask(z: np.ndarray) -> np.ndarray:
    """Looks that belong to a blip: z jumps by more than BLIP_Z from the last
    look outside a blip, stays there for 1 to MAX_BLIP_LOOKS looks (moving by
    no more than BLIP_Z between them) and then jumps back by more than BLIP_Z.
    Works along the last axis. The first and the last look cannot be judged
    and are never marked."""
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        return _blips_1d(z)
    out = np.zeros(z.shape, dtype=bool)
    if z.shape[-1] < 3:
        return out
    # A row can only hold a blip if it has a step larger than BLIP_Z somewhere.
    rows = np.argwhere((np.abs(np.diff(z, axis=-1)) > BLIP_Z).any(axis=-1))
    for idx in map(tuple, rows):
        out[idx] = _blips_1d(z[idx])
    return out


def episodes(mask) -> int:
    """Number of separate runs of True in a 1-D mask."""
    m = np.asarray(mask, dtype=bool)
    return int(m[0]) + int((m[1:] & ~m[:-1]).sum()) if len(m) else 0


def confirmed(hit: np.ndarray) -> np.ndarray:
    """True where `hit` is true at this look and at the one before it (along
    the last axis)."""
    hit = np.asarray(hit, dtype=bool)
    out = np.zeros(hit.shape, dtype=bool)
    out[..., 1:] = hit[..., 1:] & hit[..., :-1]
    return out


def info_fractions(users_c, users_t) -> np.ndarray:
    """Share of the final information available at each look. For a comparison
    of two means, information grows as n_c * n_t / (n_c + n_t)."""
    c, t = np.asarray(users_c, dtype=float), np.asarray(users_t, dtype=float)
    info = c * t / (c + t)
    if (np.diff(info) < 0).any():
        raise ValueError("information falls between looks")
    return info / info[-1]


# ------------------------------------------------------------ real tests, replayed
def look_stats(looks: pd.DataFrame) -> pd.DataFrame:
    """The Part 2 test, the always-valid ratio and the blip mark at each look
    of one arm x metric (rows of sql/daily_looks.sql). Looks that cannot be
    tested (no variance, an empty group, a group in which every user has the
    same value, a zero control mean) are dropped."""
    rows = []
    for r in looks.sort_values("day").itertuples(index=False):
        if (pd.isna(r.variance_c) or pd.isna(r.variance_t) or r.users_c < 2 or r.users_t < 2
                or r.mean_c <= 0 or r.variance_c <= 0 or r.variance_t <= 0):
            continue
        _, _, _, p = R.welch(r.mean_c, r.mean_t, r.variance_c, r.variance_t, r.users_c, r.users_t)
        lift, lift_se, _, _ = R.relative_lift(r.mean_c, r.mean_t, r.variance_c, r.variance_t,
                                              r.users_c, r.users_t)
        rows.append({"day": r.day, "users_c": r.users_c, "users_t": r.users_t, "lift": lift,
                     "lift_se": lift_se, "z": lift / lift_se, "p": p,
                     "ratio": mixture_ratio(lift, lift_se)})
    out = pd.DataFrame(rows, columns=["day", "users_c", "users_t", "lift", "lift_se", "z", "p", "ratio"])
    out["blip"] = blip_mask(out["z"].to_numpy())
    return out


def decision_looks(days, first_day: float = FIRST_DECISION_DAY) -> np.ndarray:
    """Looks at which a rule may stop the test: from `first_day` on. The last
    look always counts (the test is over, so it is decided then)."""
    d = np.asarray(days, dtype=float) >= first_day
    if len(d):
        d[-1] = True
    return d


def decide(s: pd.DataFrame, first_day: float = FIRST_DECISION_DAY, alpha: float = ALPHA) -> dict:
    """Decide one arm x metric by every rule from its looks (output of
    `look_stats`). For each rule: whether it stops, the day, the relative
    change on that day and whether the look that triggered it is in a blip.
    `end_jump`: the last look differs by more than BLIP_Z from the last
    earlier look that is not in a blip (a possible fault that no later look
    can confirm)."""
    last = s.iloc[-1]
    d = decision_looks(s["day"], first_day)
    sig = (s["p"] < alpha).to_numpy()
    strong = (s["ratio"] >= 1 / alpha).to_numpy()
    hits = {"peek": sig & d, "peek_confirmed": confirmed(sig) & d,
            "always": strong & d, "always_confirmed": confirmed(strong) & d}
    before = s["z"].iloc[:-1][~s["blip"].iloc[:-1].to_numpy()]
    out = {"looks": len(s), "decision_looks": int(d.sum()),
           "blips": episodes(s["blip"]), "blip_looks": int(s["blip"].sum()),
           "max_abs_z": float(s["z"].abs().max()),
           "last_day": last.day, "lift_end": last.lift, "p_end": last.p, "sig_end": bool(last.p < alpha),
           "end_jump": bool(len(before) and abs(last.z - before.iloc[-1]) > BLIP_Z)}
    for name, hit in hits.items():
        i = np.flatnonzero(hit)
        out[f"{name}_stops"] = bool(len(i))
        out[f"{name}_day"] = float(s["day"].iloc[i[0]]) if len(i) else float("nan")
        out[f"{name}_lift"] = float(s["lift"].iloc[i[0]]) if len(i) else float("nan")
        out[f"{name}_on_blip"] = bool(len(i) and s["blip"].iloc[i[0]])
    return out


def replayed_arms(arm_checks: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """The arms whose daily record can be replayed: read in Part 2, first
    recorded within a week, and no gap of three days or more."""
    read = table[table["reading"].isin(["read", "read with warning"])]
    keys = read[["experiment_id", "variant_id"]].drop_duplicates()
    arms = keys.merge(arm_checks, on=["experiment_id", "variant_id"])
    bad = arms["cautions"].fillna("").str.split("; ").apply(lambda i: any(c in NOT_REPLAYED for c in i))
    return arms[~bad].reset_index(drop=True)


def replay(looks: pd.DataFrame, table: pd.DataFrame, arm_checks: pd.DataFrame
           ) -> tuple[pd.DataFrame, dict]:
    """Replay every arm x metric that Part 2 read and whose record allows it.

    Returns (one row per test, {(experiment, arm, metric): its looks})."""
    arms = replayed_arms(arm_checks, table)
    keep = set(zip(arms["experiment_id"], arms["variant_id"]))
    read = table[table["reading"].isin(["read", "read with warning"])]
    by_test = {k: g for k, g in looks.groupby(["experiment_id", "variant_id", "metric_id"])}
    rows, paths = [], {}
    for r in read.itertuples(index=False):
        if (r.experiment_id, r.variant_id) not in keep:
            continue
        key = (r.experiment_id, r.variant_id, r.metric_id)
        s = look_stats(by_test[key])
        paths[key] = s
        rows.append({"experiment_id": r.experiment_id, "variant_id": r.variant_id,
                     "metric_id": r.metric_id, "reading": r.reading, **decide(s),
                     "p_holm": r.p_holm, "lift_se_end": float(s["lift_se"].iloc[-1])})
    return pd.DataFrame(rows), paths


# ------------------------------------------------------------ no-change tests, simulated
def null_z(fractions, paths: int, rng: np.random.Generator) -> np.ndarray:
    """z-statistics of `paths` simulated tests in which nothing changed, at
    looks taken at the given information fractions (rows: tests, columns:
    looks). Between looks the statistic moves like a random walk in
    information time, which is how a difference of two means of independent
    users behaves as users arrive."""
    t = np.asarray(fractions, dtype=float)
    if len(t) == 0 or (t <= 0).any() or (np.diff(t) < 0).any():
        raise ValueError("fractions must be positive and non-decreasing")
    steps = np.sqrt(np.diff(np.concatenate([[0.0], t])))
    walk = np.cumsum(rng.standard_normal((paths, len(t))) * steps, axis=1)
    return walk / np.sqrt(t)


def ratio_from_z(z: np.ndarray, fractions, tau_over_se_end: float) -> np.ndarray:
    """The mixture ratio written in terms of z. With r = tau^2 / V at the end
    and V at a look = V_end / fraction:

        ratio = (1 + r t)^(-1/2) * exp(z^2 / 2 * r t / (1 + r t))
    """
    rt = tau_over_se_end ** 2 * np.asarray(fractions, dtype=float)
    return np.exp(0.5 * z ** 2 * rt / (1 + rt)) / np.sqrt(1 + rt)


def null_rates(z: np.ndarray, fractions, decision, tau_over_se_end: dict,
               alpha: float = ALPHA) -> dict:
    """Share of the simulated no-change tests in `z` that each rule calls a
    result. `decision`: which looks a rule may stop at; `tau_over_se_end`: for
    each metric, tau divided by the final standard error of the relative
    change (it sets how the always-valid rule behaves on that test)."""
    t = np.asarray(fractions, dtype=float)
    d = np.asarray(decision, dtype=bool)
    sig = np.abs(z) > stats.norm.isf(alpha / 2)
    out = {"end": float(sig[:, -1].mean()),
           "peek": float((sig & d).any(axis=1).mean()),
           "peek_confirmed": float((confirmed(sig) & d).any(axis=1).mean()),
           "always": {}, "always_confirmed": {},
           "blip_looks": int(blip_mask(z).sum()), "looks_simulated": int(z.size)}
    for metric, k in tau_over_se_end.items():
        strong = ratio_from_z(z, t, k) >= 1 / alpha
        out["always"][metric] = float((strong & d).any(axis=1).mean())
        out["always_confirmed"][metric] = float((confirmed(strong) & d).any(axis=1).mean())
    return out


def simulate_null(fractions, decision, tau_over_se_end: dict, paths: int = PATHS,
                  seed: int = SEED, alpha: float = ALPHA) -> dict:
    """False-alarm rates of one arm's look schedule when nothing changed."""
    z = null_z(fractions, paths, np.random.default_rng(seed))
    return null_rates(z, fractions, decision, tau_over_se_end, alpha)


def equal_looks_rate(looks: int, paths: int = PATHS_EQUAL, seed: int = SEED,
                     alpha: float = ALPHA) -> float:
    """False-alarm rate of testing at `looks` equally spaced looks and calling
    a result at the first p < alpha, when nothing changed."""
    t = np.arange(1, looks + 1) / looks
    z = null_z(t, paths, np.random.default_rng(seed))
    return float((np.abs(z) > stats.norm.isf(alpha / 2)).any(axis=1).mean())
