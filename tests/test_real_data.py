"""The committed data: SQL against independent pandas calculations, the
statements the report makes in words, and byte-for-byte regeneration."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abtest import checks as C, load, source
from scripts import data_checks_report as R


@pytest.fixture(scope="module")
def b():
    return R.build()


@pytest.fixture(scope="module")
def f(b):
    return R.facts(b)


def test_committed_file_is_the_pinned_one():
    assert source.sha256(source.RAW_PATH.read_bytes()) == source.CSV_SHA256


def test_shape_matches_the_datasheet(b, f):
    # the datasheet: 24,153 rows, 78 experiments, 99 treatments, 2 to 5 variants each
    assert (f["rows"], f["experiments"], f["arms"]) == (24_153, 78, 99)
    assert set(f["arms_per_experiment"]) == {1, 2, 3, 4}
    assert sum(k * v for k, v in f["arms_per_experiment"].items()) == 99
    assert f["variance_end"] == 5      # "5 out of 99 treatments ... missing the variance entries"


def test_arm_summary_agrees_with_pandas(b):
    raw, arms = b["raw"], b["arms"].set_index(["experiment_id", "variant_id"])
    m1 = raw[raw.metric_id == 1].sort_values("time_since_start")
    g = m1.groupby(["experiment_id", "variant_id"])
    assert (g.size() == arms["snapshots"]).all()
    assert (g.time_since_start.min() == arms["first_day"]).all()
    assert (g.time_since_start.max() == arms["last_day"]).all()
    assert (g.count_c.last() == arms["users_c"]).all()
    assert (g.count_t.last() == arms["users_t"]).all()
    gap = g.time_since_start.apply(lambda s: s.diff().max()).fillna(0)
    assert np.allclose(gap.sort_index(), arms["max_gap_days"].sort_index())
    assert np.allclose(arms["treatment_share"], arms["users_t"] / (arms["users_c"] + arms["users_t"]))


def test_srm_p_values_agree_with_scipy(b):
    for a in b["arms"].itertuples():
        expected = stats.chisquare([a.users_c, a.users_t]).pvalue
        assert a.srm_p == pytest.approx(expected, rel=1e-9, abs=1e-300)


def test_integrity_counts_agree_with_pandas(b, f):
    raw, v = b["raw"], f["viol"]
    assert v["duplicate_keys"] == raw.duplicated(["experiment_id", "variant_id", "metric_id", "time_since_start"]).sum()
    assert v["rows_missing_variance"] == (raw.variance_c.isna() | raw.variance_t.isna()).sum()
    assert v["empty_groups"] == ((raw.count_c <= 0) | (raw.count_t <= 0)).sum()
    binary = raw[raw.metric_id == 1]
    assert np.allclose(binary.variance_c, binary.mean_c * (1 - binary.mean_c), atol=1e-12)
    assert v["binary_variance_mismatch"] == 0


def test_statements_made_in_words(b, f):
    """Sentences in the report that name an experiment are checked here."""
    raw, v = b["raw"], f["viol"]
    empty = raw[(raw.count_c <= 0) | (raw.count_t <= 0)]
    assert set(empty.experiment_id) == {"64dc88"} and set(empty.time_since_start) == {3.0}
    assert empty.time_since_start.min() == raw[raw.experiment_id == "64dc88"].time_since_start.min()
    assert empty.variant_id.nunique() == 2 and len(empty) == v["empty_groups"] == 8
    per_snapshot = raw.groupby(["experiment_id", "variant_id", "time_since_start"]).metric_id.agg(list)
    short = per_snapshot[per_snapshot.apply(len) < 4]
    assert len(short) == v["snapshots_missing_a_metric"] == 1
    assert short.index[0][0] == "a4386f" and short.iloc[0] == [1]
    missing = raw[raw.variance_c.isna() | raw.variance_t.isna()]
    assert set(missing.metric_id) == {2, 3, 4}
    assert v["user_count_falls"] == f["falling_arms"] == 2
    # the named designs: 33/67 and 2/3
    named = b["arms"][b["arms"].fits_design != ""]
    assert sorted(named.fits_design) == ["1:2", "34:33", "34:33", "34:33", "34:33"]
    assert (named[named.fits_design == "34:33"].arms_in_experiment == 2).all()


def test_verdict_counts_add_up(b, f):
    arms = b["arms"]
    assert f["even"] + f["small"] + f["unequal"] == f["arms"]
    assert f["naive_alarms"] == f["small"] + f["unequal"]
    assert sum(f["stability"].values()) == f["arms"]
    assert f["clean_arms"] == (arms.cautions == "").sum()
    assert (arms[arms.split == "small imbalance"].users_short > 0).all()
    assert ((arms[arms.split == "small imbalance"].treatment_share - 0.5).abs() < C.NEAR_EVEN).all()
    assert ((arms[arms.split == "unequal"].treatment_share - 0.5).abs() >= C.NEAR_EVEN).all()
    assert (arms[arms.split == "even"].srm_p >= C.ALPHA).all()


def test_timing_only_arms_have_stable_weeks(b):
    arms = b["arms"]
    t = arms[arms.stability == "timing only"]
    assert (t.stability_p_snapshot < C.ALPHA).all() and (t.stability_p_weekly >= C.ALPHA).all()
    d = arms[arms.stability == "drifting"]
    assert (d.stability_p_weekly < C.ALPHA).all()


def test_new_users_add_back_to_the_final_counts(b):
    raw = b["raw"]
    m1 = raw[raw.metric_id == 1]
    for (e, v), g in m1.groupby(["experiment_id", "variant_id"]):
        inc = C.new_users(g.time_since_start, g.count_c, g.count_t)
        assert (inc[["new_c", "new_t"]] >= 0).all().all()
        week = C.weekly(inc)
        last = g.sort_values("time_since_start").iloc[-1]
        # a final snapshot is only skipped if a count fell at the very end
        if inc.day.max() == last.time_since_start:
            assert week.new_c.sum() == last.count_c and week.new_t.sum() == last.count_t


def test_growth_curves(b, f):
    curves = b["curves"]
    assert f["growth_experiments"] == 63
    end = curves[curves.run_share == 1.0]
    assert np.allclose(end.mean_vs_final, 1.0)
    assert len(curves) == 63 * 4 * 11
    for m in (1, 2, 3, 4):
        assert 0.5 < f["half_time"][m] < 1
        assert f["half_below"][m] > 0.9 * 63


def test_report_csvs_and_charts_regenerate_byte_for_byte(b, f, tmp_path):
    assert R.render(b, f) == (R.RESULTS / "DATA_CHECKS.md").read_text()
    b["arms"].to_csv(tmp_path / "a.csv", index=False, float_format="%.6g", lineterminator="\n")
    b["integrity"].to_csv(tmp_path / "i.csv", index=False, lineterminator="\n")
    b["metrics"].to_csv(tmp_path / "m.csv", index=False, float_format="%.6g", lineterminator="\n")
    assert (tmp_path / "a.csv").read_bytes() == (R.RESULTS / "arm_checks.csv").read_bytes()
    assert (tmp_path / "i.csv").read_bytes() == (R.RESULTS / "integrity_checks.csv").read_bytes()
    assert (tmp_path / "m.csv").read_bytes() == (R.RESULTS / "metric_profile.csv").read_bytes()
    R.chart_split(b["arms"], tmp_path / "1.png")
    R.chart_coverage(b["raw"], b["arms"], tmp_path / "2.png")
    R.chart_growth(b["curves"], tmp_path / "3.png")
    for made, kept in (("1.png", "01_sample_ratio.png"), ("2.png", "02_coverage.png"), ("3.png", "03_metrics_grow.png")):
        assert (tmp_path / made).read_bytes() == (R.CHARTS / kept).read_bytes(), kept
