-- Structural checks on the raw table `snapshots`
-- (one row per experiment x treatment arm x metric x snapshot time).
-- Returns one row per check: how many rows or groups break the rule.
-- A zero means the rule holds everywhere.

SELECT 'duplicate_keys' AS check_name,
       'experiment, arm, metric and time identify a row' AS rule,
       count(*) AS violations
FROM (SELECT 1 FROM snapshots
      GROUP BY experiment_id, variant_id, metric_id, time_since_start
      HAVING count(*) > 1)

UNION ALL
SELECT 'arms_without_four_metrics', 'every arm reports metrics 1 to 4', count(*)
FROM (SELECT 1 FROM snapshots GROUP BY experiment_id, variant_id
      HAVING count(DISTINCT metric_id) <> 4)

UNION ALL
SELECT 'snapshots_missing_a_metric', 'every snapshot of an arm reports all four metrics', count(*)
FROM (SELECT 1 FROM snapshots GROUP BY experiment_id, variant_id, time_since_start
      HAVING count(DISTINCT metric_id) <> 4)

UNION ALL
SELECT 'user_counts_differ_by_metric', 'the four metrics of a snapshot share one pair of user counts', count(*)
FROM (SELECT 1 FROM snapshots GROUP BY experiment_id, variant_id, time_since_start
      HAVING count(DISTINCT count_c) > 1 OR count(DISTINCT count_t) > 1)

UNION ALL
SELECT 'control_differs_between_arms', 'arms of one experiment are compared with the same control group', count(*)
FROM (SELECT 1 FROM snapshots GROUP BY experiment_id, metric_id, time_since_start
      HAVING count(DISTINCT count_c) > 1 OR count(DISTINCT mean_c) > 1)

UNION ALL
SELECT 'non_integer_user_counts', 'user counts are whole numbers', count(*)
FROM snapshots WHERE count_c <> round(count_c) OR count_t <> round(count_t)

UNION ALL
SELECT 'empty_groups', 'both groups have at least one user', count(*)
FROM snapshots WHERE count_c <= 0 OR count_t <= 0

UNION ALL
SELECT 'missing_means', 'every row has both means', count(*)
FROM snapshots WHERE mean_c IS NULL OR mean_t IS NULL

UNION ALL
SELECT 'negative_means', 'no metric is negative on average', count(*)
FROM snapshots WHERE mean_c < 0 OR mean_t < 0

UNION ALL
SELECT 'binary_metric_out_of_range', 'metric 1 is a share between 0 and 1', count(*)
FROM snapshots WHERE metric_id = 1 AND (mean_c NOT BETWEEN 0 AND 1 OR mean_t NOT BETWEEN 0 AND 1)

UNION ALL
SELECT 'binary_variance_mismatch', 'metric 1 variance equals mean x (1 - mean)', count(*)
FROM snapshots
WHERE metric_id = 1
  AND (abs(variance_c - mean_c * (1 - mean_c)) > 1e-9 OR abs(variance_t - mean_t * (1 - mean_t)) > 1e-9)

UNION ALL
SELECT 'rows_missing_variance', 'every row has both variances', count(*)
FROM snapshots WHERE variance_c IS NULL OR variance_t IS NULL

UNION ALL
SELECT 'user_count_falls', 'cumulative user counts never fall from one snapshot to the next', count(*)
FROM (SELECT count_c - lag(count_c) OVER w AS new_c, count_t - lag(count_t) OVER w AS new_t
      FROM snapshots WHERE metric_id = 1
      WINDOW w AS (PARTITION BY experiment_id, variant_id ORDER BY time_since_start))
WHERE new_c < 0 OR new_t < 0
ORDER BY check_name;
