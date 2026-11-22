-- One row per metric, read at each arm's last snapshot (control group).
-- "sd_over_mean" is the standard deviation divided by the mean: the larger it
-- is, the more users a test needs to detect the same relative change.

WITH ranked AS (
    SELECT *,
           row_number() OVER (PARTITION BY experiment_id, variant_id, metric_id
                              ORDER BY time_since_start DESC) AS from_last
    FROM snapshots
),
final AS (
    SELECT * FROM ranked WHERE from_last = 1
)
SELECT f.metric_id,
       count(*) AS arms,
       min(f.mean_c) AS min_mean,
       median(f.mean_c) AS median_mean,
       max(f.mean_c) AS max_mean,
       median(sqrt(f.variance_c) / f.mean_c) AS median_sd_over_mean,
       count(*) FILTER (WHERE f.variance_c IS NULL OR f.variance_t IS NULL) AS arms_missing_variance
FROM final f
GROUP BY f.metric_id
ORDER BY f.metric_id;
