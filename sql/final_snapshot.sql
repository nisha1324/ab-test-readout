-- One row per treatment arm x metric at the arm's last snapshot: the figures
-- the readout is computed from. An arm's last snapshot is the latest time at
-- which it reports metric 1 (user counts are shared by the four metrics, see
-- integrity.sql). A metric that is absent at that time gets no row.

WITH last_snapshot AS (
    SELECT experiment_id, variant_id, max(time_since_start) AS last_day
    FROM snapshots
    WHERE metric_id = 1
    GROUP BY experiment_id, variant_id
)
SELECT s.experiment_id,
       s.variant_id,
       s.metric_id,
       s.time_since_start AS day,
       CAST(s.count_c AS BIGINT) AS users_c,
       CAST(s.count_t AS BIGINT) AS users_t,
       s.mean_c,
       s.mean_t,
       s.variance_c,
       s.variance_t
FROM snapshots s
JOIN last_snapshot l
  ON s.experiment_id = l.experiment_id
 AND s.variant_id = l.variant_id
 AND s.time_since_start = l.last_day
ORDER BY s.experiment_id, s.variant_id, s.metric_id;
