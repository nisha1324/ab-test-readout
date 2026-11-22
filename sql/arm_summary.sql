-- One row per treatment arm (experiment x variant): how long it ran, how often
-- it was measured, and the user counts at its last snapshot.
-- User counts are the same for all four metrics (see integrity.sql), so the
-- timeline is read from metric 1.

WITH steps AS (
    SELECT experiment_id, variant_id, time_since_start, count_c, count_t,
           time_since_start - lag(time_since_start) OVER w AS gap_days,
           count_c - lag(count_c) OVER w AS new_c,
           count_t - lag(count_t) OVER w AS new_t,
           row_number() OVER (PARTITION BY experiment_id, variant_id
                              ORDER BY time_since_start DESC) AS from_last
    FROM snapshots
    WHERE metric_id = 1
    WINDOW w AS (PARTITION BY experiment_id, variant_id ORDER BY time_since_start)
),
timeline AS (
    SELECT experiment_id, variant_id,
           count(*) AS snapshots,
           min(time_since_start) AS first_day,
           max(time_since_start) AS last_day,
           coalesce(max(gap_days), 0) AS max_gap_days,
           -- no new user in either group since the previous snapshot
           count(*) FILTER (WHERE new_c = 0 AND new_t = 0) AS frozen_snapshots,
           -- one group gained users and the other gained none
           count(*) FILTER (WHERE (new_c = 0) <> (new_t = 0)) AS one_sided_snapshots,
           count(*) FILTER (WHERE new_c < 0 OR new_t < 0) AS falling_snapshots,
           max(count_c) FILTER (WHERE from_last = 1) AS users_c,
           max(count_t) FILTER (WHERE from_last = 1) AS users_t
    FROM steps
    GROUP BY experiment_id, variant_id
),
variance_gaps AS (
    -- metrics with a missing variance at any snapshot, and at the last one
    SELECT s.experiment_id, s.variant_id,
           count(DISTINCT s.metric_id) AS metrics_missing_variance,
           count(DISTINCT s.metric_id) FILTER (WHERE s.time_since_start = t.last_day)
               AS metrics_missing_variance_at_end
    FROM snapshots s
    JOIN timeline t USING (experiment_id, variant_id)
    WHERE s.variance_c IS NULL OR s.variance_t IS NULL
    GROUP BY s.experiment_id, s.variant_id
)
SELECT t.experiment_id,
       t.variant_id,
       count(*) OVER (PARTITION BY t.experiment_id) AS arms_in_experiment,
       t.snapshots,
       t.first_day,
       t.last_day,
       t.max_gap_days,
       t.frozen_snapshots,
       t.one_sided_snapshots,
       t.falling_snapshots,
       CAST(t.users_c AS BIGINT) AS users_c,
       CAST(t.users_t AS BIGINT) AS users_t,
       CAST(t.users_c + t.users_t AS BIGINT) AS users,
       t.users_t / (t.users_c + t.users_t) AS treatment_share,
       coalesce(v.metrics_missing_variance, 0) AS metrics_missing_variance,
       coalesce(v.metrics_missing_variance_at_end, 0) AS metrics_missing_variance_at_end
FROM timeline t
LEFT JOIN variance_gaps v USING (experiment_id, variant_id)
ORDER BY t.experiment_id, t.variant_id;
