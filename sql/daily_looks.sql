-- One row per treatment arm x metric x day: what someone checking the test
-- once a day would have seen. Part 3 replays these looks.
--
-- A daily look is the latest snapshot in each 24-hour window (d-1, d] since
-- the start of the experiment, so an arm recorded every 12 hours gets one look
-- a day, like an arm recorded every 24. A day with no snapshot has no look.
-- A snapshot where either cumulative user count is lower than at an earlier
-- snapshot is not a look (a cumulative count cannot really fall).
-- User counts are shared by the four metrics (see integrity.sql), so the
-- timeline is read from metric 1.

WITH timeline AS (
    SELECT experiment_id, variant_id, time_since_start, count_c, count_t,
           max(count_c) OVER earlier AS earlier_c,
           max(count_t) OVER earlier AS earlier_t
    FROM snapshots
    WHERE metric_id = 1
    WINDOW earlier AS (PARTITION BY experiment_id, variant_id ORDER BY time_since_start
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
),
kept AS (
    SELECT experiment_id, variant_id, time_since_start,
           CAST(ceil(time_since_start) AS INTEGER) AS look_day
    FROM timeline
    WHERE count_c >= coalesce(earlier_c, 0)
      AND count_t >= coalesce(earlier_t, 0)
),
daily AS (
    SELECT experiment_id, variant_id, look_day, max(time_since_start) AS look_time
    FROM kept
    GROUP BY experiment_id, variant_id, look_day
)
SELECT s.experiment_id,
       s.variant_id,
       s.metric_id,
       d.look_day,
       s.time_since_start AS day,
       CAST(s.count_c AS BIGINT) AS users_c,
       CAST(s.count_t AS BIGINT) AS users_t,
       s.mean_c,
       s.mean_t,
       s.variance_c,
       s.variance_t
FROM snapshots s
JOIN daily d
  ON s.experiment_id = d.experiment_id
 AND s.variant_id = d.variant_id
 AND s.time_since_start = d.look_time
ORDER BY s.experiment_id, s.variant_id, s.metric_id, s.time_since_start;
