-- Description: This query creates a new getinvasiveventdurations table, which
--     contains the durations of invasive ventilation sessions for the patients
--     in the AmsterdamUMCdb database. This table is useful for filtering out
--     ventilation sessions according to specified exclusion/inclusion criteria.
-- Number of rows: 18846 (18.8 thousand)
-- Time required: Roughly 10 seconds

SET search_path TO public, amsterdamumcdb, amsterdamumcdb_derived;

DROP TABLE IF EXISTS getinvasiveventdurations; CREATE TABLE getinvasiveventdurations AS

WITH vent_sessions AS (
  SELECT mv.admissionid, mv.charttime, mv.mechvent
    , SUM(mv.new_session) OVER (PARTITION BY mv.admissionid ORDER BY mv.admissionid, mv.charttime ASC ROWS UNBOUNDED PRECEDING) +1 AS vent_session
  FROM (
    SELECT t.admissionid, t.charttime, t.mechvent
      , (CASE
        WHEN t.mechvent <>  LAG(t.mechvent) OVER (ORDER BY t.admissionid, t.charttime) THEN 1
        -- WHEN t.admissionid <>  LAG(t.admissionid) OVER (ORDER BY t.admissionid, t.charttime) THEN 0
      ELSE 0 END) AS new_session
    FROM getmechvent t
  ) mv
  -- GROUP BY mv.admissionid, mv.charttime, mv.mechvent, mv.new_session
  ORDER BY mv.admissionid, mv.charttime
)

-- SELECT * FROM vent_sessions

SELECT admissionid
  , vent_session
  , mechvent
  , MIN(charttime) AS starttime
  , MAX(charttime) AS endtime
  , (MAX(charttime) - MIN(charttime)) / 60 AS vent_duration_h
FROM vent_sessions
GROUP BY admissionid, vent_session, mechvent
ORDER BY admissionid, vent_session
