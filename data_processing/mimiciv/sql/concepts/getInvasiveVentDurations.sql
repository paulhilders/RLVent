-- Description: This query creates a new getinvasiveventdurations table, which
--     contains the duration of invasive ventilation for the patients in the
--     MIMIC-IV database. This table is useful for filtering out ventilation
--     sessions according to specified exclusion/inclusion criteria.
-- Execution time: A few seconds.

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS getinvasiveventdurations; CREATE TABLE getinvasiveventdurations AS

SELECT stay_id, ventilation_status
    , MIN(starttime) AS starttime
    , MAX(endtime) AS endtime
    , AVG((endtime - starttime)) AS vent_duration_h
FROM ventilation
WHERE ventilation_status = 'InvasiveVent'
GROUP BY stay_id, ventilation_status
ORDER BY stay_id, ventilation_status;