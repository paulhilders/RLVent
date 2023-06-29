-- Description: This query creates a new getweight table, which contains
--      the weight for patients in the MIMIC-IV database.
-- Execution time: Roughly 1 minute.
-- Number of Rows: 273734 (273 thousand)

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS getweight; CREATE TABLE getweight AS

SELECT
  	subject_id
  , hadm_id
  , wd.stay_id
  , wd.starttime AS charttime
  , AVG(wd.weight)
  FROM weight_durations wd
  LEFT JOIN icustays ic
  ON ic.stay_id=wd.stay_id
  WHERE weight > 20 and weight < 300
  GROUP BY ic.subject_id, ic.hadm_id, wd.stay_id, charttime
  ORDER BY ic.subject_id, ic.hadm_id, wd.stay_id, charttime