-- Description: This query creates a new getmechvent table, which contains
--   the mechanical ventilation status for patients in the MIMIC-IV database.
-- Execution time: Roughly 1 minute.
-- Number of Rows: 109369 (109 thousand)

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS getmechvent; CREATE TABLE getmechvent AS

SELECT
  	ic.subject_id
  , ic.hadm_id
  , vent.stay_id
  , vent.starttime AS charttime
  , vent.ventilation_status
  , MAX(CASE
    WHEN vent.ventilation_status = 'InvasiveVent' THEN 1
    ELSE 0
    END
  ) AS MechVent
  FROM ventilation vent
  LEFT JOIN icustays ic
  ON ic.stay_id=vent.stay_id
  GROUP BY ic.subject_id, ic.hadm_id, vent.stay_id, charttime, vent.ventilation_status
  ORDER BY ic.subject_id, ic.hadm_id, vent.stay_id, charttime, vent.ventilation_status