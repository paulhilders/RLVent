-- Description: This query creates a new merged_vitalsigns table, which contains
--   the vital signs for patients in the MIMIC-IV database.
-- Execution time: A few seconds.

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS merged_vitalsigns; CREATE TABLE merged_vitalsigns AS

select
  	ce.subject_id
  , ce.stay_id
  , ce.charttime
  , gcs.gcs
  , avg(heart_rate) as HeartRate
  , avg(sbp) as SysBP
  , avg(dbp) as DiasBP
  , avg(mbp) as MeanBP
  , avg(resp_rate) as RespRate
  , avg(temperature) as TempC
  , avg(spo2) as SpO2
  from vitalsign ce
  LEFT JOIN gcs
  ON (ce.stay_id = gcs.stay_id)
  AND (ce.charttime = gcs.charttime)
  group by ce.subject_id, ce.stay_id, ce.charttime, gcs.gcs

