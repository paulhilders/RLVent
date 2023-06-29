-- Description: This query creates a new getdemographics table, which contains
--          demographic information about the patients in the MIMIC-IV database.
-- Source: https://github.com/florisdenhengst/ventai/blob/main/demographics.sql
-- Execution time: Roughly 1 minute.
-- Number of Rows: 73141

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS getdemographics; CREATE TABLE getdemographics AS

WITH prev_hadms as (
  SELECT DISTINCT subject_id
  , hadm_id
  , LAG(hadm_id, 1) OVER(
	PARTITION BY subject_id
	ORDER BY ADMITTIME ASC
  ) AS prev_hadm_id
  from admissions
)
SELECT ie.subject_id, ie.hadm_id, ie.stay_id

-- patient level factors
, pat.gender, pat.dod

-- hospital level factors
, adm.admittime, adm.dischtime, adm.deathtime
, DATETIME_DIFF(adm.dischtime, adm.admittime, 'DAY') as los_hospital
, pat.anchor_age as admission_age
, adm.race
, case when race in
  (
       'WHITE' --  40996
     , 'WHITE - RUSSIAN' --    164
     , 'WHITE - OTHER EUROPEAN' --     81
     , 'WHITE - BRAZILIAN' --     59
     , 'WHITE - EASTERN EUROPEAN' --     25
  ) then 'white'
  when race in
  (
      'BLACK/AFRICAN AMERICAN' --   5440
    , 'BLACK/CAPE VERDEAN' --    200
    , 'BLACK/HAITIAN' --    101
    , 'BLACK/AFRICAN' --     44
    , 'CARIBBEAN ISLAND' --      9
  ) then 'black'
  when race in
    (
      'HISPANIC OR LATINO' --   1696
    , 'HISPANIC/LATINO - PUERTO RICAN' --    232
    , 'HISPANIC/LATINO - DOMINICAN' --     78
    , 'HISPANIC/LATINO - GUATEMALAN' --     40
    , 'HISPANIC/LATINO - CUBAN' --     24
    , 'HISPANIC/LATINO - SALVADORAN' --     19
    , 'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)' --     13
    , 'HISPANIC/LATINO - MEXICAN' --     13
    , 'HISPANIC/LATINO - COLOMBIAN' --      9
    , 'HISPANIC/LATINO - HONDURAN' --      4
  ) then 'hispanic'
  when race in
  (
      'ASIAN' --   1509
    , 'ASIAN - CHINESE' --    277
    , 'ASIAN - ASIAN INDIAN' --     85
    , 'ASIAN - VIETNAMESE' --     53
    , 'ASIAN - FILIPINO' --     25
    , 'ASIAN - CAMBODIAN' --     17
    , 'ASIAN - OTHER' --     17
    , 'ASIAN - KOREAN' --     13
    , 'ASIAN - JAPANESE' --      7
    , 'ASIAN - THAI' --      4
  ) then 'asian'
  when race in
  (
       'AMERICAN INDIAN/ALASKA NATIVE' --     51
     , 'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE' --      3
  ) then 'native'
  when race in
  (
      'UNKNOWN/NOT SPECIFIED' --   4523
    , 'UNABLE TO OBTAIN' --    814
    , 'PATIENT DECLINED TO ANSWER' --    559
  ) then 'unknown'
  else 'other' end as ethnicity_grouped
, CASE WHEN pat.dod <= DATETIME_ADD(admittime, INTERVAL '90 DAY') THEN True ELSE False END AS mort90day
, CASE WHEN adm.deathtime <= adm.dischtime THEN True ELSE False END AS hospmort
, adm.hospital_expire_flag
, DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS hospstay_seq
, CASE
    WHEN DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) = 1 THEN True
    ELSE False END AS first_hosp_stay

-- icu level factors
, ie.intime, ie.outtime
, DATETIME_DIFF(ie.outtime, ie.intime, 'DAY') as los_icu
, DATETIME_DIFF(ie.outtime, ie.intime, 'HOUR') as los_icu_h
, DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) AS icustay_seq

-- first ICU stay *for the current hospitalization*
, CASE
    WHEN DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) = 1 THEN True
    ELSE False END AS first_icu_stay
, CASE
    WHEN DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) = 1 THEN False
    ELSE True END AS icu_readmission
-- premorbidity based on *previous hospitalization*
, eh.elixhauser_vanwalraven

FROM icustays ie
INNER JOIN admissions adm
    ON ie.hadm_id = adm.hadm_id
-- premorbidity based on *previous hospitalization*
INNER JOIN prev_hadms
    ON ie.hadm_id = prev_hadms.hadm_id
LEFT JOIN getelixhausercomorbidityindex eh
    ON prev_hadms.prev_hadm_id = eh.hadm_id
INNER JOIN patients pat
    ON ie.subject_id = pat.subject_id
-- WHERE adm.has_chartevents_data = 1
ORDER BY ie.subject_id, adm.admittime, ie.intime;