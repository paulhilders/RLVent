-- Description: This query creates a new getvasopressors table, which
--     contains the vasopressor rates and total vasopressor dose
--     for the patients in the AmsterdamUMCdb database.
-- Inspired by: https://github.com/AmsterdamUMC/AmsterdamUMCdb/blob/master/amsterdamumcdb/sql/common/vasopressors_inotropes.sql
-- Number of rows: 295291 (295 thousand)
-- Time required: Roughly 3 seconds

SET search_path TO public, amsterdamumcdb, amsterdamumcdb_derived;

DROP TABLE IF EXISTS getvasopressors; CREATE TABLE getvasopressors AS

WITH vaso_rates AS (
  SELECT di.admissionid
    , (start - a.admittedat)/(1000*60) AS starttime
    , (stop - a.admittedat)/(1000*60) AS stoptime
    , item
    , itemid
    , duration
    , rate
    , rateunit
    , dose
    , doseunit
    , doseunitid
    , doserateperkg
    , doserateunitid
    , doserateunit
    , CASE
        WHEN weightgroup LIKE '59' THEN 55
        WHEN weightgroup LIKE '60' THEN 65
        WHEN weightgroup LIKE '70' THEN 75
        WHEN weightgroup LIKE '80' THEN 85
        WHEN weightgroup LIKE '90' THEN 95
        WHEN weightgroup LIKE '100' THEN 105
        WHEN weightgroup LIKE '110' THEN 115
        ELSE 80 --mean weight for all years
      END as patientweight
  FROM drugitems di
  LEFT JOIN admissions a
    ON di.admissionid = a.admissionid
  WHERE ordercategoryid = 65 -- continuous i.v. perfusor
    AND itemid IN (
      7179, -- Dopamine (Inotropin)
      7178, -- Dobutamine (Dobutrex)
      6818, -- Adrenaline (Epinefrine)
      7229  -- Noradrenaline (Norepinefrine)
    )
    AND rate > 0.1
),
vaso_norm AS (
  SELECT admissionid
  , starttime
  , stoptime
  , item
  , itemid
  , duration
  , rate
  , rateunit
  , dose
  , doseunit
  , doseunitid
  , doserateperkg
  , doserateunitid
  , doserateunit
  , patientweight
  , CASE
    --recalculate the dose to µg/kg/min --> norepinephrine equivalents.
    WHEN doserateperkg = B'0' AND doseunitid = 11 AND doserateunitid = 4 --unit: µg/min -> µg/kg/min
      THEN CASE
        WHEN patientweight > 0
        THEN dose/patientweight
        ELSE dose/80 --mean weight
      END
    WHEN doserateperkg = B'0' AND doseunitid = 10 AND
    doserateunitid = 4 --unit: mg/min  -> µg/kg/min
      THEN CASE
        WHEN patientweight > 0
        THEN dose*1000/patientweight
        ELSE dose*1000/80 --mean weight
      END
    WHEN doserateperkg = B'0' AND doseunitid = 10 AND doserateunitid = 5 --unit: mg/uur  -> µg/kg/min
      THEN CASE
        WHEN patientweight > 0
        THEN dose*1000/patientweight/60
        ELSE dose*1000/80 --mean weight
      END
    WHEN doserateperkg = B'1' AND doseunitid = 11 AND doserateunitid = 4 --unit: µg/kg/min (no conversion needed)
      THEN dose
    WHEN doserateperkg = B'1' AND doseunitid = 11 AND doserateunitid = 5 --unit: µg/kg/uur -> µg/kg/min
      THEN dose/60
    END AS rate_norm
  FROM vaso_rates
  ORDER BY admissionid, starttime
),
vasopressors AS (
  SELECT admissionid
  , starttime
  , stoptime
  , rateunit
  , (CASE WHEN itemid = 7179 THEN rate_norm else null end) as dopamine_rate
  , (CASE WHEN itemid = 7178 THEN rate_norm else null end) as dobutamine_rate
  , (CASE WHEN itemid = 6818 THEN rate_norm else null end) as epinefrine_rate
  , (CASE WHEN itemid = 7229 THEN rate_norm else null end) as norepinefrine_rate
  FROM vaso_norm
  ORDER BY admissionid, starttime
),
vaso AS (
  SELECT admissionid
  , starttime
  , stoptime
  , rateunit
  , MAX(dopamine_rate) AS dopamine_rate
  , MAX(dobutamine_rate) AS dobutamine_rate
  , MAX(epinefrine_rate) AS epinefrine_rate
  , MAX(norepinefrine_rate) AS norepinefrine_rate
  FROM vasopressors
  GROUP BY admissionid, starttime, stoptime, rateunit
  ORDER BY admissionid, starttime
)
SELECT *
  , coalesce(dopamine_rate, 0) + coalesce(dobutamine_rate, 0) +
    coalesce(epinefrine_rate, 0) + coalesce(norepinefrine_rate, 0)
    AS vaso_total
FROM vaso
ORDER BY admissionid, starttime



