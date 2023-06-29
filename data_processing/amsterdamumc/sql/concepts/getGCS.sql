-- Description: This query creates a new getgcs table, which contains the
--   Glasgow Coma Scale (GCS) scores for patients in the AmsterdamUMCdb database.
-- Number of rows: 266269 (266 thousand)
-- Time required: Roughly 10 seconds

SET search_path TO public, amsterdamumcdb, amsterdamumcdb_derived;

DROP TABLE IF EXISTS getgcs; CREATE TABLE getgcs AS

WITH gcs_components AS (
  SELECT
    eyes.admissionid,
    CASE eyes.itemid
      WHEN 6732 THEN 5 - eyes.valueid     --Actief openen van de ogen
      WHEN 13077 THEN eyes.valueid        --A_Eye
      WHEN 14470 THEN eyes.valueid - 4    --RA_Eye
      WHEN 16628 THEN eyes.valueid - 4    --MCA_Eye
      WHEN 19635 THEN eyes.valueid - 4    --E_EMV_NICE_24uur
      WHEN 19638 THEN eyes.valueid - 8    --E_EMV_NICE_Opname
    END AS eyes_score,
    CASE motor.itemid
      WHEN 6734 THEN 7 - motor.valueid    --Beste motore reactie van de armen
      WHEN 13072 THEN motor.valueid       --A_Motoriek
      WHEN 14476 THEN motor.valueid - 6   --RA_Motoriek
      WHEN 16634 THEN motor.valueid - 6   --MCA_Motoriek
      WHEN 19636 THEN motor.valueid - 6   --M_EMV_NICE_24uur
      WHEN 19639 THEN motor.valueid - 12  --M_EMV_NICE_Opname
    END AS motor_score,
    CASE verbal.itemid
      WHEN 6735 THEN 6 - verbal.valueid   --Beste verbale reactie
      WHEN 13066 THEN verbal.valueid      --A_Verbal
      WHEN 14482 THEN verbal.valueid - 5  --RA_Verbal
      WHEN 16640 THEN verbal.valueid - 5  --MCA_Verbal
      WHEN 19637 THEN verbal.valueid - 9 --V_EMV_NICE_24uur
      WHEN 19640 THEN verbal.valueid - 15 --V_EMV_NICE_Opname
    END AS verbal_score,
    eyes.measuredat,
    (eyes.measuredat - a.admittedat) / (1000*60) AS charttime
  FROM listitems eyes
  LEFT JOIN admissions a ON
    eyes.admissionid = a.admissionid
  LEFT JOIN listitems motor ON
    eyes.admissionid = motor.admissionid AND
    eyes.measuredat = motor.measuredat AND
    motor.itemid IN (
      6734, --Beste motore reactie van de armen
      13072, --A_Motoriek
      14476, --RA_Motoriek
      16634, --MCA_Motoriek
      19636, --M_EMV_NICE_24uur
      19639 --M_EMV_NICE_Opname
    )
  LEFT JOIN listitems verbal ON
    eyes.admissionid = verbal.admissionid AND
    eyes.measuredat = verbal.measuredat AND
    verbal.itemid IN (
      6735, --Beste verbale reactie
      13066, --A_Verbal
      14482, --RA_Verbal
      16640, --MCA_Verbal
      19637, --V_EMV_NICE_24uur
      19640 --V_EMV_NICE_Opname
    )
  WHERE eyes.itemid IN (
    6732, --Actief openen van de ogen
    13077, --A_Eye
    14470, --RA_Eye
    16628, --MCA_Eye
    19635, --E_EMV_NICE_24uur
    19638 --E_EMV_NICE_Opname
  )
), gcs AS (
  SELECT *
    , eyes_score + motor_score + (
      CASE
        WHEN verbal_score < 1 THEN 1
        ELSE verbal_score
      END
    ) AS gcs_score
  FROM gcs_components
)
SELECT
    gcs.admissionid,
    gcs.measuredat,
    gcs.charttime,
    gcs.eyes_score,
    gcs.motor_score,
    gcs.verbal_score,
    gcs_score
FROM gcs
GROUP BY gcs.admissionid, gcs.measuredat, gcs.charttime, gcs.eyes_score, gcs.motor_score, gcs.verbal_score, gcs_score
ORDER BY gcs.admissionid, gcs.measuredat
