-- Description: This query creates a new getmechvent table, which contains
--   the mechanical ventilation status for patients in the AmsterdamUMCdb database.
-- Number of rows: 15705787 (15.7 million)
-- Time required: Roughly 1 minute
-- Inspired by: https://github.com/AmsterdamUMC/AmsterdamUMCdb/blob/master/amsterdamumcdb/sql/lifesupport/mechanical_ventilation.sql

SET search_path TO public, amsterdamumcdb, amsterdamumcdb_derived;

DROP TABLE IF EXISTS getmechvent; CREATE TABLE getmechvent AS

WITH mv AS (
  SELECT li.admissionid
  , li.measuredat
  , ((li.measuredat - a.admittedat)/(1000*60)) as charttime
  , li.itemid
  , li.item
  , li.value
  , li.valueid
    FROM listitems li
    LEFT JOIN admissions a
    ON li.admissionid = a.admissionid
    WHERE (
                itemid = 9534 --Type beademing Evita 1
            AND valueid IN (
                            1, --IPPV
                            2, --IPPV_Assist
                            3, --CPPV
                            4, --CPPV_Assist
                            5, --SIMV
                            6, --SIMV_ASB
                            7, --ASB
                            8, --CPAP
                            9, --CPAP_ASB
                            10, --MMV
                            11, --MMV_ASB
                            12, --BIPAP
                            13 --Pressure Controled
            )
        )
       OR (
                itemid = 6685 --Type Beademing Evita 4
            AND valueid IN (
                            1, --CPPV
                            3, --ASB
                            5, --CPPV/ASSIST
                            6, --SIMV/ASB
                            8, --IPPV
                            9, --IPPV/ASSIST
                            10, --CPAP
                            11, --CPAP/ASB
                            12, --MMV
                            13, --MMV/ASB
                            14, --BIPAP
                            20, --BIPAP-SIMV/ASB
                            22 --BIPAP/ASB
            )
        )
       OR (
                itemid = 8189 --Toedieningsweg O2
            AND valueid = 16 --CPAP
        )
       OR (
                itemid IN (
                           12290, --Ventilatie Mode (Set) - Servo-I and Servo-U ventilators
                           12347 --Ventilatie Mode (Set) (2) Servo-I and Servo-U ventilators
                )
            AND valueid IN (
            --IGNORE: 1, --Stand By
                            2, --PC
                            3, --VC
                            4, --PRVC
                            5, --VS
                            6, --SIMV(VC)+PS
                            7, --SIMV(PC)+PS
                            8, --PS/CPAP
                            9, --Bi Vente
                            10, --PC (No trig)
                            11, --VC (No trig)
                            12, --PRVC (No trig)
                            13, --PS/CPAP (trig)
                            14, --VC (trig)
                            15, --PRVC (trig)
                            16, --PC in NIV
                            17, --PS/CPAP in NIV
                            18 --NAVA
            )
        )
       OR itemid = 12376 --Mode (Bipap Vision)
        AND valueid IN (
                        1, --CPAP
                        2 --BIPAP
            )
    GROUP BY li.admissionid, li.measuredat, charttime, li.itemid, li.item, li.value, li.valueid
),
imv AS (
  SELECT *,
    MAX(CASE
      WHEN itemid = 9534 --Type beademing Evita 1
                 AND valueid IN (
                                 1, --IPPV
                                 2, --IPPV_Assist
                                 3, --CPPV
                                 4, --CPPV_Assist
                                 5, --SIMV
                                 6, --SIMV_ASB
                                 7, --ASB
                                 8, --CPAP
                                 9, --CPAP_ASB
                                 10, --MMV
                                 11, --MMV_ASB
                                 12, --BIPAP
                                 13 --Pressure Controled
                 ) THEN 1
      WHEN itemid = 6685 --Type Beademing Evita 4
                 AND valueid IN (
                                 1, --CPPV
                                 3, --ASB
                                 5, --CPPV/ASSIST
                                 6, --SIMV/ASB
                                 8, --IPPV
                                 9, --IPPV/ASSIST
                                 10, --CPAP
                                 11, --CPAP/ASB
                                 12, --MMV
                                 13, --MMV/ASB
                                 14, --BIPAP
                                 20, --BIPAP-SIMV/ASB
                                 22 --BIPAP/ASB
                 ) THEN 1
      WHEN itemid IN (
                                12290, --Ventilatie Mode (Set) - Servo-I and Servo-U ventilators
                                12347 --Ventilatie Mode (Set) (2) Servo-I and Servo-U ventilators
                     )
                 AND valueid IN (
                                 2, --PC
                                 3, --VC
                                 4, --PRVC
                                 5, --VS
                                 6, --SIMV(VC)+PS
                                 7, --SIMV(PC)+PS
                                 8, --PS/CPAP
                                 9, --Bi Vente
                                 10, --PC (No trig)
                                 11, --VC (No trig)
                                 12, --PRVC (No trig)
                                 13, --PS/CPAP (trig)
                                 14, --VC (trig)
                                 15, --PRVC (trig)
                                 18 --NAVA
                 ) THEN 1
      ELSE 0
      END
) AS mechvent
  FROM mv
  GROUP BY admissionid, measuredat, charttime, itemid, item, value, valueid
  ORDER BY admissionid, charttime
)

SELECT *
FROM imv
