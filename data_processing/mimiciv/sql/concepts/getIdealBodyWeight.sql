-- Description: This query creates a new getidealbodyweight table, which
--      contains derived IBW for the patients in the MIMIC-IV database.
-- Adapted from: https://github.com/florisdenhengst/ventai/blob/main/getAdultIdealBodyWeight.sql
-- Execution time: A few seconds.
-- Number of Rows: 73141 (73 thousand)

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS getidealbodyweight; CREATE TABLE getidealbodyweight AS

WITH ht_stg AS
(
    SELECT ic.subject_id
    , ic.hadm_id
    , fdh.stay_id
    , pt.gender
    , AVG(height) AS height
    FROM first_day_height fdh
    LEFT JOIN icustays ic
    ON ic.stay_id = fdh.stay_id
    LEFT JOIN patients pt
    ON pt.subject_id = fdh.subject_id
    AND ic.subject_id = fdh.subject_id
    GROUP BY ic.subject_id, ic.hadm_id, fdh.stay_id, pt.gender
    ORDER BY ic.subject_id, ic.hadm_id, fdh.stay_id, pt.gender
),
NORM_WEIGHT AS (
	SELECT subject_id
    , hadm_id
    , stay_id
    , MODE() WITHIN GROUP (ORDER BY gender) AS gender
    , AVG(height) AS height FROM ht_stg
	GROUP BY subject_id, hadm_id, stay_id
)

-- Calculation of IBW according to the gender-specific ARDSnet formulas:
-- https://pubs.asahq.org/anesthesiology/article/127/1/203/18747/Calculating-Ideal-Body-Weight-Keep-It-Simple
SELECT subject_id
    , hadm_id
    , stay_id
    , gender
	, height
	, CASE WHEN gender = 'M'
		    THEN 50 + 0.91 * (height - 152.4)
		ELSE
		CASE WHEN gender = 'F'
		    THEN 45.5 + 0.91 * (height - 152.4)
		ELSE NULL
		END
	END
	AS adult_ibw
FROM NORM_WEIGHT
