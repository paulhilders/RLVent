-- Description: This query creates a new getintravenous table, which contains
-- 	the average amount of intravenous fluids administered to patients in the
-- 	MIMIC-IV database.
-- Adapted from: https://github.com/arnepeine/ventai/blob/main/getIntravenous.sql
-- Execution time: Roughly 1 minute.

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS getintravenous; CREATE TABLE getintravenous AS

WITH intra_mv AS
(
	SELECT *
	FROM inputevents
	WHERE ordercategoryname IN ('03-IV Fluid Bolus','02-Fluids (Crystalloids)','04-Fluids (Colloids)','07-Blood Products')
	OR secondaryordercategoryname IN ('03-IV Fluid Bolus','02-Fluids (Crystalloids)','04-Fluids (Colloids)','07-Blood Products')
)

SELECT subject_id, hadm_id, stay_id, storetime as charttime
	 , avg(totalamount) as amount
FROM intra_mv

group by subject_id, hadm_id, stay_id, charttime
order by subject_id, hadm_id, stay_id, charttime