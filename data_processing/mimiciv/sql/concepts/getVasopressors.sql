-- Description: This query creates a new getvasopressors table, which
--	 	contains the vasopressor rates and total vasopressor dose
--	 	for the patients in the MIMIC-IV database.
-- Source: https://github.com/arnepeine/ventai/blob/main/getVasopressors.sql
-- Execution time: Roughly 1 minute.
-- Number of Rows: 588680 (589 thousand)

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS getvasopressors; CREATE TABLE getvasopressors AS

WITH vaso_union AS
(SELECT stay_id, starttime,
	vaso_rate as rate_norepinephrine,
	null::double precision as rate_epinephrine,
	null::double precision as rate_phenylephrine,
	null::double precision as rate_dopamine,
	null::double precision as rate_vasopressin

FROM norepinephrine

UNION ALL

SELECT stay_id, starttime,
	null::double precision as rate_norepinephrine,
	vaso_rate as rate_epinephrine,
	null::double precision as rate_phenylephrine,
	null::double precision as rate_dopamine,
	null::double precision as rate_vasopressin

FROM epinephrine

UNION ALL

SELECT stay_id, starttime,
	null::double precision as rate_norepinephrine,
	null::double precision as rate_epinephrine,
	vaso_rate as rate_phenylephrine,
	null::double precision as rate_dopamine,
	null::double precision as rate_vasopressin

FROM phenylephrine

UNION ALL

SELECT stay_id, starttime,
	null::double precision as rate_norepinephrine,
	null::double precision as rate_epinephrine,
	null::double precision as rate_phenylephrine,
	vaso_rate as rate_dopamine,
	null::double precision as rate_vasopressin

FROM dopamine

UNION ALL

SELECT stay_id, starttime,
	null::double precision as rate_norepinephrine,
	null::double precision as rate_epinephrine,
	null::double precision as rate_phenylephrine,
	null::double precision as rate_dopamine,
	vaso_rate as rate_vasopressin

FROM vasopressin
),
vaso as
(
SELECT stay_id, starttime,
  --max command is used to merge different vasopressors taken at the same time into a single row.
	max(rate_norepinephrine) as rate_norepinephrine,
	max(rate_epinephrine) as rate_epinephrine,
	max(rate_phenylephrine) as rate_phenylephrine,
	max(rate_dopamine) as rate_dopamine,
	max(rate_vasopressin) as rate_vasopressin

FROM vaso_union

GROUP BY stay_id, starttime
 )
 SELECT *,
    coalesce(rate_norepinephrine,0) + coalesce(rate_epinephrine,0) +
	coalesce(rate_phenylephrine/2.2,0) + coalesce(rate_dopamine/100,0) +
	coalesce(rate_vasopressin*8.33,0) as vaso_total

FROM vaso

ORDER BY stay_id, starttime