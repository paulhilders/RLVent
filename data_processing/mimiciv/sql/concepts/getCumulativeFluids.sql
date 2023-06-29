-- Description: This query creates a new getcumulativefluids table, which
-- 		merges the fluids information from the inputevents table with the
-- 		information from the outputevents table.
-- Adapted from: https://github.com/arnepeine/ventai/blob/main/getCumFluid.sql
-- Execution time: Roughly 1 minute.
-- Number of Rows: 7808306 (7.8 million)

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS getcumulativefluids; CREATE TABLE getcumulativefluids AS

SELECT subject_id, hadm_id, stay_id, charttime, in_amount, in_cum_amt, out_amount, out_cum_amt,
       sum(out_amount) OVER (PARTITION BY in_out.stay_id ORDER BY charttime)
	   -sum(in_amount) OVER (PARTITION BY in_out.stay_id ORDER BY charttime) as cum_fluid_balance
FROM (
	-- Input Events
	SELECT subject_id, hadm_id,merged.stay_id,charttime, in_amount,
		sum(in_amount) OVER (PARTITION BY merged.stay_id ORDER BY charttime) AS in_cum_amt,
		null::double precision AS out_amount, null::double precision AS out_cum_amt
	FROM (
		SELECT stay_id, starttime as charttime,
		--Some unit conversions that will end up in 'mL'.
		(CASE WHEN amountuom='ml' THEN sum(amount)
				WHEN amountuom='L'  THEN sum(amount)*0.001
				WHEN amountuom='uL' THEN sum(amount)*1000  END) as in_amount
		FROM inputevents ie
		WHERE amountuom in ('L','ml','uL')
		GROUP BY stay_id, charttime, amountuom
	) AS merged
	INNER JOIN icustays ic
	ON ic.stay_id=merged.stay_id

	UNION ALL

	--Output events.
	SELECT subject_id, hadm_id, merged.stay_id, charttime,
		null::double precision AS in_amount, null::double precision AS in_cum_amt, out_amount,
		sum(out_amount) OVER (PARTITION BY merged.stay_id ORDER BY charttime) AS out_cum_amt
	FROM (
		SELECT stay_id, charttime, sum(value) as out_amount
		FROM outputevents oe
		WHERE valueuom in ('mL','ml')
		GROUP BY stay_id, charttime) as merged
	INNER JOIN icustays ic
	ON ic.stay_id=merged.stay_id
) AS in_out
ORDER BY stay_id, charttime
