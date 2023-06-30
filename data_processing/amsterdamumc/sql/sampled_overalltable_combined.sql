-- Description: This query creates a new sampled_overalltable_combined_hourly table,
--   	which inserts the records from the Demographics, SIRS, and SOFA into
--      the sampled overalltable. In addition, it computes an approximated value
--		for the compliance of the lungs. As this value is only an estimate, we
--		decided to not include it in the final version of the paper.
-- Source: https://github.com/florisdenhengst/ventai/blob/main/sampled_data_with_scdem_withventparams.sql
-- Execution time: Roughly 10 seconds.
-- Number of rows: 1764311 (1.8 million) for 4-hour window, 4927360 (4.9 million) for 1-hour window.

SET search_path TO public, amsterdamumcdb, amsterdamumcdb_derived;

DROP TABLE IF EXISTS sampled_overalltable_combined_hourly; CREATE TABLE sampled_overalltable_combined_hourly AS

WITH combined AS (
	SELECT samp.admissionid, dem.admittedat as intime, dem.dischargedat as outtime, samp.start_time
		, dem.age as admission_age, dem.gender
		, dem.weight AS weight
		, (CASE
			WHEN dem.gender = 'Vrouw'
				THEN 45 + 0.91 * (height - 152.4)
			WHEN dem.gender = 'Man'
				THEN 50 + 0.91 * (height - 152.4)
			ELSE NULL
		END) as adult_ibw
		, dem.icu_readmission
		, sf.sofa , sr.sirs
		, samp.gcs , samp.heartrate , samp.sysbp, samp.diasbp, samp.meanbp, samp.shockindex
		, (CASE WHEN samp.resprate IS NOT NULL THEN samp.resprate
				WHEN samp.resprate_set IS NOT NULL THEN samp.resprate_set
				ELSE NULL
			END) AS resprate
		, samp.resprate_spont AS resprate_spont
		, samp.tempc, samp.spo2, samp.potassium
		, samp.sodium, samp.chloride, samp.glucose, samp.bun, samp.creatinine, samp.magnesium
		, samp.calcium, samp.ionizedcalcium, samp.bilirubin, samp.albumin, samp.hemoglobin
		, samp.wbc, samp.platelet, samp.ptt, samp.inr, samp.ph, samp.pao2, samp.paco2, samp.base_excess
		, samp.bicarbonate, samp.lactate
		, samp.CRP
		, samp.pao2fio2ratio, samp.mechvent, samp.fio2, samp.urineoutput
		, samp.vaso_total
		, samp.dopamine_rate, samp.dobutamine_rate, samp.epinefrine_rate, samp.norepinefrine_rate
		, samp.iv_total, samp.cum_fluid_balance, samp.peep
		, (CASE WHEN samp.tidal_volume IS NOT NULL THEN samp.tidal_volume
				WHEN samp.tidal_volume_set IS NOT NULL THEN samp.tidal_volume_set
				ELSE NULL
			END) AS tidal_volume
		, samp.tidal_volume_spont AS tidal_volume_spont
		, samp.plateau_pressure
		, samp.PIP, samp.MAP
		, samp.EtCO2

		, dem.hospmort, dem.mort90day, dem.dischargedat
		, dem.dateofdeath, dem.admittedat as hadmittime, dem.dischargedat as hdischtime

	FROM sampled_overalltable_hourly samp

	LEFT JOIN getdemographics dem
	ON samp.admissionid = dem.admissionid

	LEFT JOIN sirs_overalltable_hourly sr
	ON samp.admissionid = sr.admissionid
	AND samp.start_time = sr.start_time

	LEFT JOIN sofa_overalltable_hourly sf
	ON samp.admissionid = sf.admissionid
	AND samp.start_time = sf.start_time

	ORDER BY samp.admissionid, samp.start_time
)

SELECT *
	, (CASE WHEN (pip - peep) > 0 AND (tidal_volume / (pip - peep) <= 200)
			THEN tidal_volume / (pip - peep)
		ELSE NULL
	   END) AS compliance
FROM combined
ORDER BY admissionid, start_time
