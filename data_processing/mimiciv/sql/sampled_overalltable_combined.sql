-- Description: This query creates a new sampled_overalltable_combined_hourly table,
--   	which inserts the records from the Demographics, SIRS, SOFA, weight,
--		and IBW tables into the samled overalltable.
-- Source: https://github.com/florisdenhengst/ventai/blob/main/sampled_data_with_scdem_withventparams.sql
-- Execution time: Roughly 1 minute.
-- Number of rows: 1555556 (1.5 million) for 4-hour window, 5956545 (6 million) for 1-hour window.

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS sampled_overalltable_combined_hourly; CREATE TABLE sampled_overalltable_combined_hourly AS

SELECT samp.stay_id, samp.subject_id, samp.hadm_id, ic.intime, ic.outtime, samp.start_time
    , dem.admission_age, dem.gender
	, (CASE WHEN fdw.weight_admit IS NOT NULL THEN fdw.weight_admit
			ELSE weig.avg
	   END) AS weight
	, ibw.adult_ibw
	, dem.icu_readmission, dem.elixhauser_vanwalraven, sf.sofa , sr.sirs
    , samp.gcs , samp.heartrate , samp.sysbp, samp.diasbp, samp.meanbp, samp.shockindex
	, (CASE WHEN samp.resprate IS NOT NULL THEN samp.resprate
			WHEN samp.RespRate_set IS NOT NULL THEN samp.RespRate_set
			ELSE NULL
		END) AS resprate
	, samp.RespRate_spont AS resprate_spont
	, samp.tempc, samp.spo2, samp.potassium
	, samp.sodium, samp.chloride, samp.glucose, samp.bun, samp.creatinine, samp.magnesium
	, samp.calcium, samp.ionizedcalcium, samp.carbondioxide, samp.bilirubin, samp.albumin, samp.hemoglobin
	, samp.wbc, samp.platelet, samp.ptt, samp.pt, samp.inr, samp.ph, samp.pao2, samp.paco2, samp.base_excess
	, samp.bicarbonate, samp.lactate
	, samp.CRP
	, samp.pao2fio2ratio, samp.mechvent, samp.fio2, samp.urineoutput
	, samp.vaso_total, samp.iv_total, samp.cum_fluid_balance, samp.peep
	, (CASE WHEN samp.tidal_volume IS NOT NULL THEN samp.tidal_volume
			WHEN samp.tidal_volume_set IS NOT NULL THEN samp.tidal_volume_set
			ELSE NULL
		END) AS tidal_volume
	, samp.tidal_volume_spont AS tidal_volume_spont
	, samp.plateau_pressure
	, samp.SysPAP, samp.DiasPAP, samp.MeanPAP, samp.PIP, samp.O2Flow, samp.MAP
	, samp.ResistanceInsp, samp.ResistanceExp, samp.Compliance, samp.VitalCapacity
	, samp.TranspulmonaryPressureExpHold, samp.TranspulmonaryPressureInspHold
	, samp.ExpRatio, samp.InspRatio, samp.ExpTC, samp.InspTC
	, samp.EtCO2

	, dem.hospmort, dem.mort90day, dem.dischtime
	, dem.deathtime, dem.admittime as hadmittime, dem.dischtime as hdischtime

FROM sampled_overalltable_hourly samp

LEFT JOIN getdemographics dem
ON samp.subject_id = dem.subject_id
AND samp.hadm_id = dem.hadm_id
AND samp.stay_id = dem.stay_id

LEFT JOIN sirs_overalltable_hourly sr
ON samp.subject_id = sr.subject_id
AND samp.stay_id = sr.stay_id
AND samp.start_time = sr.start_time

LEFT JOIN sofa_overalltable_hourly sf
ON samp.subject_id = sf.subject_id
AND samp.stay_id = sf.stay_id
AND samp.start_time = sf.start_time

LEFT JOIN getweight weig
ON samp.subject_id = weig.subject_id
AND samp.hadm_id = weig.hadm_id
AND samp.stay_id = weig.stay_id
AND samp.start_time = weig.charttime

LEFT JOIN getidealbodyweight ibw
ON samp.subject_id = ibw.subject_id
AND samp.hadm_id = ibw.hadm_id
AND samp.stay_id = ibw.stay_id

LEFT JOIN first_day_weight fdw
ON samp.stay_id = fdw.stay_id

INNER JOIN icustays ic
ON samp.stay_id=ic.stay_id

ORDER BY samp.stay_id, samp.subject_id, samp.start_time
