-- Description: This query creates a new sampled_overalltable table,
--      which discretizes the ICU stays into a grid using a predefined 4-hour window.
--      The table contains the average or maximum values of the measurements in the window.
-- Source: https://github.com/arnepeine/ventai/blob/main/sampling_lab_withventparams.sql
-- Execution time: Roughly 1 hour.
-- Number of Rows: 52092393 (52 million) for 4-hour window, 58875742 (5.9 million) for 1-hour window.

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS sampled_overalltable; CREATE TABLE sampled_overalltable AS

WITH minmax as
(
	SELECT subject_id, hadm_id, stay_id , min(charttime) as mint, max(charttime) as maxt
	FROM merged_overalltable
	GROUP BY stay_id, subject_id, hadm_id
	ORDER BY stay_id, subject_id, hadm_id
),
grid as
(
	SELECT stay_id, subject_id, hadm_id, generate_series(mint, maxt, interval '4 hours') as start_time
	FROM minmax
	GROUP BY stay_id, subject_id, hadm_id, mint, maxt
    ORDER BY stay_id, subject_id, hadm_id
)

SELECT ot.stay_id, ot.subject_id, ot.hadm_id, start_time
	-- vital signs
	, round(avg(gcs)) as gcs , avg(HeartRate) as heartrate , avg(SysBP) as sysbp
	, avg(DiasBP) as diasbp , avg(MeanBP) as meanbp
	, avg(shockindex) as shockindex, avg(RespRate) as RespRate
	, avg(TempC) as TempC , avg(SpO2) as SpO2
	--lab values
	, avg(POTASSIUM) as POTASSIUM , avg(SODIUM) as SODIUM , avg(CHLORIDE) as CHLORIDE , avg(GLUCOSE) as GLUCOSE
	, avg(BUN) as BUN , avg(CREATININE) as CREATININE , avg(MAGNESIUM) as MAGNESIUM , avg(CALCIUM) as CALCIUM
	, avg(CARBONDIOXIDE) as CARBONDIOXIDE , avg(BILIRUBIN) as BILIRUBIN , avg(ALBUMIN) as ALBUMIN
	, avg(HEMOGLOBIN) as HEMOGLOBIN , avg(WBC) as WBC , avg(PLATELET) as PLATELET , avg(PTT) as PTT
	, avg(PT) as PT , avg(INR) as INR , avg(PH) as PH , avg(PaO2) as PaO2 , avg(PaCO2) as PaCO2
	, avg(BASE_EXCESS) as BASE_EXCESS , avg(BICARBONATE) as BICARBONATE , avg(LACTATE) as LACTATE
	, avg(PaO2FiO2ratio) as pao2fio2ratio, avg(BANDS) as BANDS -- this is only included in order to calculate SIRS score
	, avg(CRP) as CRP
	-- others
	, avg(ionizedcalcium) as ionizedcalcium
	-- MechVent
	, avg(MechVent) as MechVent
	--ventilation parameters
	, avg(FiO2) as FiO2
	--urine output
	, sum(urineoutput) as urineoutput
	-- vasopressors
	, max(rate_norepinephrine) as rate_norepinephrine , max(rate_epinephrine) as rate_epinephrine
	, max(rate_phenylephrine) as rate_phenylephrine , max(rate_vasopressin) as rate_vasopressin
	, max(rate_dopamine) as rate_dopamine , max(vaso_total) as vaso_total
	-- intravenous fluids
	, sum(iv_total) as iv_total
	-- cumulative fluid balance
	, avg(cum_fluid_balance) as cum_fluid_balance
	-- ventilation parameters
	, max(PEEP) as PEEP, max(tidal_volume) as tidal_volume, max(plateau_pressure) as plateau_pressure
	, avg(resprate_set) as RespRate_set, avg(resprate_spont) as RespRate_spont, avg(minutevolume) as minutevolume, avg(flowrate) as flowrate
	, avg(tidal_volume_set) as tidal_volume_set, avg(tidal_volume_spont) as tidal_volume_spont

	-- Others
	, avg(SysPAP) as SysPAP, avg(DiasPAP) as DiasPAP, avg(MeanPAP) as MeanPAP
	, avg(PIP) as PIP, avg(O2Flow) as O2Flow, avg(MAP) as MAP, avg(ResistanceInsp) as ResistanceInsp
	, avg(ResistanceExp) as ResistanceExp, avg(Compliance) as Compliance, avg(VitalCapacity) as VitalCapacity
	, avg(TranspulmonaryPressureExpHold) as TranspulmonaryPressureExpHold
	, avg(TranspulmonaryPressureInspHold) as TranspulmonaryPressureInspHold
	, avg(ExpRatio) as ExpRatio, avg(InspRatio) as InspRatio, avg(ExpTC) as ExpTC, avg(InspTC) as InspTC
	, avg(EtCO2) as EtCO2

FROM grid g
LEFT JOIN merged_overalltable ot
	ON ot.charttime >= g.start_time
	AND ot.charttime <  g.start_time + '4 hours'
	AND ot.stay_id = g.stay_id
	AND ot.subject_id = g.subject_id
	AND ot.hadm_id = g.hadm_id

GROUP BY ot.stay_id, ot.subject_id, ot.hadm_id, start_time
ORDER BY stay_id, subject_id, hadm_id, start_time