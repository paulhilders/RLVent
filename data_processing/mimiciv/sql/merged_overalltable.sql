-- Description: This script creates a table with all the variables and measurements
-- 		we want to use for our analysis.
-- Adapted from: https://github.com/arnepeine/ventai/blob/main/overalltable_Lab_withventparams.sql
-- Execution time: Roughly 1 minute.
-- Number of Rows: 20312713 (20 million)

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS merged_overalltable; CREATE TABLE merged_overalltable AS


SELECT merged.subject_id, hadm_id, stay_id, charttime
	--vital signs
	, avg(gcs) as gcs, avg(HeartRate) as HeartRate , avg(SysBP) as SysBP
	, avg(DiasBP) as DiasBP , avg(MeanBP) as MeanBP , avg(SysBP)/avg(HeartRate) as shockindex
	, avg(RespRate) as RespRate, avg(TempC) as TempC , avg(SpO2) as SpO2
	--lab values
	, avg(POTASSIUM) as POTASSIUM , avg(SODIUM) as SODIUM , avg(CHLORIDE) as CHLORIDE , avg(GLUCOSE) as GLUCOSE
	, avg(BUN) as BUN , avg(CREATININE) as CREATININE , avg(MAGNESIUM) as MAGNESIUM , avg(CALCIUM) as CALCIUM
	, avg(CARBONDIOXIDE) as CARBONDIOXIDE, avg(BILIRUBIN) as BILIRUBIN , avg(ALBUMIN) as ALBUMIN
	, avg(HEMOGLOBIN) as HEMOGLOBIN , avg(WBC) as WBC , avg(PLATELET) as PLATELET , avg(PTT) as PTT
	, avg(PT) as PT , avg(INR) as INR , avg(PH) as PH , avg(PaO2) as PaO2 , avg(PaCO2) as PaCO2
	, avg(BASE_EXCESS) as BASE_EXCESS , avg(BICARBONATE) as BICARBONATE , avg(LACTATE) as LACTATE
	, avg(CRP) as CRP
	-- others
	, avg(ionizedcalcium) ionizedcalcium
	-- multiply by 100 because FiO2 is in a % but should be a fraction. This idea is retrieved from https://github.com/MIT-LCP/mimic-code/blob/master/concepts/firstday/blood-gas-first-day-arterial.sql
	, avg(PaO2)/avg(Fio2)*100 as PaO2FiO2ratio
	, avg(BANDS) as BANDS
	-- MechVent
	, avg(MechVent) as MechVent
	--ventilation parameters
	, avg(FiO2) as FiO2
	--urine output
	, avg(urineoutput) as urineoutput
	-- vasopressors
	, avg(rate_norepinephrine) as rate_norepinephrine , avg(rate_epinephrine) as rate_epinephrine
	, avg(rate_phenylephrine) as rate_phenylephrine , avg(rate_vasopressin) as rate_vasopressin
	, avg(rate_dopamine) as rate_dopamine , avg(vaso_total) as vaso_total
	-- intravenous fluids
	, avg(iv_total) as iv_total
	-- cumulated fluid balance
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

FROM
(
	SELECT vit.subject_id, ic.hadm_id, vit.stay_id, vit.charttime
	-- vital signs
	, gcs, heartrate, sysbp, diasbp, meanbp, resprate, tempc, spo2
	-- lab values
	, null::double precision as POTASSIUM , null::double precision as SODIUM , null::double precision as CHLORIDE , null::double precision as GLUCOSE
	, null::double precision as BUN , null::double precision as CREATININE , null::double precision as MAGNESIUM
	, null::double precision as CALCIUM , null::double precision as CARBONDIOXIDE , null::double precision as BILIRUBIN , null::double precision as ALBUMIN
	, null::double precision as HEMOGLOBIN , null::double precision as WBC , null::double precision as PLATELET , null::double precision as PTT
	, null::double precision as PT , null::double precision as INR , null::double precision as PH , null::double precision as PaO2 , null::double precision as PaCO2
	, null::double precision as BASE_EXCESS , null::double precision as BICARBONATE , null::double precision as LACTATE , null::double precision as BANDS
	, null::double precision as CRP
	-- others
	, null::double precision as IONIZEDCALCIUM
	-- mechvent
	, null::integer as MechVent
	-- ventilation parameters
	, null::double precision as FiO2
	-- urine output
	, null::double precision as urineoutput
	-- vasopressors
	, null::double precision as rate_norepinephrine , null::double precision as rate_epinephrine
	, null::double precision as rate_phenylephrine , null::double precision as rate_vasopressin
	, null::double precision as rate_dopamine , null::double precision as vaso_total
	-- intravenous fluids
	, null::double precision as iv_total
	-- cumulative fluid balance
	, null::double precision as cum_fluid_balance
	-- ventilation parameters
	, null::double precision as PEEP, null::double precision as tidal_volume, null::double precision as plateau_pressure
	, null::double precision as resprate_set, null::double precision as resprate_spont, null::double precision as minutevolume, null::double precision as flowrate
	, null::double precision as tidal_volume_set, null::double precision as tidal_volume_spont

	-- Others
	, null::double precision as SysPAP, null::double precision as DiasPAP, null::double precision as MeanPAP
	, null::double precision as PIP, null::double precision as O2Flow, null::double precision as MAP, null::double precision as ResistanceInsp
	, null::double precision as ResistanceExp, null::double precision as Compliance, null::double precision as VitalCapacity
	, null::double precision as TranspulmonaryPressureExpHold
	, null::double precision as TranspulmonaryPressureInspHold
	, null::double precision as ExpRatio, null::double precision as InspRatio, null::double precision as ExpTC, null::double precision as InspTC
	, null::double precision as EtCO2

	FROM merged_vitalsigns vit
	LEFT JOIN icustays ic
	ON ic.subject_id = vit.subject_id
	AND ic.stay_id = vit.stay_id

UNION ALL

	SELECT lab.subject_id, lab.hadm_id, lab.stay_id, lab.charttime
	-- vital signs
	, null::double precision as gcs, null::double precision as heartrate, null::double precision as sysbp, null::double precision as diasbp
	, null::double precision as meanbp,  null::double precision as resprate, null::double precision as tempc, null::double precision as spo2
	-- lab values
	, POTASSIUM , SODIUM , CHLORIDE , GLUCOSE , BUN , CREATININE , MAGNESIUM , CALCIUM , CARBONDIOXIDE
	, BILIRUBIN , ALBUMIN , HEMOGLOBIN , WBC , PLATELET , PTT , PT , INR , PH , PaO2 , PaCO2
	, BASE_EXCESS , BICARBONATE , LACTATE , BANDS
	, CRP
	-- others
	, null::double precision as IONIZEDCALCIUM
	-- mechvent
	, null::integer as MechVent
	-- ventilation parameters
	, null::double precision as FiO2
	-- urine output
	, null::double precision as urineoutput
	-- vasopressors
	, null::double precision as rate_norepinephrine , null::double precision as rate_epinephrine
	, null::double precision as rate_phenylephrine , null::double precision as rate_vasopressin
	, null::double precision as rate_dopamine , null::double precision as vaso_total
	-- intravenous fluids
	, null::double precision as iv_total
	-- cumulative fluid balance
	, null::double precision as cum_fluid_balance
	-- ventilation parameters
	, null::double precision as PEEP, null::double precision as tidal_volume, null::double precision as plateau_pressure
	, null::double precision as resprate_set, null::double precision as resprate_spont, null::double precision as minutevolume, null::double precision as flowrate
	, null::double precision as tidal_volume_set, null::double precision as tidal_volume_spont

	-- Others
	, null::double precision as SysPAP, null::double precision as DiasPAP, null::double precision as MeanPAP
	, null::double precision as PIP, null::double precision as O2Flow, null::double precision as MAP, null::double precision as ResistanceInsp
	, null::double precision as ResistanceExp, null::double precision as Compliance, null::double precision as VitalCapacity
	, null::double precision as TranspulmonaryPressureExpHold
	, null::double precision as TranspulmonaryPressureInspHold
	, null::double precision as ExpRatio, null::double precision as InspRatio, null::double precision as ExpTC, null::double precision as InspTC
	, null::double precision as EtCO2

	FROM getlabvalues lab

UNION ALL

	SELECT others.subject_id, others.hadm_id, others.stay_id, others.charttime
	-- vital signs
	, null::double precision as gcs, null::double precision as heartrate, null::double precision as sysbp, null::double precision as diasbp
	, null::double precision as meanbp,  null::double precision as resprate, null::double precision as tempc, null::double precision as spo2
	-- lab values
	, null::double precision as POTASSIUM , null::double precision as SODIUM , null::double precision as CHLORIDE , null::double precision as GLUCOSE
	, null::double precision as BUN , null::double precision as CREATININE , null::double precision as MAGNESIUM
	, null::double precision as CALCIUM , null::double precision as CARBONDIOXIDE , null::double precision as BILIRUBIN , null::double precision as ALBUMIN
	, null::double precision as HEMOGLOBIN , null::double precision as WBC , null::double precision as PLATELET , null::double precision as PTT
	, null::double precision as PT , null::double precision as INR , null::double precision as PH , null::double precision as PaO2 , null::double precision as PaCO2
	, null::double precision as BASE_EXCESS , null::double precision as BICARBONATE , null::double precision as LACTATE , null::double precision as BANDS
	, null::double precision as CRP
	-- others
	, ionizedcalcium as IONIZEDCALCIUM
	-- mechvent
	, null::integer as MechVent
	-- ventilation parameters
	, null::double precision as FiO2
	-- urine output
	, null::double precision as urineoutput
	-- vasopressors
	, null::double precision as rate_norepinephrine , null::double precision as rate_epinephrine
	, null::double precision as rate_phenylephrine , null::double precision as rate_vasopressin
	, null::double precision as rate_dopamine , null::double precision as vaso_total
	-- intravenous fluids
	, null::double precision as iv_total
	-- cumulative fluid balance
	, null::double precision as cum_fluid_balance
	-- ventilation parameters
	, null::double precision as PEEP, null::double precision as tidal_volume, null::double precision as plateau_pressure
	, null::double precision as resprate_set, null::double precision as resprate_spont, null::double precision as minutevolume, null::double precision as flowrate
	, null::double precision as tidal_volume_set, null::double precision as tidal_volume_spont

	-- Others
	, SysPAP as SysPAP, DiasPAP as DiasPAP, MeanPAP as MeanPAP
	, PIP as PIP, O2Flow as O2Flow, MAP as MAP, ResistanceInsp as ResistanceInsp
	, ResistanceExp as ResistanceExp, Compliance as Compliance, VitalCapacity as VitalCapacity
	, TranspulmonaryPressureExpHold as TranspulmonaryPressureExpHold
	, TranspulmonaryPressureInspHold as TranspulmonaryPressureInspHold
	, ExpRatio as ExpRatio, InspRatio as InspRatio, ExpTC as ExpTC, InspTC as InspTC
	, EtCO2 as EtCO2

	FROM getothers others

UNION ALL

	SELECT mechvent.subject_id, mechvent.hadm_id, mechvent.stay_id, mechvent.charttime
	-- vital signs
	, null::double precision as gcs, null::double precision as heartrate, null::double precision as sysbp, null::double precision as diasbp
	, null::double precision as meanbp,  null::double precision as resprate, null::double precision as tempc, null::double precision as spo2
	-- lab values
	, null::double precision as POTASSIUM , null::double precision as SODIUM , null::double precision as CHLORIDE , null::double precision as GLUCOSE
	, null::double precision as BUN , null::double precision as CREATININE , null::double precision as MAGNESIUM
	, null::double precision as CALCIUM , null::double precision as CARBONDIOXIDE , null::double precision as BILIRUBIN , null::double precision as ALBUMIN
	, null::double precision as HEMOGLOBIN , null::double precision as WBC , null::double precision as PLATELET , null::double precision as PTT
	, null::double precision as PT , null::double precision as INR , null::double precision as PH , null::double precision as PaO2 , null::double precision as PaCO2
	, null::double precision as BASE_EXCESS , null::double precision as BICARBONATE , null::double precision as LACTATE , null::double precision as BANDS
	, null::double precision as CRP
	-- others
	, null::double precision as IONIZEDCALCIUM
	-- mechvent
	, mechvent as MechVent
	-- ventilation parameters
	, null::double precision as FiO2
	-- urine output
	, null::double precision as urineoutput
	-- vasopressors
	, null::double precision as rate_norepinephrine , null::double precision as rate_epinephrine
	, null::double precision as rate_phenylephrine , null::double precision as rate_vasopressin
	, null::double precision as rate_dopamine , null::double precision as vaso_total
	-- intravenous fluids
	, null::double precision as iv_total
	-- cumulative fluid balance
	, null::double precision as cum_fluid_balance
	-- ventilation parameters
	, null::double precision as PEEP, null::double precision as tidal_volume, null::double precision as plateau_pressure
	, null::double precision as resprate_set, null::double precision as resprate_spont, null::double precision as minutevolume, null::double precision as flowrate
	, null::double precision as tidal_volume_set, null::double precision as tidal_volume_spont

	-- Others
	, null::double precision as SysPAP, null::double precision as DiasPAP, null::double precision as MeanPAP
	, null::double precision as PIP, null::double precision as O2Flow, null::double precision as MAP, null::double precision as ResistanceInsp
	, null::double precision as ResistanceExp, null::double precision as Compliance, null::double precision as VitalCapacity
	, null::double precision as TranspulmonaryPressureExpHold
	, null::double precision as TranspulmonaryPressureInspHold
	, null::double precision as ExpRatio, null::double precision as InspRatio, null::double precision as ExpTC, null::double precision as InspTC
	, null::double precision as EtCO2

	FROM getmechvent mechvent

UNION ALL

	SELECT mv.subject_id, ic.hadm_id, mv.stay_id, mv.charttime
	-- vital signs
	, null::double precision as gcs, null::double precision as heartrate, null::double precision as sysbp, null::double precision as diasbp
	, null::double precision as meanbp,  null::double precision as resprate, null::double precision as tempc, null::double precision as spo2
	-- lab values
	, null::double precision as POTASSIUM , null::double precision as SODIUM , null::double precision as CHLORIDE , null::double precision as GLUCOSE
	, null::double precision as BUN , null::double precision as CREATININE , null::double precision as MAGNESIUM
	, null::double precision as CALCIUM , null::double precision as CARBONDIOXIDE , null::double precision as BILIRUBIN , null::double precision as ALBUMIN
	, null::double precision as HEMOGLOBIN , null::double precision as WBC , null::double precision as PLATELET , null::double precision as PTT
	, null::double precision as PT , null::double precision as INR , null::double precision as PH , null::double precision as PaO2 , null::double precision as PaCO2
	, null::double precision as BASE_EXCESS , null::double precision as BICARBONATE , null::double precision as LACTATE , null::double precision as BANDS
	, null::double precision as CRP
	-- others
	, null::double precision as IONIZEDCALCIUM
	-- mechvent
	, null::integer as MechVent
	-- ventilation parameters
	, fio2 as FiO2
	-- urine output
	, null::double precision as urineoutput
	-- vasopressors
	, null::double precision as rate_norepinephrine , null::double precision as rate_epinephrine
	, null::double precision as rate_phenylephrine , null::double precision as rate_vasopressin
	, null::double precision as rate_dopamine , null::double precision as vaso_total
	-- intravenous fluids
	, null::double precision as iv_total
	-- cumulative fluid balance
	, null::double precision as cum_fluid_balance
	-- ventilation parameters
	, peep as PEEP, tidal_volume_observed as tidal_volume, plateau_pressure
	, respiratory_rate_set as resprate_set, respiratory_rate_spontaneous as resprate_spont, minute_volume as minutevolume, flow_rate as flowrate
	, tidal_volume_set, tidal_volume_spontaneous as tidal_volume_spont

	-- Others
	, null::double precision as SysPAP, null::double precision as DiasPAP, null::double precision as MeanPAP
	, null::double precision as PIP, null::double precision as O2Flow, null::double precision as MAP, null::double precision as ResistanceInsp
	, null::double precision as ResistanceExp, null::double precision as Compliance, null::double precision as VitalCapacity
	, null::double precision as TranspulmonaryPressureExpHold
	, null::double precision as TranspulmonaryPressureInspHold
	, null::double precision as ExpRatio, null::double precision as InspRatio, null::double precision as ExpTC, null::double precision as InspTC
	, null::double precision as EtCO2

	FROM ventilator_setting mv
	LEFT JOIN icustays ic
	ON ic.subject_id = mv.subject_id
	AND ic.stay_id = mv.stay_id

UNION ALL

	SELECT ic.subject_id, ic.hadm_id, uo.stay_id, uo.charttime
	-- vital signs
	, null::double precision as gcs, null::double precision as heartrate, null::double precision as sysbp, null::double precision as diasbp
	, null::double precision as meanbp,  null::double precision as resprate, null::double precision as tempc, null::double precision as spo2
	-- lab values
	, null::double precision as POTASSIUM , null::double precision as SODIUM , null::double precision as CHLORIDE , null::double precision as GLUCOSE
	, null::double precision as BUN , null::double precision as CREATININE , null::double precision as MAGNESIUM
	, null::double precision as CALCIUM , null::double precision as CARBONDIOXIDE , null::double precision as BILIRUBIN , null::double precision as ALBUMIN
	, null::double precision as HEMOGLOBIN , null::double precision as WBC , null::double precision as PLATELET , null::double precision as PTT
	, null::double precision as PT , null::double precision as INR , null::double precision as PH , null::double precision as PaO2 , null::double precision as PaCO2
	, null::double precision as BASE_EXCESS , null::double precision as BICARBONATE , null::double precision as LACTATE , null::double precision as BANDS
	, null::double precision as CRP
	-- others
	, null::double precision as IONIZEDCALCIUM
	-- mechvent
	, null::integer as MechVent
	-- ventilation parameters
	, null::double precision as FiO2
	-- urine output
	, urineoutput
	-- vasopressors
	, null::double precision as rate_norepinephrine , null::double precision as rate_epinephrine
	, null::double precision as rate_phenylephrine , null::double precision as rate_vasopressin
	, null::double precision as rate_dopamine , null::double precision as vaso_total
	-- intravenous fluids
	, null::double precision as iv_total
	-- cumulative fluid balance
	, null::double precision as cum_fluid_balance
	-- ventilation parameters
	, null::double precision as PEEP, null::double precision as tidal_volume, null::double precision as plateau_pressure
	, null::double precision as resprate_set, null::double precision as resprate_spont, null::double precision as minutevolume, null::double precision as flowrate
	, null::double precision as tidal_volume_set, null::double precision as tidal_volume_spont

	-- Others
	, null::double precision as SysPAP, null::double precision as DiasPAP, null::double precision as MeanPAP
	, null::double precision as PIP, null::double precision as O2Flow, null::double precision as MAP, null::double precision as ResistanceInsp
	, null::double precision as ResistanceExp, null::double precision as Compliance, null::double precision as VitalCapacity
	, null::double precision as TranspulmonaryPressureExpHold
	, null::double precision as TranspulmonaryPressureInspHold
	, null::double precision as ExpRatio, null::double precision as InspRatio, null::double precision as ExpTC, null::double precision as InspTC
	, null::double precision as EtCO2

	FROM urine_output uo
	LEFT JOIN icustays ic
	ON ic.stay_id = uo.stay_id

UNION ALL

	SELECT ic.subject_id, ic.hadm_id, vo.stay_id, vo.starttime as charttime
	-- vital signs
	, null::double precision as gcs, null::double precision as heartrate, null::double precision as sysbp, null::double precision as diasbp
	, null::double precision as meanbp,  null::double precision as resprate, null::double precision as tempc, null::double precision as spo2
	-- lab values
	, null::double precision as POTASSIUM , null::double precision as SODIUM , null::double precision as CHLORIDE , null::double precision as GLUCOSE
	, null::double precision as BUN , null::double precision as CREATININE , null::double precision as MAGNESIUM
	, null::double precision as CALCIUM , null::double precision as CARBONDIOXIDE , null::double precision as BILIRUBIN , null::double precision as ALBUMIN
	, null::double precision as HEMOGLOBIN , null::double precision as WBC , null::double precision as PLATELET , null::double precision as PTT
	, null::double precision as PT , null::double precision as INR , null::double precision as PH , null::double precision as PaO2 , null::double precision as PaCO2
	, null::double precision as BASE_EXCESS , null::double precision as BICARBONATE , null::double precision as LACTATE , null::double precision as BANDS
	, null::double precision as CRP
	-- others
	, null::double precision as IONIZEDCALCIUM
	-- mechvent
	, null::integer as MechVent
	-- ventilation parameters
	, null::double precision as FiO2
	-- urine output
	, null::double precision as urineoutput
	-- vasopressors
	, rate_norepinephrine , rate_epinephrine
	, rate_phenylephrine , rate_vasopressin
	, rate_dopamine , vaso_total
	-- intravenous fluids
	, null::double precision as iv_total
	-- cumulative fluid balance
	, null::double precision as cum_fluid_balance
	-- ventilation parameters
	, null::double precision as PEEP, null::double precision as tidal_volume, null::double precision as plateau_pressure
	, null::double precision as resprate_set, null::double precision as resprate_spont, null::double precision as minutevolume, null::double precision as flowrate
	, null::double precision as tidal_volume_set, null::double precision as tidal_volume_spont

	-- Others
	, null::double precision as SysPAP, null::double precision as DiasPAP, null::double precision as MeanPAP
	, null::double precision as PIP, null::double precision as O2Flow, null::double precision as MAP, null::double precision as ResistanceInsp
	, null::double precision as ResistanceExp, null::double precision as Compliance, null::double precision as VitalCapacity
	, null::double precision as TranspulmonaryPressureExpHold
	, null::double precision as TranspulmonaryPressureInspHold
	, null::double precision as ExpRatio, null::double precision as InspRatio, null::double precision as ExpTC, null::double precision as InspTC
	, null::double precision as EtCO2

	FROM getvasopressors vo
	LEFT JOIN icustays ic
	ON ic.stay_id = vo.stay_id

UNION ALL

	SELECT subject_id, hadm_id, stay_id, charttime
	-- vital signs
	, null::double precision as gcs, null::double precision as heartrate, null::double precision as sysbp, null::double precision as diasbp
	, null::double precision as meanbp,  null::double precision as resprate, null::double precision as tempc, null::double precision as spo2
	-- lab values
	, null::double precision as POTASSIUM , null::double precision as SODIUM , null::double precision as CHLORIDE , null::double precision as GLUCOSE
	, null::double precision as BUN , null::double precision as CREATININE , null::double precision as MAGNESIUM
	, null::double precision as CALCIUM , null::double precision as CARBONDIOXIDE , null::double precision as BILIRUBIN , null::double precision as ALBUMIN
	, null::double precision as HEMOGLOBIN , null::double precision as WBC , null::double precision as PLATELET , null::double precision as PTT
	, null::double precision as PT , null::double precision as INR , null::double precision as PH , null::double precision as PaO2 , null::double precision as PaCO2
	, null::double precision as BASE_EXCESS , null::double precision as BICARBONATE , null::double precision as LACTATE , null::double precision as BANDS
	, null::double precision as CRP
	-- others
	, null::double precision as IONIZEDCALCIUM
	-- mechvent
	, null::integer as MechVent
	-- ventilation parameters
	, null::double precision as FiO2
	-- urine output
	, null::double precision as urineoutput
	-- vasopressors
	, null::double precision as rate_norepinephrine , null::double precision as rate_epinephrine
	, null::double precision as rate_phenylephrine , null::double precision as rate_vasopressin
	, null::double precision as rate_dopamine , null::double precision as vaso_total
	-- intravenous fluids
	, amount as iv_total
	-- cumulative fluid balance
	, null::double precision as cum_fluid_balance
	-- ventilation parameters
	, null::double precision as PEEP, null::double precision as tidal_volume, null::double precision as plateau_pressure
	, null::double precision as resprate_set, null::double precision as resprate_spont, null::double precision as minutevolume, null::double precision as flowrate
	, null::double precision as tidal_volume_set, null::double precision as tidal_volume_spont

	-- Others
	, null::double precision as SysPAP, null::double precision as DiasPAP, null::double precision as MeanPAP
	, null::double precision as PIP, null::double precision as O2Flow, null::double precision as MAP, null::double precision as ResistanceInsp
	, null::double precision as ResistanceExp, null::double precision as Compliance, null::double precision as VitalCapacity
	, null::double precision as TranspulmonaryPressureExpHold
	, null::double precision as TranspulmonaryPressureInspHold
	, null::double precision as ExpRatio, null::double precision as InspRatio, null::double precision as ExpTC, null::double precision as InspTC
	, null::double precision as EtCO2

	FROM getintravenous

UNION ALL

	SELECT subject_id, hadm_id, stay_id, charttime
	-- vital signs
	, null::double precision as gcs, null::double precision as heartrate, null::double precision as sysbp, null::double precision as diasbp
	, null::double precision as meanbp,  null::double precision as resprate, null::double precision as tempc, null::double precision as spo2
	-- lab values
	, null::double precision as POTASSIUM , null::double precision as SODIUM , null::double precision as CHLORIDE , null::double precision as GLUCOSE
	, null::double precision as BUN , null::double precision as CREATININE , null::double precision as MAGNESIUM
	, null::double precision as CALCIUM , null::double precision as CARBONDIOXIDE , null::double precision as BILIRUBIN , null::double precision as ALBUMIN
	, null::double precision as HEMOGLOBIN , null::double precision as WBC , null::double precision as PLATELET , null::double precision as PTT
	, null::double precision as PT , null::double precision as INR , null::double precision as PH , null::double precision as PaO2 , null::double precision as PaCO2
	, null::double precision as BASE_EXCESS , null::double precision as BICARBONATE , null::double precision as LACTATE , null::double precision as BANDS
	, null::double precision as CRP
	-- others
	, null::double precision as IONIZEDCALCIUM
	-- mechvent
	, null::integer as MechVent
	-- ventilation parameters
	, null::double precision as FiO2
	-- urine output
	, null::double precision as urineoutput
	-- vasopressors
	, null::double precision as rate_norepinephrine , null::double precision as rate_epinephrine
	, null::double precision as rate_phenylephrine , null::double precision as rate_vasopressin
	, null::double precision as rate_dopamine , null::double precision as vaso_total
	-- intravenous fluids
	, null::double precision as iv_total
	-- cumulative fluid balance
	, cum_fluid_balance
	-- ventilation parameters
	, null::double precision as PEEP, null::double precision as tidal_volume, null::double precision as plateau_pressure
	, null::double precision as resprate_set, null::double precision as resprate_spont, null::double precision as minutevolume, null::double precision as flowrate
	, null::double precision as tidal_volume_set, null::double precision as tidal_volume_spont

	-- Others
	, null::double precision as SysPAP, null::double precision as DiasPAP, null::double precision as MeanPAP
	, null::double precision as PIP, null::double precision as O2Flow, null::double precision as MAP, null::double precision as ResistanceInsp
	, null::double precision as ResistanceExp, null::double precision as Compliance, null::double precision as VitalCapacity
	, null::double precision as TranspulmonaryPressureExpHold
	, null::double precision as TranspulmonaryPressureInspHold
	, null::double precision as ExpRatio, null::double precision as InspRatio, null::double precision as ExpTC, null::double precision as InspTC
	, null::double precision as EtCO2

	FROM getcumulativefluids

) merged

group by subject_id, hadm_id, stay_id, charttime
order by subject_id, hadm_id, stay_id, charttime