-- Inspired by: https://github.com/arnepeine/ventai/blob/main/sampling_lab_withventparams.sql
-- Execution time: Roughly 2.5 hours for 1-hour window.
-- Number of Rows: 4927360 (4.9 million) for 1-hour window

SET search_path TO public, amsterdamumcdb, amsterdamumcdb_derived;

DROP TABLE IF EXISTS sampled_overalltable_hourly; CREATE TABLE sampled_overalltable_hourly AS 

WITH minmax as
(
	SELECT admissionid, min(charttime) as mint, max(charttime) as maxt
	FROM merged_overalltable
	GROUP BY admissionid
	ORDER BY admissionid
),
grid as
(
	SELECT admissionid, GENERATE_SERIES(mint::integer, maxt::integer, 1 * 60) as start_time
	FROM minmax
	GROUP BY admissionid, mint, maxt
    ORDER BY admissionid
)

SELECT ot.admissionid, start_time
    -- GCS
    , round(avg(gcs)) as gcs

    --vital signs
    , avg(HeartRate) as HeartRate , avg(SysBP) as SysBP
    , avg(DiasBP) as DiasBP , avg(MeanBP) as MeanBP , avg(shockindex) as shockindex
    , avg(resprate) as resprate, avg(resprate_spont) as resprate_spont, avg(resprate_set) as resprate_set
    , avg(TempC) as TempC , avg(SpO2) as SpO2

    --lab values
    , avg(POTASSIUM) as POTASSIUM , avg(SODIUM) as SODIUM , avg(CHLORIDE) as CHLORIDE , avg(GLUCOSE) as GLUCOSE
    , avg(BUN) as BUN , avg(CREATININE) as CREATININE , avg(MAGNESIUM) as MAGNESIUM , avg(CALCIUM) as CALCIUM
    , avg(BILIRUBIN) as BILIRUBIN , avg(ALBUMIN) as ALBUMIN 
    , avg(HEMOGLOBIN) as HEMOGLOBIN , avg(WBC) as WBC , avg(PLATELET) as PLATELET , avg(PTT) as PTT
    , avg(INR) as INR , avg(PH) as PH , avg(PaO2) as PaO2 , avg(PaCO2) as PaCO2
    , avg(BASE_EXCESS) as BASE_EXCESS , avg(BICARBONATE) as BICARBONATE , avg(LACTATE) as LACTATE 
    , avg(CRP) as CRP, avg(ionizedcalcium) as ionizedcalcium
    , avg(ANIONGAP) as ANIONGAP

    -- Mechanical Ventilation
    , max(PEEP) as PEEP, max(PCabovePEEP) as PCabovePEEP, max(PSabovePEEP) as PSabovePEEP
    , max(tidal_volume) as tidal_volume, avg(FiO2) as FiO2
    , avg(tidal_volume_set) as tidal_volume_set, avg(tidal_volume) as tidal_volume_spont
    , avg(PIP) as PIP, avg(MAP) as MAP
    , avg(EtCO2) as EtCO2
    , max(plateau_pressure) as plateau_pressure_direct
    , (CASE
        WHEN max(plateau_pressure) IS NOT NULL THEN max(plateau_pressure)
        WHEN max(PCabovePEEP) IS NOT NULL AND max(PEEP) IS NOT NULL THEN max(PCabovePEEP) + max(PEEP)
        WHEN max(PSabovePEEP) IS NOT NULL AND max(PEEP) IS NOT NULL THEN max(PSabovePEEP) + max(PEEP)
        ELSE NULL
    END) as plateau_pressure

    -- multiply by 100 because FiO2 is in a % but should be a fraction. This idea is retrieved from https://github.com/MIT-LCP/mimic-code/blob/master/concepts/firstday/blood-gas-first-day-arterial.sql
    , avg(PaO2FiO2ratio) as pao2fio2ratio

    -- MechVent
    , max(MechVent) as MechVent

    -- Urine output
    , sum(urineoutput) as urineoutput

    -- vasopressors
    , max(vaso_total) as vaso_total
    , max(dopamine_rate) as dopamine_rate, max(dobutamine_rate) as dobutamine_rate
    , max(epinefrine_rate) as epinefrine_rate, max(norepinefrine_rate) as norepinefrine_rate

    -- intravenous fluids
    , sum(iv_total) as iv_total

    -- cumulated fluid balance
    , avg(cum_fluid_balance) as cum_fluid_balance

FROM grid g
LEFT JOIN merged_overalltable ot
    ON g.admissionid = ot.admissionid
    AND ot.charttime < g.start_time + (1 * 60)
    AND ot.charttime >= g.start_time
GROUP BY ot.admissionid, start_time
ORDER BY admissionid, start_time
