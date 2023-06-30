-- Description: This script creates a table with all the variables and measurements
-- 		we want to use for our analysis.
-- Inspired by: https://github.com/arnepeine/ventai/blob/main/overalltable_Lab_withventparams.sql
-- Execution time: Roughly 1 minute.
-- Number of Rows: 40867783 (40.9 million)

SET search_path TO public, amsterdamumcdb, amsterdamumcdb_derived;

DROP TABLE IF EXISTS merged_overalltable; CREATE TABLE merged_overalltable AS

SELECT merged.admissionid, charttime
    -- GCS
    , avg(gcs) as gcs

    --vital signs
    , avg(HeartRate) as HeartRate , avg(SysBP) as SysBP
    , avg(DiasBP) as DiasBP , avg(MeanBP) as MeanBP , avg(SysBP)/avg(HeartRate) as shockindex
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
    , max(plateau_pressure) as plateau_pressure

    -- multiply by 100 because FiO2 is in a % but should be a fraction. This idea is retrieved from https://github.com/MIT-LCP/mimic-code/blob/master/concepts/firstday/blood-gas-first-day-arterial.sql
    , avg(PaO2)/avg(Fio2)*100 as PaO2FiO2ratio

    -- MechVent
    , avg(MechVent) as MechVent

    -- Urine output
    , avg(urineoutput) as urineoutput

    -- vasopressors
    , avg(vaso_total) as vaso_total
    , avg(dopamine_rate) as dopamine_rate, avg(dobutamine_rate) as dobutamine_rate
    , avg(epinefrine_rate) as epinefrine_rate, avg(norepinefrine_rate) as norepinefrine_rate

    -- intravenous fluids
    , avg(iv_total) as iv_total

    -- cumulated fluid balance
    , avg(cum_fluid_balance) as cum_fluid_balance

FROM
(
    SELECT n.admissionid, charttime
    -- GCS
    , null::double precision as gcs

    --vital signs
    , HeartRate , SysBP
    , DiasBP , MeanBP
    , resprate, resprate_spont, resprate_set
    , TempC , SpO2

    --lab values
    , POTASSIUM , SODIUM , CHLORIDE , GLUCOSE
    , BUN , CREATININE , MAGNESIUM , CALCIUM
    , BILIRUBIN , ALBUMIN
    , HEMOGLOBIN , WBC , PLATELET , PTT
    , INR , PH , PaO2 , PaCO2
    , BASE_EXCESS , BICARBONATE , LACTATE
    , CRP, ionizedcalcium
    , ANIONGAP

    -- Mechanical Ventilation
    , PEEP, PCabovePEEP, PSabovePEEP
    , tidal_volume, FiO2
    , tidal_volume_set
    , PIP, MAP
    , EtCO2
    , plateau_pressure

    -- MechVent
    , null::integer as MechVent

    -- Urine output
    , null::double precision as urineoutput

    -- vasopressors
    , null::double precision as vaso_total
    , null::double precision as dopamine_rate, null::double precision as dobutamine_rate
    , null::double precision as epinefrine_rate, null::double precision as norepinefrine_rate

    -- intravenous fluids
    , null::double precision as iv_total

    -- cumulated fluid balance
    , null::double precision as cum_fluid_balance

    FROM getnumevents n

UNION ALL

    SELECT n.admissionid, charttime
    -- GCS
    , gcs_score as gcs

    --vital signs
    , null::double precision as HeartRate , null::double precision as SysBP
    , null::double precision as DiasBP , null::double precision as MeanBP
    , null::double precision as resprate, null::double precision as resprate_spont, null::double precision as resprate_set
    , null::double precision as TempC , null::double precision as SpO2

    --lab values
    , null::double precision as POTASSIUM , null::double precision as SODIUM , null::double precision as CHLORIDE , null::double precision as GLUCOSE
    , null::double precision as BUN , null::double precision as CREATININE , null::double precision as MAGNESIUM , null::double precision as CALCIUM
    , null::double precision as BILIRUBIN , null::double precision as ALBUMIN
    , null::double precision as HEMOGLOBIN , null::double precision as WBC , null::double precision as PLATELET , null::double precision as PTT
    , null::double precision as INR , null::double precision as PH , null::double precision as PaO2 , null::double precision as PaCO2
    , null::double precision as BASE_EXCESS , null::double precision as BICARBONATE , null::double precision as LACTATE
    , null::double precision as CRP, null::double precision as ionizedcalcium
    , null::double precision as ANIONGAP

    -- Mechanical Ventilation
    , null::double precision as PEEP, null::double precision as PCabovePEEP, null::double precision as PSabovePEEP
    , null::double precision as tidal_volume, null::double precision as FiO2
    , null::double precision as tidal_volume_set
    , null::double precision as PIP, null::double precision as MAP
    , null::double precision as EtCO2
    , null::double precision as plateau_pressure

    -- MechVent
    , null::integer as MechVent

    -- Urine output
    , null::double precision as urineoutput

    -- vasopressors
    , null::double precision as vaso_total
    , null::double precision as dopamine_rate, null::double precision as dobutamine_rate
    , null::double precision as epinefrine_rate, null::double precision as norepinefrine_rate


    -- intravenous fluids
    , null::double precision as iv_total

    -- cumulated fluid balance
    , null::double precision as cum_fluid_balance

    FROM getgcs n

UNION ALL

    SELECT n.admissionid, charttime
    -- GCS
    , null::double precision as gcs

    --vital signs
    , null::double precision as HeartRate , null::double precision as SysBP
    , null::double precision as DiasBP , null::double precision as MeanBP
    , null::double precision as resprate, null::double precision as resprate_spont, null::double precision as resprate_set
    , null::double precision as TempC , null::double precision as SpO2

    --lab values
    , null::double precision as POTASSIUM , null::double precision as SODIUM , null::double precision as CHLORIDE , null::double precision as GLUCOSE
    , null::double precision as BUN , null::double precision as CREATININE , null::double precision as MAGNESIUM , null::double precision as CALCIUM
    , null::double precision as BILIRUBIN , null::double precision as ALBUMIN
    , null::double precision as HEMOGLOBIN , null::double precision as WBC , null::double precision as PLATELET , null::double precision as PTT
    , null::double precision as INR , null::double precision as PH , null::double precision as PaO2 , null::double precision as PaCO2
    , null::double precision as BASE_EXCESS , null::double precision as BICARBONATE , null::double precision as LACTATE
    , null::double precision as CRP, null::double precision as ionizedcalcium
    , null::double precision as ANIONGAP

    -- Mechanical Ventilation
    , null::double precision as PEEP, null::double precision as PCabovePEEP, null::double precision as PSabovePEEP
    , null::double precision as tidal_volume, null::double precision as FiO2
    , null::double precision as tidal_volume_set
    , null::double precision as PIP, null::double precision as MAP
    , null::double precision as EtCO2
    , null::double precision as plateau_pressure

    -- MechVent
    , mechvent as MechVent

    -- Urine output
    , null::double precision as urineoutput

    -- vasopressors
    , null::double precision as vaso_total
    , null::double precision as dopamine_rate, null::double precision as dobutamine_rate
    , null::double precision as epinefrine_rate, null::double precision as norepinefrine_rate


    -- intravenous fluids
    , null::double precision as iv_total

    -- cumulated fluid balance
    , null::double precision as cum_fluid_balance

    FROM getmechvent n

UNION ALL

    SELECT n.admissionid, charttime
    -- GCS
    , null::double precision as gcs

    --vital signs
    , null::double precision as HeartRate , null::double precision as SysBP
    , null::double precision as DiasBP , null::double precision as MeanBP
    , null::double precision as resprate, null::double precision as resprate_spont, null::double precision as resprate_set
    , null::double precision as TempC , null::double precision as SpO2

    --lab values
    , null::double precision as POTASSIUM , null::double precision as SODIUM , null::double precision as CHLORIDE , null::double precision as GLUCOSE
    , null::double precision as BUN , null::double precision as CREATININE , null::double precision as MAGNESIUM , null::double precision as CALCIUM
    , null::double precision as BILIRUBIN , null::double precision as ALBUMIN
    , null::double precision as HEMOGLOBIN , null::double precision as WBC , null::double precision as PLATELET , null::double precision as PTT
    , null::double precision as INR , null::double precision as PH , null::double precision as PaO2 , null::double precision as PaCO2
    , null::double precision as BASE_EXCESS , null::double precision as BICARBONATE , null::double precision as LACTATE
    , null::double precision as CRP, null::double precision as ionizedcalcium
    , null::double precision as ANIONGAP

    -- Mechanical Ventilation
    , null::double precision as PEEP, null::double precision as PCabovePEEP, null::double precision as PSabovePEEP
    , null::double precision as tidal_volume, null::double precision as FiO2
    , null::double precision as tidal_volume_set
    , null::double precision as PIP, null::double precision as MAP
    , null::double precision as EtCO2
    , null::double precision as plateau_pressure

    -- MechVent
    , null::integer as MechVent

    -- Urine output
    , urineoutput

    -- vasopressors
    , null::double precision as vaso_total
    , null::double precision as dopamine_rate, null::double precision as dobutamine_rate
    , null::double precision as epinefrine_rate, null::double precision as norepinefrine_rate


    -- intravenous fluids
    , null::double precision as iv_total

    -- cumulated fluid balance
    , null::double precision as cum_fluid_balance

    FROM geturineoutput n

UNION ALL

    SELECT n.admissionid, n.starttime as charttime
    -- GCS
    , null::double precision as gcs

    --vital signs
    , null::double precision as HeartRate , null::double precision as SysBP
    , null::double precision as DiasBP , null::double precision as MeanBP
    , null::double precision as resprate, null::double precision as resprate_spont, null::double precision as resprate_set
    , null::double precision as TempC , null::double precision as SpO2

    --lab values
    , null::double precision as POTASSIUM , null::double precision as SODIUM , null::double precision as CHLORIDE , null::double precision as GLUCOSE
    , null::double precision as BUN , null::double precision as CREATININE , null::double precision as MAGNESIUM , null::double precision as CALCIUM
    , null::double precision as BILIRUBIN , null::double precision as ALBUMIN
    , null::double precision as HEMOGLOBIN , null::double precision as WBC , null::double precision as PLATELET , null::double precision as PTT
    , null::double precision as INR , null::double precision as PH , null::double precision as PaO2 , null::double precision as PaCO2
    , null::double precision as BASE_EXCESS , null::double precision as BICARBONATE , null::double precision as LACTATE
    , null::double precision as CRP, null::double precision as ionizedcalcium
    , null::double precision as ANIONGAP

    -- Mechanical Ventilation
    , null::double precision as PEEP, null::double precision as PCabovePEEP, null::double precision as PSabovePEEP
    , null::double precision as tidal_volume, null::double precision as FiO2
    , null::double precision as tidal_volume_set
    , null::double precision as PIP, null::double precision as MAP
    , null::double precision as EtCO2
    , null::double precision as plateau_pressure

    -- MechVent
    , null::integer as MechVent

    -- Urine output
    , null::double precision as urineoutput

    -- vasopressors
    , vaso_total
    , dopamine_rate, dobutamine_rate
    , epinefrine_rate, norepinefrine_rate


    -- intravenous fluids
    , null::double precision as iv_total

    -- cumulated fluid balance
    , null::double precision as cum_fluid_balance

    FROM getvasopressors n

UNION ALL

    SELECT n.admissionid, charttime
    -- GCS
    , null::double precision as gcs

    --vital signs
    , null::double precision as HeartRate , null::double precision as SysBP
    , null::double precision as DiasBP , null::double precision as MeanBP
    , null::double precision as resprate, null::double precision as resprate_spont, null::double precision as resprate_set
    , null::double precision as TempC , null::double precision as SpO2

    --lab values
    , null::double precision as POTASSIUM , null::double precision as SODIUM , null::double precision as CHLORIDE , null::double precision as GLUCOSE
    , null::double precision as BUN , null::double precision as CREATININE , null::double precision as MAGNESIUM , null::double precision as CALCIUM
    , null::double precision as BILIRUBIN , null::double precision as ALBUMIN
    , null::double precision as HEMOGLOBIN , null::double precision as WBC , null::double precision as PLATELET , null::double precision as PTT
    , null::double precision as INR , null::double precision as PH , null::double precision as PaO2 , null::double precision as PaCO2
    , null::double precision as BASE_EXCESS , null::double precision as BICARBONATE , null::double precision as LACTATE
    , null::double precision as CRP, null::double precision as ionizedcalcium
    , null::double precision as ANIONGAP

    -- Mechanical Ventilation
    , null::double precision as PEEP, null::double precision as PCabovePEEP, null::double precision as PSabovePEEP
    , null::double precision as tidal_volume, null::double precision as FiO2
    , null::double precision as tidal_volume_set
    , null::double precision as PIP, null::double precision as MAP
    , null::double precision as EtCO2
    , null::double precision as plateau_pressure

    -- MechVent
    , null::integer as MechVent

    -- Urine output
    , null::double precision as urineoutput

    -- vasopressors
    , null::double precision as vaso_total
    , null::double precision as dopamine_rate, null::double precision as dobutamine_rate
    , null::double precision as epinefrine_rate, null::double precision as norepinefrine_rate


    -- intravenous fluids
    , amount as iv_total

    -- cumulated fluid balance
    , null::double precision as cum_fluid_balance

    FROM getintravenous n

UNION ALL

    SELECT n.admissionid, charttime
    -- GCS
    , null::double precision as gcs

    --vital signs
    , null::double precision as HeartRate , null::double precision as SysBP
    , null::double precision as DiasBP , null::double precision as MeanBP
    , null::double precision as resprate, null::double precision as resprate_spont, null::double precision as resprate_set
    , null::double precision as TempC , null::double precision as SpO2

    --lab values
    , null::double precision as POTASSIUM , null::double precision as SODIUM , null::double precision as CHLORIDE , null::double precision as GLUCOSE
    , null::double precision as BUN , null::double precision as CREATININE , null::double precision as MAGNESIUM , null::double precision as CALCIUM
    , null::double precision as BILIRUBIN , null::double precision as ALBUMIN
    , null::double precision as HEMOGLOBIN , null::double precision as WBC , null::double precision as PLATELET , null::double precision as PTT
    , null::double precision as INR , null::double precision as PH , null::double precision as PaO2 , null::double precision as PaCO2
    , null::double precision as BASE_EXCESS , null::double precision as BICARBONATE , null::double precision as LACTATE
    , null::double precision as CRP, null::double precision as ionizedcalcium
    , null::double precision as ANIONGAP

    -- Mechanical Ventilation
    , null::double precision as PEEP, null::double precision as PCabovePEEP, null::double precision as PSabovePEEP
    , null::double precision as tidal_volume, null::double precision as FiO2
    , null::double precision as tidal_volume_set
    , null::double precision as PIP, null::double precision as MAP
    , null::double precision as EtCO2
    , null::double precision as plateau_pressure

    -- MechVent
    , null::integer as MechVent

    -- Urine output
    , null::double precision as urineoutput

    -- vasopressors
    , null::double precision as vaso_total
    , null::double precision as dopamine_rate, null::double precision as dobutamine_rate
    , null::double precision as epinefrine_rate, null::double precision as norepinefrine_rate


    -- intravenous fluids
    , null::double precision as iv_total

    -- cumulated fluid balance
    , cum_fluid_balance

    FROM getcumulativefluids n
) merged

GROUP BY admissionid, charttime
ORDER BY admissionid, charttime
