-- Description: This query creates a new getothers table, which contains
-- 	  the values of other variables for patients in the MIMIC-IV database.
--    The variables included in this table are retrieved by itemid
--    separately, as they were not available in the provided concepts tables.
-- Inspired by: https://github.com/arnepeine/ventai/blob/main/getOthers.sql
-- Execution time: Roughly 1 minute.
-- Number of Rows: 1709391 (1.7 million)

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS getothers; CREATE TABLE getothers AS

WITH ce AS
(
  select ce.stay_id
    , ce.subject_id
    , ce.hadm_id
    , ce.charttime
    , (case when itemid in (225667)  then valuenum else null end) as IonizedCalcium
    , (case when itemid in (220059)  then valuenum else null end) as SysPAP
    , (case when itemid in (220060)  then valuenum else null end) as DiasPAP
    , (case when itemid in (220061)  then valuenum else null end) as MeanPAP
    , (case when itemid in (224695)  then valuenum else null end) as PIP
    , (case when itemid in (223834, 227287, 227582)  then valuenum else null end) as O2Flow
    , (case when itemid in (224697)  then valuenum else null end) as MAP
    , (case when itemid in (229665)  then valuenum else null end) as ResistanceInsp
    , (case when itemid in (229664)  then valuenum else null end) as ResistanceExp
    , (case when itemid in (229661)  then valuenum else null end) as Compliance
    , (case when itemid in (224420)  then valuenum else null end) as VitalCapacity
    , (case when itemid in (224746)  then valuenum else null end) as TranspulmonaryPressureExpHold
    , (case when itemid in (224747)  then valuenum else null end) as TranspulmonaryPressureInspHold
    , (case when itemid in (226871)  then valuenum else null end) as ExpRatio
    , (case when itemid in (226873)  then valuenum else null end) as InspRatio
    , (case when itemid in (229660)  then valuenum else null end) as ExpTC
    , (case when itemid in (224738)  then valuenum else null end) as InspTC
    , (case when itemid in (228640)  then valuenum else null end) as EtCO2
    FROM chartevents ce
    WHERE ce.itemid in (
      225667, -- Ionized Calcium
      220059, -- PAP Systolic
      220060, -- PAP Diastolic
      220061, -- PAP Mean
      224695, -- Peak Insp. Pressure (PIP)
      223834, 227287, 227582, -- O2 Flow
      224697, -- Mean Airway Pressure (MAP)
      229665, -- Resistance Insp.
      229664, -- Resistance Exp.
      229661, -- Compliance
      224420, -- Vital Capacity
      224746, -- Transpulmonary Pressure (Exp. Hold)
      224747, -- Transpulmonary Pressure (Insp. Hold)
      226871, -- Expiratory Ratio
      226873, -- Inspiratory Ratio
      229660, -- Expiration Time Constant (RCexp)
      224738, -- Inspiratory Time Constant (seconds)
      228640  -- EtCO2
    )
)

SELECT
  	subject_id
  , hadm_id
  , ce.stay_id
  , ce.charttime
  , AVG(IonizedCalcium) as IonizedCalcium
  , AVG(SysPAP) as SysPAP
  , AVG(DiasPAP) as DiasPAP
  , AVG(MeanPAP) as MeanPAP
  , AVG(PIP) as PIP
  , AVG(O2Flow) as O2Flow
  , AVG(MAP) as MAP
  , AVG(ResistanceInsp) as ResistanceInsp
  , AVG(ResistanceExp) as ResistanceExp
  , AVG(Compliance) as Compliance
  , AVG(VitalCapacity) as VitalCapacity
  , AVG(TranspulmonaryPressureExpHold) as TranspulmonaryPressureExpHold
  , AVG(TranspulmonaryPressureInspHold) as TranspulmonaryPressureInspHold
  , AVG(ExpRatio) as ExpRatio
  , AVG(InspRatio) as InspRatio
  , AVG(ExpTC) as ExpTC
  , AVG(InspTC) as InspTC
  , AVG(EtCO2) as EtCO2
  from ce
  group by ce.subject_id,ce.hadm_id,ce.stay_id, ce.charttime