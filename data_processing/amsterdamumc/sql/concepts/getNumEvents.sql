-- Description: This query creates a new getnumevents table, which retrieves
-- 	  the relevant measurements by itemid, for each patient in the
--    AmsterdamUMCdb database.
-- Number of rows: 39181747 (39.2 million)
-- Time required: Roughly 10 minutes

SET search_path TO public, amsterdamumcdb, amsterdamumcdb_derived;

DROP TABLE IF EXISTS getnumevents; CREATE TABLE getnumevents AS

WITH ne AS
(
  select n.admissionid
    , n.measuredat AS measuredat
    , ((n.measuredat - a.admittedat)/(1000*60)) as charttime

    -- Lab Values
    , (CASE WHEN itemid = 6801 THEN (value / 10) else null end) as ALBUMIN -- g/L --> g/dL 'ALBUMIN' (divide by 10 to convert g/L to g/dL))
    , (CASE WHEN itemid = 9559 THEN value else null end) as ANIONGAP -- mEq/L 'ANION GAP'
    , (CASE WHEN itemid = 9992 THEN value else null end) as BICARBONATE -- mEq/L 'BICARBONATE'
    , (CASE WHEN itemid = 9945 THEN (value * 0.06) else null end) as BILIRUBIN -- umol/L --> mg/dL 'BILIRUBIN' (converts umol/L to mg/dL)
    , (CASE WHEN itemid = 9930 THEN (value) else null end) as CHLORIDE -- mEq/L 'CHLORIDE'
    , (CASE WHEN itemid = 9941 THEN (value * 0.01131) else null end) as CREATININE-- umol/L --> mg/dL 'CREATININE'
    , (CASE WHEN itemid = 9947 THEN (value * 18) else null end) as GLUCOSE -- mmol/L --> mg/dL 'GLUCOSE' (converts mmol/L to mg/dL)
    , (CASE WHEN itemid IN (9960, 10286) THEN (value * 1.6113) else null end) as HEMOGLOBIN -- mmol/L --> g/dL 'HEMOGLOBIN'
    , (CASE WHEN itemid IN (10053) THEN (value) else null end) as LACTATE -- mmol/L 'LACTATE'
    , (CASE WHEN itemid IN (9964) THEN (value) else null end) as PLATELET -- K/uL 'PLATELET'
    , (CASE WHEN itemid IN (9927, 10285) THEN (value) else null end) as POTASSIUM -- mEq/L 'POTASSIUM'
    , (CASE WHEN itemid IN (11944, 17982) THEN (value) else null end) as PTT -- sec 'PTT'
    , (CASE WHEN itemid IN (11893, 11894) THEN (value) else null end) as INR -- 'INR'
    , (CASE WHEN itemid IN (9924, 10284) THEN (value) else null end) as SODIUM -- mEq/L == mmol/L 'SODIUM'
    , (CASE WHEN itemid IN (9943) THEN (value * 6) else null end) as BUN --mmol/L --> mg/dL 'BUN'
    , (CASE WHEN itemid IN (9965) THEN (value) else null end) as WBC -- K/uL 'WBC'
    , (CASE WHEN itemid IN (9952) THEN (value) else null end) as MAGNESIUM -- mEq/L == mmol/L 'MAGNESIUM'
    , (CASE WHEN itemid IN (9994) THEN (value) else null end) as BASE_EXCESS -- mEq/L == mmol/L 'BASE_EXCESS'
    , (CASE WHEN itemid IN (9933) THEN (value) else null end) as CALCIUM -- mEq/L == mmol/L 'CALCIUM'
    , (CASE WHEN itemid IN (10267) THEN (value) else null end) as ionizedcalcium
    , (CASE WHEN itemid IN (12310) THEN (value) else null end) as pH
    , (CASE WHEN itemid IN (9996) THEN (value) else null end) as PaO2
    , (CASE WHEN itemid IN (9990) THEN (value) else null end) as PaCO2
    , (CASE WHEN itemid IN (10079) THEN (value) else null end) as CRP -- mg/L 'CRP'

    -- Vital Signs
    , (CASE WHEN itemid IN (6640) and value != 0 THEN (value) else null end) as HeartRate
    , (CASE WHEN itemid IN (6641) THEN (value) else null end) as SysBP
    , (CASE WHEN itemid IN (6643) THEN (value) else null end) as DiasBP
    , (CASE WHEN itemid IN (6642) THEN (value) else null end) as MeanBP
    , (CASE WHEN itemid IN (8850, 8874, 12266) THEN (value) else null end) as resprate
    , (CASE WHEN itemid IN (7726) THEN (value) else null end) as resprate_spont
    , (CASE WHEN itemid IN (12283) THEN (value) else null end) as resprate_set
    , (CASE WHEN itemid IN (8658, 11889) THEN (value) else null end) as TempC
    , (CASE WHEN itemid IN (6709) THEN (value) else null end) as SpO2

    -- Mechanical Ventilation
    , (CASE WHEN itemid IN (6699, 12279, 12282) and value != 0 THEN (value) else null end) as FiO2
    , (CASE WHEN itemid IN (6694, 12284) THEN (value) else null end) as PEEP
    , (CASE WHEN itemid IN (12285) THEN (value) else null end) as PCabovePEEP
    , (CASE WHEN itemid IN (12286) THEN (value) else null end) as PSabovePEEP
    , (CASE WHEN itemid IN (12275, 12277) THEN (value) else null end) as tidal_volume
    , (CASE WHEN itemid IN (12291, 8851) THEN (value) else null end) as tidal_volume_set
    , (CASE WHEN itemid IN (6707) THEN (value) else null end) as EtCO2
    , (CASE WHEN itemid IN (12281) THEN (value) else null end) as PIP
    , (CASE WHEN itemid IN (12278) THEN (value) else null end) as MAP
    , (CASE WHEN itemid IN (8878) THEN (value) else null end) as plateau_pressure

  from amsterdamumcdb.numericitems n

  left join amsterdamumcdb.admissions a
      on n.admissionid = a.admissionid
  where n.itemid in
  (
    6801, -- Albumin
    9559, -- Anion Gap
    9992, -- Bicarbonate
    9945, -- Bilirubin
    9930, -- Chloride
    9941, -- Creatinine
    9947, -- Glucose
    9960, 10286, -- Hemoglobin
    10053, -- Lactate
    9964, -- Platelet
    9927, 10285, -- Potassium
    11944, 17982, -- PTT
    11893, 11894, -- INR
    9924, 10284, -- Sodium
    9943, -- BUN
    9965, -- WBC
    9952, -- Magnesium
    9994, -- Base Excess
    9933, -- Calcium
    10267, -- ionized calcium
    12310, -- pH
    9996, -- PaO2
    9990, -- PaCO2
    10079, -- CRP

    -- Vital Signs
    6640, -- HeartRate
    6641, -- SysBP
    6643, -- DiasBP
    6642, -- MeanBP
    8850, 8874, 12266, -- resprate
    7726, -- resprate_spont
    12283, -- resprate_set
    8658, 11889, -- TempC
    6709, -- SpO2

    -- Mechanical Ventilation
    6699, 12279, 12282, -- FiO2
    6694, 12284, -- PEEP
    12285, -- PCabovePEEP
    12286, -- PSabovePEEP
    12275, 12277, -- tidal_volume
    12291, 8851, -- tidal_volume_set
    6707, -- EtCO2
    12281, -- PIP
    8878, -- plateau_pressure
    12278 -- MAP
	)
)

SELECT admissionid
  , measuredat
  , charttime
  , avg(ALBUMIN) as ALBUMIN, avg(ANIONGAP) as ANIONGAP, avg(BICARBONATE) AS BICARBONATE
  , avg(BILIRUBIN) as BILIRUBIN, avg(CHLORIDE) as CHLORIDE, avg(CREATININE) as CREATININE
  , avg(GLUCOSE) as GLUCOSE, avg(HEMOGLOBIN) as HEMOGLOBIN, avg(LACTATE) as LACTATE
  , avg(PLATELET) as PLATELET, avg(POTASSIUM) as POTASSIUM, avg(PTT) as PTT
  , avg(INR) as INR, avg(SODIUM) as SODIUM, avg(BUN) as BUN, avg(WBC) as WBC
  , avg(MAGNESIUM) as MAGNESIUM, avg(BASE_EXCESS) as BASE_EXCESS, avg(CALCIUM) as CALCIUM
  , avg(ionizedcalcium) as ionizedcalcium, avg(pH) as pH, avg(PaO2) as PaO2
  , avg(PaCO2) as PaCO2, avg(CRP) as CRP

  , avg(HeartRate) as HeartRate, avg(SysBP) as SysBP, avg(DiasBP) as DiasBP
  , avg(MeanBP) as MeanBP, avg(resprate) as resprate, avg(resprate_spont) as resprate_spont
  , avg(resprate_set) as resprate_set, avg(TempC) as TempC, avg(SpO2) as SpO2

  , avg(FiO2) as FiO2, avg(PEEP) as PEEP, avg(PCabovePEEP) as PCabovePEEP
  , avg(PSabovePEEP) as PSabovePEEP, avg(tidal_volume) as tidal_volume
  , avg(tidal_volume_set) as tidal_volume_set, avg(EtCO2) as EtCO2
  , avg(PIP) as PIP, avg(MAP) as MAP
  , avg(plateau_pressure) AS plateau_pressure
  from ne
  group by ne.admissionid, ne.measuredat, ne.charttime
