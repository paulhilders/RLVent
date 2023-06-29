-- Description: This query creates a new sofa_overalltable_hourly table, which contains
--      the SOFA scores for patients in the MIMIC-IV database.
-- Source: https://github.com/florisdenhengst/ventai/blob/main/getSOFA_withventparams.sql
-- Execution time: Roughly 1 minute.
-- Number of rows: 52092393 (52 million) for 4-hour window, 58875742 (59 million) for 1-hour window.

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS sofa_overalltable_hourly; CREATE TABLE sofa_overalltable_hourly AS

with scorecomp as(
    SELECT stay_id, subject_id, start_time
        -- respiration
        , PaO2FiO2ratio , mechvent
        -- nervous system
        , gcs
        -- cardiovascular system
        , meanbp, rate_dopamine
        , rate_norepinephrine, rate_epinephrine
        -- liver
        , bilirubin
        -- coagulation
        , platelet
        -- kidneys (renal)
        , creatinine, urineoutput

    FROM sampled_overalltable_hourly -- Switch to alternative table to use other sampling window.
),
scorecalc as
(
SELECT stay_id, subject_id, start_time , PaO2FiO2ratio , mechvent , gcs, meanbp , rate_dopamine , rate_norepinephrine, rate_epinephrine
       , bilirubin , platelet , creatinine, urineoutput

	, case
        when PaO2FiO2ratio < 100 and mechvent=1 then 4
        when PaO2FiO2ratio < 200 and mechvent=1 then 3
        when PaO2FiO2ratio < 300                then 2
        when PaO2FiO2ratio < 400                then 1
        when PaO2FiO2ratio is null then null
        else 0
    end as respiration

	  -- Neurological failure (GCS)
    , case
        when (gcs >= 13 and gcs <= 14) then 1
        when (gcs >= 10 and gcs <= 12) then 2
        when (gcs >=  6 and gcs <=  9) then 3
        when  gcs <   6 then 4
        when  gcs is null then null
        else 0
    end as cns

    -- Cardiovascular
    , case
        when rate_dopamine > 15 or rate_epinephrine >  0.1 or rate_norepinephrine >  0.1 then 4
        when rate_dopamine >  5 or rate_epinephrine <= 0.1 or rate_norepinephrine <= 0.1 then 3
        when rate_dopamine <=  5 /*or rate_dobutamine > 0*/ then 2
        when MeanBP < 70 then 1
        when coalesce(MeanBP, rate_dopamine, /*rate_dobutamine,*/ rate_epinephrine, rate_norepinephrine) is null then null
        else 0
    end as cardiovascular

        -- Liver
    , case
        -- Bilirubin checks in mg/dL
        when Bilirubin >= 12.0 then 4
        when Bilirubin >= 6.0  then 3
        when Bilirubin >= 2.0  then 2
        when Bilirubin >= 1.2  then 1
        when Bilirubin is null then null
        else 0
    end as liver

        -- Coagulation
    , case
        when platelet < 20  then 4
        when platelet < 50  then 3
        when platelet < 100 then 2
        when platelet < 150 then 1
        when platelet is null then null
        else 0
    end as coagulation

        -- Renal failure - high creatinine or low urine output
    , case
        when (Creatinine >= 5.0) then 4
        when  UrineOutput < 200 then 4
        when (Creatinine >= 3.5 and Creatinine < 5.0) then 3
        when  UrineOutput < 500 then 3
        when (Creatinine >= 2.0 and Creatinine < 3.5) then 2
        when (Creatinine >= 1.2 and Creatinine < 2.0) then 1
        when coalesce(UrineOutput, Creatinine) is null then null
        else 0
    end as renal

	from scorecomp
)

SELECT stay_id, subject_id, start_time
    -- parameters from scorecomp
    , PaO2FiO2ratio , mechvent , gcs, meanbp , rate_dopamine , rate_norepinephrine, rate_epinephrine
    , bilirubin , platelet , creatinine, urineoutput
    -- parameters from scorecalc, contains separate scores to estimate the final SOFA score
    , respiration , cns , cardiovascular , liver , coagulation , renal
    -- overall SOFA score calculation
    , coalesce(respiration,0) + coalesce(cns,0)
    + coalesce(cardiovascular,0) + coalesce(liver,0)
    + coalesce(coagulation,0) + coalesce(renal,0) as SOFA

FROM scorecalc

ORDER BY stay_id, subject_id, start_time
