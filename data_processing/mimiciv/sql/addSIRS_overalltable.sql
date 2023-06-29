-- Description: This query creates a new sirs_overalltable_hourly table, which contains
--      the SIRS score for patients in the MIMIC-IV database.
-- Source: https://github.com/florisdenhengst/ventai/blob/main/getSIRS_withventparams.sql
-- Execution time: Roughly 1 minute.
-- Number of rows: 52092393 (52 million) for 4-hour window, 58875742 (59 million) for 1-hour window.

SET search_path TO public, mimiciv_derived, mimiciv_core, mimiciv_hosp, mimiciv_icu, mimiciv_ed;

DROP TABLE IF EXISTS sirs_overalltable_hourly; CREATE TABLE sirs_overalltable_hourly AS

with scorecomp as(
    SELECT stay_id, subject_id , start_time
        , tempC , heartrate , resprate , paco2 , wbc , bands
    FROM sampled_overalltable_hourly -- Switch to alternative table to use other sampling window.
),
scorecalc as
(
    SELECT stay_id, subject_id , start_time
        , tempC , heartrate , resprate , paco2 , wbc , bands

    , case
        when Tempc < 36.0 then 1
        when Tempc > 38.0 then 1
        when Tempc is null then null
        else 0
    end as Temp_score


    , case
        when HeartRate > 90.0  then 1
        when HeartRate is null then null
        else 0
    end as HeartRate_score

    , case
        when RespRate > 20.0  then 1
        when PaCO2 < 32.0  then 1
        when coalesce(RespRate, PaCO2) is null then null
        else 0
    end as Resp_score

    , case
        when WBC <  4.0  then 1
        when WBC > 12.0  then 1
        when Bands > 10 then 1-- > 10% immature neurophils (band forms)
        when coalesce(WBC, Bands) is null then null
        else 0
    end as WBC_score

    from scorecomp
)

select stay_id, subject_id , start_time
  -- Combine all the scores to get SIRS
  -- Impute 0 if the score is missing
  , coalesce(Temp_score,0)
  + coalesce(HeartRate_score,0)
  + coalesce(Resp_score,0)
  + coalesce(WBC_score,0)
    as SIRS
  , Temp_score, HeartRate_score, Resp_score, WBC_score
from scorecalc;

