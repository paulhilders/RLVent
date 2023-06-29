-- Description: This query creates a new sirs_overalltable_hourly table, which contains
--      the SIRS score for patients in the AmsterdamUMCdb database.
-- Source: https://github.com/florisdenhengst/ventai/blob/main/getSIRS_withventparams.sql
-- Execution time: Roughly 1 minute.
-- Number of rows: 1764311 (1.8 million) for 4-hour window, 4927360 (4.9 million) for 1-hour window.

SET search_path TO public, amsterdamumcdb, amsterdamumcdb_derived;

DROP TABLE IF EXISTS sirs_overalltable_hourly; CREATE TABLE sirs_overalltable_hourly AS

with scorecomp as(
    SELECT admissionid , start_time
        , tempC , heartrate , resprate , paco2 , wbc
    FROM sampled_overalltable_hourly
),
scorecalc as
(
    SELECT admissionid , start_time
        , tempC , heartrate , resprate , paco2 , wbc

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
        when WBC is null then null
        else 0
    end as WBC_score

    from scorecomp
)

select admissionid , start_time
  -- Combine all the scores to get SIRS
  -- Impute 0 if the score is missing
  , coalesce(Temp_score,0)
  + coalesce(HeartRate_score,0)
  + coalesce(Resp_score,0)
  + coalesce(WBC_score,0)
    as SIRS
  , Temp_score, HeartRate_score, Resp_score, WBC_score
from scorecalc;

