-- Description: This query copies all records from the sampled overalltable combined
--              table to a CSV file.
-- Source: https://github.com/florisdenhengst/ventai/blob/main/to_csv.sql
-- Execution time: Roughly 5 seconds.
-- Number of Rows copied: 525680 for 4-hour window, 2068197 for 1-hour window.

COPY
(
    SELECT * FROM sampled_overalltable_combined_hourly
    WHERE
    sampled_overalltable_combined_hourly.stay_id IN (
        SELECT distinct(imvd.stay_id)
        FROM getinvasiveventdurations imvd
        LEFT JOIN getdemographics dem
            ON imvd.stay_id = dem.stay_id

        WHERE imvd.vent_duration_h >= '24 hours'
        AND dem.admission_age >= 18
        AND (dem.mort90day is not null
            OR dem.hospmort is not null))
)
TO '/tmp/ventilatedpatients_hourly.csv' DELIMITER ',' CSV HEADER;
-- Note: do not forget to move the CSV file from the /tmp/ folder to your data folder.
--       E.g., cp /tmp/ventilatedpatients_hourly.csv ../../data