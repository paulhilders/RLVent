-- Description: This query creates a new getintravenous table, which contains
-- 	  the average amount of intravenous fluids administered to patients in the
-- 	  AmsterdamUMCdb database.
-- Number of rows: 3197398 (3.2 million)
-- Time required: Roughly 10 seconds

SET search_path TO public, amsterdamumcdb, amsterdamumcdb_derived;

DROP TABLE IF EXISTS getintravenous; CREATE TABLE getintravenous AS

WITH intra_fluids AS (
  SELECT n.admissionid
    , (start - a.admittedat)/(1000*60) AS starttime
    , (stop - a.admittedat)/(1000*60) AS stoptime
    , fluidin
  FROM drugitems n
  LEFT JOIN admissions a ON
    n.admissionid = a.admissionid
)

SELECT admissionid
  , starttime AS charttime
  , AVG(fluidin) AS amount
FROM intra_fluids
GROUP BY admissionid, starttime
ORDER BY admissionid, starttime
