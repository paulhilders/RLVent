-- Description: This query creates a new geturineoutput table, which
--     contains the urine output for the patients in the AmsterdamUMCdb database.
-- Number of rows: 1681873 (1.7 million)
-- Time required: Roughly 10 seconds

SET search_path TO public, amsterdamumcdb, amsterdamumcdb_derived;

DROP TABLE IF EXISTS geturineoutput; CREATE TABLE geturineoutput AS

SELECT n.admissionid
  , n.measuredat AS measuredat
  , ((n.measuredat - a.admittedat)/(1000*60)) as charttime
  , n.value AS urineoutput
FROM numericitems n
LEFT JOIN admissions a ON
  n.admissionid = a.admissionid
WHERE n.itemid IN (
  8794, --UrineCAD
  8796, --UrineSupraPubis
  8798, --UrineSpontaan
  8800, --UrineIncontinentie
  8803, --UrineUP
  10743, --Nefrodrain li Uit
  10745, --Nefrodrain re Uit
  19921, --UrineSplint Li
  19922 --UrineSplint Re
)
