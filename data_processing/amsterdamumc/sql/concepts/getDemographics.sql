-- Description: This query creates a new getdemographics table, which contains
--          demographic information about the patients in the AmsterdamUMCdb database.
-- Number of rows: 23106 (23 thousand)
-- Time required: Roughly 3 seconds
-- Note, the mean weight, age, and height have been based on dataset
--    statistics for the AmsterdamUMCdb specifically. Therefore, these values
--    are not generalizable to other datasets.

SET search_path TO public, amsterdamumcdb, amsterdamumcdb_derived;

DROP TABLE IF EXISTS getdemographics; CREATE TABLE getdemographics AS

SELECT *
  , CASE
      WHEN weightgroup LIKE '%59%' THEN 55
      WHEN weightgroup LIKE '%60%' THEN 65
      WHEN weightgroup LIKE '%70%' THEN 75
      WHEN weightgroup LIKE '%80%' THEN 85
      WHEN weightgroup LIKE '%90%' THEN 95
      WHEN weightgroup LIKE '%100%' THEN 105
      WHEN weightgroup LIKE '%110%' THEN 115
      ELSE 80 --mean weight
    END as weight
  , CASE
      WHEN agegroup LIKE '%18%' THEN 29
      WHEN agegroup LIKE '%40%' THEN 45
      WHEN agegroup LIKE '%50%' THEN 55
      WHEN agegroup LIKE '%60%' THEN 65
      WHEN agegroup LIKE '%70%' THEN 75
      WHEN agegroup LIKE '%80%' THEN 85
      ELSE 62 --mean age
    END as age
  , CASE
      WHEN heightgroup LIKE '%159%' THEN 155
      WHEN heightgroup LIKE '%160%' THEN 165
      WHEN heightgroup LIKE '%170%' THEN 175
      WHEN heightgroup LIKE '%180%' THEN 185
      WHEN heightgroup LIKE '%190%' THEN 195
      ELSE 175 --mean height
    END as height
  , CASE WHEN (a.destination = 'Overleden') THEN True ELSE False END AS hospmort
  , CAST((a.destination = 'Overleden') AS INT) AS hospmort_int
  , CASE WHEN to_timestamp(a.dateofdeath/1000)::date <= (to_timestamp(a.admittedat/1000)::date + interval '90 day') THEN True ELSE False END AS mort90day
  , CASE WHEN (a.admissioncount = 1) THEN True ELSE False END AS first_icu_stay
  , CASE WHEN (a.admissioncount > 1) THEN True ELSE False END AS icu_readmission
FROM admissions a

