-- Description: This query creates a new getcumulativefluids table, which
-- 		merges the fluids information from the inputevents table with the
-- 		information from the outputevents table.
-- Adapted from: https://github.com/arnepeine/ventai/blob/main/getCumFluid.sql
-- Number of rows: 4546804 (4.5 million)
-- Time required: Roughly 1 minute.

SET search_path TO public, amsterdamumcdb, amsterdamumcdb_derived;

DROP TABLE IF EXISTS getcumulativefluids; CREATE TABLE getcumulativefluids AS

SELECT in_out.admissionid, charttime, in_amount, in_cum_amt, out_amount, out_cum_amt,
      sum(out_amount) OVER (PARTITION BY admissionid ORDER BY charttime) -
      sum(in_amount) OVER (PARTITION BY admissionid ORDER BY charttime) AS cum_fluid_balance
FROM (
  -- Input Events
  SELECT merged.admissionid, charttime, in_amount,
    sum(in_amount) OVER (PARTITION BY merged.admissionid ORDER BY charttime) AS in_cum_amt,
    null::double precision AS out_amount, null::double precision AS out_cum_amt
  FROM (
    SELECT di.admissionid
    , ((di.start - a.admittedat)/(1000*60)) as charttime
    , sum(fluidin) AS in_amount
    FROM drugitems di
    LEFT JOIN admissions a
    ON di.admissionid = a.admissionid
    GROUP BY di.admissionid, charttime
    HAVING sum(fluidin) > 0
  ) AS merged
  INNER JOIN admissions a
  ON a.admissionid=merged.admissionid

  UNION ALL

  --Output events.
  SELECT merged.admissionid, charttime,
    null::double precision AS in_amount, null::double precision AS in_cum_amt, out_amount,
    sum(out_amount) OVER (PARTITION BY merged.admissionid ORDER BY charttime) AS out_cum_amt
  FROM (
    SELECT n.admissionid
    , ((n.measuredat - a.admittedat)/(1000*60)) as charttime
    , sum(fluidout) AS out_amount
    FROM numericitems n
    LEFT JOIN admissions a
    ON n.admissionid = a.admissionid
    GROUP BY n.admissionid, charttime
    HAVING sum(fluidout) > 0
  ) AS merged
  INNER JOIN admissions a
  ON a.admissionid=merged.admissionid
) AS in_out
ORDER BY admissionid, charttime
