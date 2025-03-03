create table {schema_name}.{cohort_table_name} as

with
-- "prepared_admissions" as 
-- (
-- 	select
-- 		person_id, observation.visit_detail_id as icustay_id, observation_datetime as intime
-- 	from
--   		{cdm_schema}.visit_detail
--  		join 
-- 		{cdm_schema}.observation
-- 		using (person_id)
-- 	where 
-- 		(observation_datetime > visit_start_datetime and observation_datetime <= visit_end_datetime) and observation_concept_id = 36034302
-- 		--observation_concept_id = 36034302
-- ),
-- "prepared_outtime" as (
-- 	select 
-- 		a.person_id, icustay_id, intime, observation_datetime as outtime
-- 	from 
-- 		{cdm_schema}.observation as a
-- 		join
-- 		prepared_admissions as b
-- 		on 
-- 		icustay_id = visit_detail_id and a.person_id = b.person_id
-- 	where 
-- 		observation_concept_id = 36659649
-- ),
-- "cohort" as (
-- 	select
-- 		icustay_id as example_id,
-- 		person_id,
-- 		intime::date as start_date,
-- 		outtime::date as end_date,
-- 		intime as start_datetime,
-- 		outtime as end_datetime,
-- 		0 as y
-- 	from
-- 		prepared_outtime
-- ),
"eicu_cohort" as (
    select
        patientunitstayid as example_id,
        patienthealthsystemstayid as person_id,
        (date '2000-1-1' + interval '0 minutes')::date as start_date,
        (date '2000-1-1' + (hospitaldischargeoffset * interval  '1 minutes'))::date as end_date,
        (date '2000-1-1' + interval '0 minutes') as start_datetime,
        (date '2000-1-1' + (hospitaldischargeoffset * interval  '1 minutes')) as end_datetime,
        0 as y
    from
    -- Ortal - change to generic way like cdm-schema
        eicu_crd.patient
)
select * from eicu_cohort
-- UNION ALL
-- select * from cohort