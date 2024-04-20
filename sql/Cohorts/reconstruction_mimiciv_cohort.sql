create table {schema_name}.{cohort_table_name} as

with
"mimiciv_cohort" as (
    select
        stay_id as example_id,
        subject_id as person_id,
        intime::date as start_date,
        outtime::date as end_date,
        intime as start_datetime,
        outtime as end_datetime,
        case
            when anchor_year_group = '2017 - 2019' then 1
            else 0
        end as last_years,
        0 as y
    from
        mimiciv_icu.icustays
    join
        mimiciv_hosp.patients
    using
        (subject_id)

)
select * from mimiciv_cohort