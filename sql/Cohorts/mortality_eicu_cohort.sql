create table {schema_name}.{cohort_table_name} as



with severalepi as (select
patientunitstayid, 
patienthealthsystemstayid,
hospitaladmitoffset,
hospitaldischargeoffset,
case 
when unitdischargestatus = 'Expired' and hospitaldischargeoffset > 24 * 7 * 60 then 1
else 0 end as y
from 
eicu_crd.patient
where
(hospitaldischargeoffset - hospitaladmitoffset) >= (48*60)
),
one_true as (
    SELECT
    patientunitstayid,
    patienthealthsystemstayid,
    hospitaladmitoffset,
    hospitaldischargeoffset,
    y

    from
    severalepi
    WHERE
    y = 1


),
one_false as (
    SELECT
    patientunitstayid,
    patienthealthsystemstayid,
    hospitaladmitoffset,
    hospitaldischargeoffset,
    y

    from
    severalepi
    WHERE
    y = 0
), all_cohort as (
select
*
FROM
one_true
union all
select
*
FROM
one_false
)
select 
patientunitstayid as example_id,
patienthealthsystemstayid as person_id, 
(date '2000-1-1' + (hospitaladmitoffset * interval  '1 minutes')) as start_datetime,
(date '2000-1-1' + (hospitaladmitoffset * interval  '1 minutes'))::date as start_date,
(date '2000-1-1' + (hospitaldischargeoffset * interval  '1 minutes')) as end_datetime, 
(date '2000-1-1' + (hospitaldischargeoffset * interval  '1 minutes'))::date as end_date,  
y

from 
all_cohort