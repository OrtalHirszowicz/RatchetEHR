create table {schema_name}.{cohort_table_name} as
with severalepi as (select
patientunitstayid, 
patienthealthsystemstayid,
hospitaladmitoffset,
hospitaldischargeoffset,
culturetakenoffset,
case 
when organism in ('Staphylococcus aureus', 'Staphylococcus epidermidis', 'gram negative rods', 'gram positive cocci', 'gram positive cocci - in clusters', 'Streptococcus species, other', 'gram positive cocci - in chains', 'gram positive rods', 'Staphylococcus hominis', 'Streptococcus pneumoniae') then 1
else 0 end as y
from 
eicu_crd.microlab 
join 
eicu_crd.patient as p
using(patientunitstayid)
where 
(culturesite = 'Blood, Central Line' or culturesite = 'Blood, Venipuncture') --and culturetakenoffset >= (48 * 60)
and (hospitaldischargeoffset - hospitaladmitoffset) >= (48*60)
),
one_true as (
    SELECT
    patientunitstayid,
    patienthealthsystemstayid,
    hospitaladmitoffset,
    hospitaldischargeoffset,
    min(culturetakenoffset),
    y

    from
    severalepi
    WHERE
    y = 1

    group by
    patientunitstayid,
    patienthealthsystemstayid,
    hospitaladmitoffset,
    hospitaldischargeoffset,
    y
),
one_false as (
    SELECT
    patientunitstayid,
    patienthealthsystemstayid,
    hospitaladmitoffset,
    hospitaldischargeoffset,
    min(culturetakenoffset),
    y

    from
    severalepi
    WHERE
    y = 0
    AND
    patientunitstayid not in (select distinct patientunitstayid from one_true)

    group by
    patientunitstayid,
    patienthealthsystemstayid,
    hospitaladmitoffset,
    hospitaldischargeoffset,
    y
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
),
hospital_info as (
    select 
        count(patientunitstayid), hospitalid 
    from 
        all_cohort
join 
    eicu_crd.patient 
using   
    (patientunitstayid) 
group by
    hospitalid
),
good_hospitals as (
    select 
    hospitalid 
    from 
    hospital_info
    where 
    count > 50
),
icu_ids_to_take as (
    select
    patientunitstayid
    from 
    eicu_crd.patient
    where
    hospitalid not in (select * from good_hospitals)
)

select 
Distinct patientunitstayid as example_id,
patienthealthsystemstayid as person_id, 
(date '2000-1-1' + (hospitaladmitoffset * interval  '1 minutes')) as start_datetime,
(date '2000-1-1' + (hospitaladmitoffset * interval  '1 minutes'))::date as start_date,
(date '2000-1-1' + (min * interval  '1 minutes')) as end_datetime, 
(date '2000-1-1' + (min * interval  '1 minutes'))::date as end_date,  
y

from 
all_cohort
where 
patientunitstayid in (select * from icu_ids_to_take)
order by patientunitstayid