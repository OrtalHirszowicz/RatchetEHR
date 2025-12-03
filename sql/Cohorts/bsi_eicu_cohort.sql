create table {schema_name}.{cohort_table_name} as
with severalepi as (select
patientunitstayid, 
patienthealthsystemstayid,
hospitaladmitoffset,
unitDischargeOffset,
culturetakenoffset,
case 
when organism in ('Staphylococcus aureus', 'Staphylococcus epidermidis', 'gram negative rods', 'gram positive cocci', 'gram positive cocci - in clusters', 'Streptococcus species, other', 'gram positive cocci - in chains', 'gram positive rods', 'Staphylococcus hominis', 'Streptococcus pneumoniae') then 1
-- when organism in ('Escherichia coli', 'Klebsiella pneumoniae', 'Klebsiella oxytoca', 'Enterobacter species', 'Proteus mirabilis', 'Pseudomonas aeruginosa', 'Haemophilus influenzae', 'Campylobacter fetus', 'Staphylococcus aureus', 'Enterococcus faecalis', 'Enterococcus faecium', 'Streptococcus pneumoniae', 'Streptococcus pyogenes', 'Candida tropicalis', 'Candida parapsilosis') then 1
else 0 end as y
from 
eicu_crd.microlab 
join 
eicu_crd.patient as p
using(patientunitstayid)
where 
(culturesite = 'Blood, Central Line' or culturesite = 'Blood, Venipuncture') --and culturetakenoffset >= (48 * 60)
and (unitDischargeOffset - hospitaladmitoffset) >= (48*60)
),
count_severalepi as (
    select count(distinct patientunitstayid) as count from severalepi
),
one_true as (
    SELECT
    patientunitstayid,
    patienthealthsystemstayid,
    hospitaladmitoffset,
    unitDischargeOffset,
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
    unitDischargeOffset,
    y
),
count_one_true as (
    select count(*) as count from one_true
),
one_false as (
    SELECT
    patientunitstayid,
    patienthealthsystemstayid,
    hospitaladmitoffset,
    unitDischargeOffset,
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
    unitDischargeOffset,
    y
),
count_one_false as (
    select count(*) as count from one_false
),
all_cohort as (
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
count_all_cohort as (
    select count(*) as count from all_cohort
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
count_good_hospitals as (
    select count(*) as count from good_hospitals
),
icu_ids_to_take as (
    select
    patientunitstayid
    from 
    eicu_crd.patient
    where
    hospitalid in (select * from good_hospitals)
),
count_final as (
    select count(*) as count from all_cohort where patientunitstayid in (select * from icu_ids_to_take)
)

select 
Distinct patientunitstayid as example_id,
patienthealthsystemstayid as person_id, 
(date '2000-1-1' + (hospitaladmitoffset * interval  '1 minutes')) as start_datetime,
(date '2000-1-1' + (hospitaladmitoffset * interval  '1 minutes'))::date as start_date,
(date '2000-1-1' + (min * interval  '1 minutes')) as end_datetime, 
(date '2000-1-1' + (min * interval  '1 minutes'))::date as end_date,  
y,
(select count from count_severalepi) as initial_count,
(select count from count_one_true) as positive_count,
(select count from count_one_false) as negative_count,
(select count from count_all_cohort) as total_count,
(select count from count_good_hospitals) as good_hospitals_count,
(select count from count_final) as final_count

from 
all_cohort
where 
patientunitstayid in (select * from icu_ids_to_take)
order by patientunitstayid