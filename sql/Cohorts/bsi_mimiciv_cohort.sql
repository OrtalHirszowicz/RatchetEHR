
create table {schema_name}.{cohort_table_name} as
with
"prepared_samples" as (
		select 
		subject_id as person_id, 
		hadm_id, 
		case 
			when charttime is null then chartdate 
			else charttime 
		end charttime, 
		spec_itemid, 
		spec_type_desc,
		max(case when org_name in (
		'AEROCOCCUS SPECIES', 
		'AEROCOCCUS VIRIDANS',
		 'BACILLUS SPECIES', 
		 'BACILLUS SPECIES''; NOT ANTHRACIS', 
		 'BETA STREPTOCOCCUS', 
		 'BETA STREPTOCOCCUS GROUP A', 
		 'BETA STREPTOCOCCUS GROUP C', 
		 'BETA STREPTOCOCCUS GROUP G', 
		 'CORYNEBACTERIUM SPECIES (DIPHTHEROIDS)', 
		 'CORYNEBACTERIUM STRIATUM', 
		 'CORYNEBCATERIUM AMYCOLATUM', 
		 'GRAM NEGATIVE COCCI', 
		 'GRAM NEGATIVE ROD #1', 
		 'GRAM NEGATIVE ROD #2', 
		 'GRAM NEGATIVE ROD #4', 
		 'GRAM NEGATIVE ROD(S)', 
		 'GRAM POSITIVE COCCUS(COCCI)', 
		 'MICROCOCCUS/STOMATOCOCCUS SPECIES', 
		 'NUTRITIONALLY VARIANT STREPTOCOCCUS', 
		 'PRESUMPTIVE PROPIONIBACTERIUM ACNES', 
		 'PRESUMPTIVE STREPTOCOCCUS BOVIS', 
		 'PROBABLE MICROCOCCUS SPECIES', 
		 'PROPIONIBACTERIUM ACNES', 
		 'PROPIONIBACTERIUM SPECIES', 
		 'RESEMBLING MICROCOCCUS/STOMATOCOCCUS SPECIES', 
		 'ROTHIA (STOMATOCOCCUS) MUCILAGINOSA', 
		 'STAPHYLOCOCCUS, COAGULASE NEGATIVE', 
		 'STAPHYLOCOCCUS EPIDERMIDIS', 
		 'STAPHYLOCOCCUS HOMINIS', 
		 'STAPHYLOCOCCUS LUGDUNENSIS', 
		 'STREPTOCOCCUS ANGINOSUS', 
		 'STREPTOCOCCUS ANGINOSUS (MILLERI) GROUP', 
		 'STREPTOCOCCUS BOVIS',  
		 'STREPTOCOCCUS CONSTELLATUS', 
		 'STREPTOCOCCUS INFANTARIUS SSP. COLI (STREPTOCOCCUS BOVIS)', 
		 'STREPTOCOCCUS MILLERI', 
		 'STREPTOCOCCUS MILLERI GROUP', 
		 'STREPTOCOCCUS MITIS', 
		 'STREPTOCOCCUS ORALIS', 
		 'STREPTOCOCCUS PNEUMONIAE', 
		 'STREPTOCOCCUS SALIVARIUS', 
		 'STREPTOCOCCUS SPECIES', 
		 'STREPTOCOCCUS VESTIBULARIS', 
		 'VIRIDANS STREPTOCOCCI'
		 ) then 1 else 0 end) as label, 
		string_agg(org_name, ', ') org_name
	from 
		mimiciv_hosp.microbiologyevents
	where 
	-- -- Getting the values of BLOOD CULTURE, BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)
		spec_type_desc in ('BLOOD CULTURE', 'BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)') 
		and (chartdate is not null)
	group by
		subject_id, 
		hadm_id, 
		case 
			when charttime is null then chartdate 
			else charttime 
		end, 
		spec_itemid, 
		spec_type_desc
	order by
		person_id, hadm_id, charttime
	),
"prepared_admissions_los" as 
(
	select
		subject_id as person_id, hadm_id, stay_id as icustay_id, los, intime, outtime --as intime, visit_end_datetime as outtime, outtime 
	from
        mimiciv_icu.icustays
	where 
		los >= 48/24 and first_careunit in ('Medical/Surgical Intensive Care Unit (MICU/SICU)', 'Medical Intensive Care Unit (MICU)', 'Surgical Intensive Care Unit (SICU)', 'Trauma SICU (TSICU)')
		--observation_concept_id = 36034302
),
"all_data" as (
	select 
		person_id, intime::date as visit_start_date, intime as visit_start_datetime, outtime::date as visit_end_date, outtime as visit_end_datetime, icustay_id, los, hadm_id, charttime, spec_type_desc, label, org_name
	from 
		prepared_admissions_los as a
	join 
		prepared_samples as b
	using 
		(person_id, hadm_id)
	where 
		charttime between intime and outtime
	order by 
		person_id, icustay_id, charttime
), 
"first_pos_first_neg" as (
	select 
		person_id, visit_start_date, visit_start_datetime, visit_end_date, visit_end_datetime, icustay_id, los, 
		(ARRAY_AGG(charttime))[1] as charttimes, 
		(ARRAY_AGG(spec_type_desc))[1], 
		(ARRAY_AGG(label))[1] as labels, 
		(ARRAY_AGG(org_name))[1] as org_names
	from 
		all_data
	where 
		label = 0
	group by
		person_id, visit_start_date, visit_start_datetime, visit_end_date, visit_end_datetime, icustay_id, los


	UNION ALL

		select 
		person_id, visit_start_date, visit_start_datetime, visit_end_date, visit_end_datetime, icustay_id, los,
		(ARRAY_AGG(charttime order by charttime))[1] as charttimes, 
		(ARRAY_AGG(spec_type_desc order by charttime))[1], 
		(ARRAY_AGG(label order by charttime))[1] as labels, 
		(ARRAY_AGG(org_name order by charttime))[1] as org_names
	from 
		all_data
	where 
		label = 1
	group by
		person_id, visit_start_date, visit_start_datetime, visit_end_date, visit_end_datetime, icustay_id, los
	order by 
		person_id, icustay_id
), 
"all_important" as
(
	select 
		icustay_id as example_id, 
		person_id, 
		visit_start_datetime::date as start_date, 
		visit_start_datetime as start_datetime,
		charttimes::date as end_date, 
		charttimes as end_datetime, 
		labels as y
	from
	   first_pos_first_neg
	where
		labels = 0 and icustay_id not in (
		select
			icustay_id
		from
		   first_pos_first_neg
		where 
			labels = 1
	) 
	UNION ALL
		select
			icustay_id as example_id, 
			person_id, 
			visit_start_datetime::date as start_date, 
			visit_start_datetime as start_datetime,
			charttimes::date as end_date, 
			charttimes as end_datetime, 
			labels as y
		from
		first_pos_first_neg
		where 
			labels = 1
), "with_anchor_year_group" as (
	select 
	a.*,
	case 
    when anchor_year_group = '2017 - 2019' then 1
	else 0
	end as last_years
	from
	all_important a
	join
	mimiciv_hosp.patients
	on
	(subject_id = person_id)
)
--select count(distinct charttime) from prepared_samples
select * from with_anchor_year_group --where anchor_year_group != '2017 - 2019' 