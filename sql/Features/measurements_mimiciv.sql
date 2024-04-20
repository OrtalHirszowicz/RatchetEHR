with measurments_raw as (
	SELECT
		example_id, 
		person_id,
        label as feature_name,
		valuenum::text as feature_value,
		charttime::timestamp without time zone as feature_start_date
		--valueuom as unit
	FROM 
		mimiciv_hosp.labevents
	join
		mimiciv_icu.icustays
	using
		(subject_id, hadm_id)
	join
		{cohort_table} b
	on
		(example_id = stay_id)
    JOIN
    mimiciv_hosp.d_labitems
    USING
        (itemid)
	where 
		charttime > b.start_datetime and charttime < b.end_datetime

    UNION ALL

    SELECT
		example_id, 
		person_id,
        result_name as feature_name,
		result_value::text as feature_value,
		chartdate::timestamp without time zone as feature_start_date
		--valueuom as unit
	FROM 
		mimiciv_hosp.omr
	join
		{cohort_table} b
	on
        (subject_id = person_id)
	where 
		chartdate > b.start_datetime and chartdate < b.end_datetime and result_name in ('Weight (Lbs)', 'Height (Inches)')

    UNION ALL

    SELECT
		example_id, 
		person_id,
        label as feature_name,
		valuenum::text as feature_value,
		charttime::timestamp without time zone as feature_start_date
		--valueuom as unit
	FROM 
		mimiciv_icu.chartevents
	join
		{cohort_table} b
	on
		(example_id = stay_id)
    JOIN
    mimiciv_icu.d_items
    USING
        (itemid)
	where 
		charttime > b.start_datetime and charttime < b.end_datetime

), 
relevant_feature_names as (
    select 
    feature_name, 
    count (feature_name)
    from 
    measurments_raw
    group by 
    feature_name
    order by
    count
    desc
    LIMIT 100
),
measurements_mimiciv as (
    select 
    example_id, 
	person_id,
    feature_name || '-- Numeric' as feature_name,
	feature_value,
	feature_start_date	
    from
    measurments_raw
    where 
    feature_name in (select feature_name from relevant_feature_names) or feature_name in 
    ('Temperature', 'C-Reactive Protein', 'Hematocrit', 'Hemoglobin', 'Potassium', 
    'Sodium', 'pH', 'pO2', 'PT', 'INR(PT)', 'WBC', 'WBC Count', 'Lactate Dehydrogenase (LD)', 
    'Lactate', 'RDW', 'RBC', 'Asparate Aminotransferase (AST)', 'MCHC', 'Bilirubin', 
    'Neutrophils', 'Albumin', 'Creatinine', 'Hematocrit', 'Alkaline Phosphatase', 'MCV', 
    'Alanine Aminotransferase (ALT)', 'Lymphocytes', 'Urea Nitrogen')
), person_with_date AS (
SELECT 
    b.example_id,
    a.subject_id as person_id, 
    'age' as feature_name,
    anchor_age as feature_value,
    Null as feature_start_date
FROM 
    mimiciv_hosp.patients a
JOIN  
    {cohort_table} b
ON
    (person_id = subject_id)
)

select distinct * from measurements_mimiciv