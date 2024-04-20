with person_with_diagnsosis_icd10 AS (
    SELECT 
        b.example_id,
        a.subject_id as person_id, 
        'diagnosis' as feature_name,
        icd10cm as feature_value,
        extract (epoch from b.start_date) as feature_start_date,
        extract (epoch from b.end_date) as feature_end_date
    FROM 
    (    mimiciv_hosp.diagnoses_icd a
    JOIN
        mimiciv_icu.icustays
    USING
        (subject_id, hadm_id)
    JOIN
        "mimiciv_bsi_100_0.5h_test"."__mimiciv_bsi_100_0.5h_cohort" b
    ON 
        (example_id = stay_id)
    )
    LEFT JOIN
        icd9_to_icd10
    ON
        (icd_code = icd9cm)
    WHERE icd_version = 9
    
    UNION ALL

    SELECT 
        b.example_id,
        a.subject_id as person_id, 
        'diagnosis' as feature_name,
        icd_code as feature_value,
        extract (epoch from b.start_date) as feature_start_date,
        extract (epoch from b.end_date) as feature_end_date
    FROM 
        mimiciv_hosp.diagnoses_icd a
    JOIN
        mimiciv_icu.icustays
    USING
        (subject_id, hadm_id)
    JOIN
        "mimiciv_bsi_100_0.5h_test"."__mimiciv_bsi_100_0.5h_cohort" b
    ON 
        (example_id = stay_id)
    WHERE icd_version = 10
), person_with_diagnosis_mimiciv AS (
    SELECT 
        example_id,
        person_id, 
        feature_name,
        'diagnosis -- ' || "Disease" as feature_value,
        feature_start_date as feature_start_date,
        feature_end_date as feature_end_date
    FROM 
        person_with_diagnsosis_icd10
    JOIN
        icd10_pheno
    ON
        (feature_value = "ICD10code")
),  person_with_medical_history_mimiciv AS (
    SELECT 
        b.example_id,
        a.subject_id as person_id, 
        'medical_history' as feature_name,
        'medical_history -- ' || value   as feature_value,
        extract(epoch from b.start_datetime) as feature_start_date,
        extract (epoch from (b.end_datetime - b.start_datetime)) as feature_end_date
    FROM 
        mimiciv_icu.chartevents a
    JOIN
        {cohort_table} b
    ON
        (example_id = stay_id)
    JOIN
        mimiciv_icu.d_items
    USING
        (itemid)
    WHERE
    label in ('Past medical history', 'CV - past medical history')
), relevant_drugs as (
    select 
    drug
    from 
    mimiciv_hosp.prescriptions
    group by
    drug
    order by
    count(*) DESC
    limit 100
), person_with_drug_mimiciv as (
	SELECT
		example_id, 
		person_id,
        'drug' as feature_name,
		'drug -- ' || drug as feature_value,
		extract (epoch from starttime - b.start_date) as feature_start_date,
		LEAST(extract (epoch from b.end_datetime - b.start_date), extract (epoch from stoptime - b.start_date)) as feature_end_date
	FROM 
		mimiciv_hosp.prescriptions
	join
		mimiciv_icu.icustays
	using
		(subject_id, hadm_id)
	join 
		{cohort_table} b
	on
		(example_id = stay_id)
	where 
		drug in (select drug from relevant_drugs) and starttime > b.start_datetime and starttime < b.end_datetime
)

select 
    DISTINCT
    *
FROM
    person_with_diagnosis_mimiciv