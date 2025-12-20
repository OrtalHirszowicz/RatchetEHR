SELECT 
    b.example_id,
    b.person_id as person_id, 
    'first_care_unit_MICU' as feature_name,
    (CASE 
        WHEN unittype = 'MICU' THEN 1
        ELSE 0
    END)::text as feature_value,
    Null::timestamp without time zone as feature_start_date,
    '' as unit
FROM 
    eicu_crd.patient a
JOIN  
    {cohort_table} b
ON
    (patientunitstayid = example_id)

UNION ALL

SELECT 
    b.example_id,
    b.person_id as person_id, 
    'first_care_unit_SICU' as feature_name,
    (CASE 
        WHEN unittype = 'SICU' THEN 1
        ELSE 0
    END)::text as feature_value,
    Null::timestamp without time zone as feature_start_date,
    '' as unit
FROM 
    eicu_crd.patient a
JOIN  
    {cohort_table} b
ON
    (patientunitstayid = example_id)

UNION ALL

SELECT 
    b.example_id,
    b.person_id as person_id, 
    'first_care_unit_TSICU' as feature_name,
    (CASE 
        WHEN unittype = 'Med-Surg ICU' THEN 1
        ELSE 0
    END)::text as feature_value,
    Null::timestamp without time zone as feature_start_date,
    '' as unit
FROM 
    eicu_crd.patient a
JOIN  
    {cohort_table} b
ON
    (patientunitstayid = example_id)