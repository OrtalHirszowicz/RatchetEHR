SELECT 
    b.example_id,
    b.person_id as person_id, 
    'gender' as feature_name,
    (CASE 
        WHEN gender = 'Male' THEN 0
        ELSE 1
    END)::text as feature_value,
    Null::timestamp without time zone as feature_start_date
FROM 
    eicu_crd.patient a
JOIN  
    {cohort_table} b
ON
    (patientunitstayid = example_id)