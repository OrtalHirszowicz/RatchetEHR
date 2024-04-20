select 
    example_id,
    person_id,
    'gender' as feature_name,
    (CASE
    WHEN gender = 'M' then 0
    else 1
    end)::text as feature_value,
    NULL::timestamp without time zone as feature_start_date
    FROM
    {cohort_table} b
    join 
    mimiciv_hosp.patients
    on
    (subject_id = person_id)