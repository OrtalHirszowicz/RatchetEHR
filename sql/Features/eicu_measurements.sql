with lab_res as (
    SELECT
        b.example_id,
        b.person_id,
        "General" as feature_name,
        labresult as feature_value,
        (date '2000-1-1' + (labresultoffset * interval  '1 minutes')) as feature_start_date,
        labMeasureNameSystem as unit
    FROM 
        eicu_crd.lab as a
    JOIN
        {cohort_table} as b
    ON 
        (example_id = patientunitstayid)
    JOIN
        public.mimic_to_eicu_converter
    ON
        ("eiCU-lab"= labname)
    WHERE
        "eiCU-table" = 'lab'

),
nurse_res as (
    SELECT
        b.example_id,
        b.person_id,
        "General" as feature_name,
        CASE WHEN nursingchartvalue~E'^\\d+$' THEN nursingchartvalue::float ELSE 0 end as feature_value,
        (date '2000-1-1' + (nursingchartentryoffset * interval  '1 minutes')) as feature_start_date,
        '' as unit

    FROM 
        eicu_crd.nursecharting as a
    JOIN
        {cohort_table} as b
    ON 
        (example_id = patientunitstayid)
    JOIN
        public.mimic_to_eicu_converter
    ON
        ("eiCU-lab"= nursingchartcelltypevalname)
    WHERE
        "eiCU-table" = 'nursingcharting'

),
resp_res as (
    SELECT
        b.example_id,
        b.person_id,
        "General" as feature_name,
        CASE WHEN respchartvalue~E'^\\d+$' THEN respchartvalue::float ELSE 0 end as feature_value,
        (date '2000-1-1' + (respchartentryoffset * interval  '1 minutes')) as feature_start_date,
        '' as unit

    FROM 
        eicu_crd.respiratorycharting as a
    JOIN
        {cohort_table} as b
    ON 
        (example_id = patientunitstayid)
    JOIN
        public.mimic_to_eicu_converter
    ON
        ("eiCU-lab"= respchartvaluelabel)
    WHERE
        "eiCU-table" = 'respchart'

),
labothername_res as (
    SELECT
        b.example_id,
        b.person_id,
        "General" as feature_name,
        CASE WHEN labotherresult~E'^\\d+$' THEN labotherresult::float ELSE 0 end as feature_value,
        (date '2000-1-1' + (labotheroffset * interval  '1 minutes')) as feature_start_date,
        '' as unit
    FROM 
        eicu_crd.customlab as a
    JOIN
        {cohort_table} as b
    ON 
        (example_id = patientunitstayid)
    JOIN
        public.mimic_to_eicu_converter
    ON
        ("eiCU-lab"= labothername)
    WHERE
        "eiCU-table" = 'labothername'

),
person_with_date AS (
SELECT 
    b.example_id,
    a.patienthealthsystemstayid as person_id, 
    'age' as feature_name,
    age::text as feature_value,
    Null as feature_start_date,
    '' as unit
FROM 
    eicu_crd.patient a
JOIN  
    {cohort_table} b
ON
    (example_id = patientunitstayid)
WHERE
    age ~ '^[0-9\.]+$' --Where age is a number
),
measurements as (
        SELECT
            *
        FROM
            lab_res
    UNION ALL
        SELECT
            *
        FROM
            nurse_res
    UNION ALL
        SELECT
            *
        FROM
            resp_res
    UNION ALL
        SELECT
            *
        FROM
            labothername_res
), 
converted as (
    select 
        example_id,
        person_id,
        feature_name,
        -- CASE
        --     WHEN lower(unit) = 'mg/l' THEN (feature_value * 0.1)
        --     WHEN lower(unit) = 'g/dl' THEN (feature_value * 100)
        --     WHEN lower(unit) = 'mmol/l' THEN (feature_value * 18)
        --     WHEN lower(unit) = 'meq/l' THEN (feature_value * 18)
        --     WHEN lower(unit) = 'deg. f' THEN (feature_value / 33.8)
        --     ELSE feature_value
        -- END as 
        feature_value,
        feature_start_date,
        unit
    FROM
        measurements
)
select 
    example_id,
    person_id,
    feature_name,
    feature_value::TEXT as feature_value,
    feature_start_date
FROM
    converted
