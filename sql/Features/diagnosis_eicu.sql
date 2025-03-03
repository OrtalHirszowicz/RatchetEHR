with "diagnosis_with_icd" as (
SELECT 
    b.example_id,
    patienthealthsystemstayid as person_id, 
    'diagnosis' as feature_name,
    string_to_array("icd9code", ',') as feature_value,
    diagnosisoffset * 60 as feature_start_date,
    (CASE 
        WHEN activeupondischarge = 'True' THEN unitdischargeoffset * 60 
        ELSE diagnosisoffset * 60
        END) as feature_end_date
FROM 
    eicu_crd.diagnosis a
JOIN
    eicu_crd.patient
USING
    (patientunitstayid)
JOIN  
    {cohort_table} b
ON
    (patientunitstayid = example_id)
WHERE 
icd9code != '' AND diagnosisoffset >= 0
),
"icd9_codes" as (
select 
    example_id,
    person_id,
    feature_name,
    replace(feature_value[1], '.', '')::text as feature_value,
    feature_start_date,
    feature_end_date
from
diagnosis_with_icd
),
"diagnosis_eicu" as (
    SELECT
    example_id,
    person_id,
    feature_name,
    'diagnosis -- ' || "Disease" as feature_value,
    feature_start_date,
    feature_end_date
    FROM
    icd9_codes
    JOIN
    public.icd9_to_icd10
    ON
    (feature_value = icd9cm)
    JOIN
    public.icd10_pheno
    ON
    ("ICD10code" = icd10cm)
),
"medical_history_eicu" as (
    SELECT 
    b.example_id,
    b.person_id as person_id, 
    'medical_history' as feature_name,
    'medical_history -- ' ||  " feature_value_mimic"  as feature_value,
    extract(epoch from b.start_date) as feature_start_date,
    extract(epoch from b.end_date) as feature_end_date
    FROM 
        eicu_crd.pasthistory a
    JOIN  
        {cohort_table} b
    ON
        (patientunitstayid = example_id)
    JOIN
        public.medical_hist_converter
    ON
        (pasthistoryvalue = feature_value_eicu)
), 
"drug_eicu" as (
    SELECT 
    b.example_id,
    b.person_id as person_id, 
    'drug' as feature_name,
    'drug -- ' || mimic as feature_value,
    infusionoffset * 60 as feature_start_date,
    infusionoffset * 60 feature_end_date
    FROM 
        eicu_crd.infusiondrug a
    JOIN  
        {cohort_table} b
    ON
        (patientunitstayid = example_id)
    JOIN
        public.drug_converter
    ON
        (drugname = eicu)
    WHERE 
        infusionoffset >= 0
)


SELECT
DISTINCT
*
FROM 
diagnosis_eicu