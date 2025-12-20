WITH blood_cultures AS (
            SELECT 
                patientunitstayid,
                (date '2000-1-1' + (culturetakenoffset * interval '1 minutes')) as culture_date,
                culturesite
            FROM eicu_crd.microlab m
            JOIN eicu_crd.patient p USING(patientunitstayid)
            WHERE (m.culturesite = 'Blood, Central Line' OR m.culturesite = 'Blood, Venipuncture')
            AND (p.unitDischargeOffset - p.hospitaladmitoffset) >= (48*60)
            AND culturetakenoffset >= 0
        ),
        lab_res as (
            SELECT
                b.example_id,
                b.person_id,
                labname as feature_name,
                labresult as feature_value,
                (date '2000-1-1' + (labresultoffset * interval  '1 minutes')) as feature_start_date,
                COALESCE(labmeasurenameinterface, labMeasureNameSystem) as unit
            FROM 
                eicu_crd.lab as a
            JOIN
                {cohort_table} as b
            ON 
                (example_id = patientunitstayid)
            JOIN
                blood_cultures bc
            ON
                (bc.patientunitstayid = b.example_id)
            WHERE
                labname IS NOT NULL
                AND labresultoffset >= 0
                AND (date '2000-1-1' + (labresultoffset * interval  '1 minutes')) < bc.culture_date
        ),
        nurse_res as (
            SELECT
                b.example_id,
                b.person_id,
                nursingchartcelltypevalname as feature_name,
                CASE WHEN nursingchartvalue~E'^[0-9]+(\\.[0-9]+)?$' THEN nursingchartvalue::float ELSE NULL end as feature_value,
                (date '2000-1-1' + (nursingChartOffset * interval  '1 minutes')) as feature_start_date,
                nursingchartcelltypevallabel as unit
            FROM 
                eicu_crd.nursecharting as a
            JOIN
                {cohort_table} as b
            ON 
                (example_id = patientunitstayid)
            JOIN
                blood_cultures bc
            ON
                (bc.patientunitstayid = b.example_id)
            WHERE
                nursingchartcelltypevalname IS NOT NULL
                AND nursingChartOffset >= 0
                AND (date '2000-1-1' + (nursingChartOffset * interval  '1 minutes')) < bc.culture_date
        ),
        resp_res as (
            SELECT
                b.example_id,
                b.person_id,
                respchartvaluelabel as feature_name,
                CASE WHEN respchartvalue~E'^[0-9]+(\\.[0-9]+)?$' THEN respchartvalue::float ELSE NULL end as feature_value,
                (date '2000-1-1' + (respChartOffset * interval  '1 minutes')) as feature_start_date,
                respcharttypecat as unit
            FROM 
                eicu_crd.respiratorycharting as a
            JOIN
                {cohort_table} as b
            ON 
                (example_id = patientunitstayid)
            JOIN
                blood_cultures bc
            ON
                (bc.patientunitstayid = b.example_id)
            WHERE
                respchartvaluelabel IS NOT NULL
                AND respChartOffset >= 0
                AND (date '2000-1-1' + (respChartOffset * interval  '1 minutes')) < bc.culture_date
        ),
        labothername_res as (
            SELECT
                b.example_id,
                b.person_id,
                labothername as feature_name,
                CASE WHEN labotherresult~E'^[0-9]+(\\.[0-9]+)?$' THEN labotherresult::float ELSE NULL end as feature_value,
                (date '2000-1-1' + (labotheroffset * interval  '1 minutes')) as feature_start_date,
                '' as unit
            FROM 
                eicu_crd.customlab as a
            JOIN
                {cohort_table} as b
            ON 
                (example_id = patientunitstayid)
            JOIN
                blood_cultures bc
            ON
                (bc.patientunitstayid = b.example_id)
            WHERE
                labothername IS NOT NULL
                AND labotheroffset >= 0
                AND (date '2000-1-1' + (labotheroffset * interval  '1 minutes')) < bc.culture_date
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
            SELECT * FROM lab_res
            UNION ALL
            SELECT * FROM nurse_res
            UNION ALL
            SELECT * FROM resp_res
            UNION ALL
            SELECT * FROM labothername_res
	    -- UNION ALL
            -- SELECT * FROM person_with_date
        ), 
        converted as (
            select 
                example_id,
                person_id,
                feature_name,
                feature_value,
                feature_start_date,
                unit
            FROM
                measurements
            WHERE
                feature_value IS NOT NULL
        )
        select 
            example_id,
            person_id,
            feature_name,
            feature_value::TEXT as feature_value,
            feature_start_date,
            unit
        FROM
            converted