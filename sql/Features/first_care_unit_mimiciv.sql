SELECT
		example_id, 
		person_id,
        'first_care_unit_MICU' as feature_name,
		(case 
        when first_careunit = '	Medical/Surgical Intensive Care Unit (MICU/SICU)' or first_careunit = 'Medical Intensive Care Unit (MICU)' then 1
        else 0
        end)::text as feature_value,
		NULL::timestamp without time zone as feature_start_date
		--valueuom as unit
	FROM 
		mimiciv_icu.icustays
	join 
		{cohort_table} b
	on
		(example_id = stay_id)

    UNION ALL

    SELECT
		example_id, 
		person_id,
        'first_care_unit_SICU' as feature_name,
		(case 
        when first_careunit = '	Medical/Surgical Intensive Care Unit (MICU/SICU)' or first_careunit = 'Surgical Intensive Care Unit (SICU)' then 1
        else 0
        end)::text as feature_value,
		NULL::timestamp without time zone as feature_start_date
		--valueuom as unit
	FROM 
		mimiciv_icu.icustays
	join 
		{cohort_table} b
	on
		(example_id = stay_id)

    UNION ALL

    SELECT
		example_id, 
		person_id,
        'first_care_unit_TSICU' as feature_name,
		(case 
        when first_careunit = 'Trauma SICU (TSICU)' then 1
        else 0
        end)::text as feature_value,
		NULL::timestamp without time zone as feature_start_date
		--valueuom as unit
	FROM 
		mimiciv_icu.icustays
	join 
		{cohort_table} b
	on
		(example_id = stay_id)
