SELECT 
    example_id,
    person_id,
    feature_name,
    feature_value::text as feature_value,
    feature_start_date::timestamp without time zone
FROM 
    person_with_date