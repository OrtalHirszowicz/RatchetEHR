import sys
sys.path.append('..')

from Utils import dbutils
import config
import pandas as pd

def check_eicu_version(db):
    print("\nChecking eICU Dataset Version:")
    print("---------------------------")
    
    try:
        # Check for version information in metadata
        version_info = db.query("""
            SELECT * 
            FROM information_schema.tables 
            WHERE table_schema = 'eicu_crd'
            AND table_name LIKE '%%version%%'
            OR table_name LIKE '%%meta%%'
            OR table_name LIKE '%%info%%'
        """)
        
        if not version_info.empty:
            print("\nFound version-related tables:")
            print("--------------------------")
            for _, row in version_info.iterrows():
                print(f"- {row['table_name']}")
        else:
            print("No version-related tables found in eicu_crd schema")
        
        # Check data collection period
        collection_period = db.query("""
            SELECT 
                MIN(hospitaladmitoffset) as earliest_admit,
                MAX(hospitaldischargeoffset) as latest_discharge
            FROM eicu_crd.patient
        """)
        print(f"\nData collection period: {collection_period['earliest_admit'][0]} to {collection_period['latest_discharge'][0]}")
        
        # Check number of hospitals
        hospital_count = db.query("""
            SELECT COUNT(DISTINCT hospitalid) as count
            FROM eicu_crd.hospital
        """)
        print(f"Number of hospitals: {hospital_count['count'][0]}")
        
        # Check if we have access to specific tables that indicate version
        tables = db.query("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'eicu_crd'
            ORDER BY table_name
        """)
        
        # Check for specific tables that might indicate version
        has_apache = 'apacheapsvar' in tables['table_name'].values
        has_apachepredvar = 'apachepredvar' in tables['table_name'].values
        has_apachepatientresult = 'apachepatientresult' in tables['table_name'].values
        
        print("\nVersion Indicators:")
        print("-----------------")
        print(f"Has apacheapsvar table: {has_apache}")
        print(f"Has apachepredvar table: {has_apachepredvar}")
        print(f"Has apachepatientresult table: {has_apachepatientresult}")
        
        # Check for specific columns that might indicate version
        try:
            microlab_columns = db.query("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'eicu_crd'
                AND table_name = 'microlab'
            """)
            print("\nMicrolab table columns:")
            print("---------------------")
            for _, row in microlab_columns.iterrows():
                print(f"- {row['column_name']}")
        except Exception as e:
            print(f"Could not check microlab columns: {e}")
            
    except Exception as e:
        print(f"Could not access eICU version information: {e}")

def analyze_cohort_counts():
    # Database connection parameters
    database_name = config.DB_NAME
    config_path = 'postgresql://{database_name}'.format(
        database_name = database_name
    )
    connect_args = {
        "host": '/var/run/postgresql/', 
        'user': config.PG_USERNAME, 
        'password': config.PG_PASSWORD, 
        'database': config.DB_NAME
    }
    
    # Schema configuration
    schema_name = 'cohort_analysis'
    cdm_schema_name = config.OMOP_CDM_SCHEMA
    
    # Initialize database connection
    db = dbutils.Database(config_path, schema_name, connect_args, cdm_schema_name)
    
    # Check eICU version first
    check_eicu_version(db)
    
    print("\nBlood Culture Counts from Different Sources:")
    print("------------------------------------------")
    
    # Count from microlab (current approach)
    microlab_count = db.query("""
        SELECT 
            COUNT(DISTINCT patientunitstayid) as unique_patients,
            COUNT(*) as total_samples
        FROM eicu_crd.microlab
        WHERE culturesite = 'Blood, Central Line' OR culturesite = 'Blood, Venipuncture'
    """)
    print(f"Patients with blood cultures from microlab: {microlab_count['unique_patients'][0]} unique patients, {microlab_count['total_samples'][0]} total samples")
    
    # Count from lab table
    lab_count = db.query("""
        SELECT COUNT(DISTINCT patientunitstayid) as count
        FROM eicu_crd.lab
        WHERE labname LIKE '%%Blood Culture%%'
    """)
    print(f"Patients with blood cultures from lab: {lab_count['count'][0]}")
    
    # Count from diagnosis table
    diagnosis_count = db.query("""
        SELECT COUNT(DISTINCT patientunitstayid) as count
        FROM eicu_crd.diagnosis
        WHERE diagnosisstring LIKE '%%sepsis%%' 
           OR diagnosisstring LIKE '%%bacteremia%%'
           OR diagnosisstring LIKE '%%blood infection%%'
    """)
    print(f"Patients with infection-related diagnoses: {diagnosis_count['count'][0]}")
    
    # Count overlap between microlab and lab
    overlap_count = db.query("""
        SELECT COUNT(DISTINCT m.patientunitstayid) as count
        FROM eicu_crd.microlab m
        JOIN eicu_crd.lab l USING(patientunitstayid)
        WHERE (m.culturesite = 'Blood, Central Line' OR m.culturesite = 'Blood, Venipuncture')
        AND l.labname LIKE '%%Blood Culture%%'
    """)
    print(f"Patients with blood cultures in both microlab and lab: {overlap_count['count'][0]}")
    
    # Show blood culture sites with both patient and sample counts
    culture_sites = db.query("""
        SELECT 
            culturesite,
            COUNT(DISTINCT patientunitstayid) as unique_patients,
            COUNT(*) as total_samples
        FROM eicu_crd.microlab
        WHERE culturesite = 'Blood, Central Line' OR culturesite = 'Blood, Venipuncture'
        GROUP BY culturesite
        ORDER BY total_samples DESC
    """)
    print("\nBlood Culture Sites from microlab:")
    print("--------------------------------")
    for _, row in culture_sites.iterrows():
        print(f"{row['culturesite']}: {row['unique_patients']} unique patients, {row['total_samples']} total samples")
    
    # Show average number of blood cultures per patient
    avg_cultures = db.query("""
        WITH patient_cultures AS (
            SELECT patientunitstayid, COUNT(*) as num_cultures
            FROM eicu_crd.microlab
            WHERE culturesite = 'Blood, Central Line' OR culturesite = 'Blood, Venipuncture'
            GROUP BY patientunitstayid
        )
        SELECT 
            AVG(num_cultures) as avg_cultures,
            MIN(num_cultures) as min_cultures,
            MAX(num_cultures) as max_cultures
        FROM patient_cultures
    """)
    print(f"\nAverage blood cultures per patient: {avg_cultures['avg_cultures'][0]:.2f}")
    print(f"Minimum blood cultures per patient: {avg_cultures['min_cultures'][0]}")
    print(f"Maximum blood cultures per patient: {avg_cultures['max_cultures'][0]}")
    
    # Show unique lab names related to blood cultures
    lab_names = db.query("""
        SELECT DISTINCT labname, COUNT(*) as count
        FROM eicu_crd.lab
        WHERE labname LIKE '%%Blood Culture%%'
        GROUP BY labname
        ORDER BY count DESC
    """)
    print("\nBlood Culture Lab Names:")
    print("----------------------")
    for _, row in lab_names.iterrows():
        print(f"{row['labname']}: {row['count']} samples")
    
    print("\nTotal Patients in eICU:")
    print("----------------------")
    
    # Count total patients
    total_patients = db.query("""
        SELECT COUNT(DISTINCT patientunitstayid) as total_patients
        FROM eicu_crd.patient
    """)
    print(f"Total unique patients in eICU: {total_patients['total_patients'][0]}")
    
    # Count patients with ICU stay >= 48 hours
    long_stay_patients = db.query("""
        SELECT COUNT(DISTINCT patientunitstayid) as long_stay_patients
        FROM eicu_crd.patient
        WHERE (hospitaldischargeoffset - hospitaladmitoffset) >= (48*60)
    """)
    print(f"Patients with ICU stay >= 48 hours: {long_stay_patients['long_stay_patients'][0]}")
    
    # Count patients with all three conditions (matching original cohort exactly)
    all_conditions = db.query("""
        WITH severalepi AS (
            SELECT DISTINCT m.patientunitstayid, p.hospitalid
            FROM eicu_crd.microlab m
            JOIN eicu_crd.patient p USING(patientunitstayid)
            WHERE (m.culturesite = 'Blood, Central Line' OR m.culturesite = 'Blood, Venipuncture')
            AND (p.hospitaldischargeoffset - p.hospitaladmitoffset) >= (48*60)
        ),
        hospital_info AS (
            SELECT hospitalid, COUNT(DISTINCT patientunitstayid) as count
            FROM severalepi
            GROUP BY hospitalid
        ),
        good_hospitals AS (
            SELECT hospitalid
            FROM hospital_info
            WHERE count > 50
        )
        SELECT COUNT(DISTINCT patientunitstayid) as all_conditions
        FROM severalepi
        WHERE hospitalid IN (SELECT hospitalid FROM good_hospitals)
    """)
    print(f"\nPatients with blood cultures, ICU stay >= 48 hours, and from hospitals with >50 qualifying patients: {all_conditions['all_conditions'][0]}")
    
    # Count patients by hospital (matching original cohort)
    hospital_counts = db.query("""
        WITH severalepi AS (
            SELECT DISTINCT m.patientunitstayid, p.hospitalid
            FROM eicu_crd.microlab m
            JOIN eicu_crd.patient p USING(patientunitstayid)
            WHERE (m.culturesite = 'Blood, Central Line' OR m.culturesite = 'Blood, Venipuncture')
            AND (p.hospitaldischargeoffset - p.hospitaladmitoffset) >= (48*60)
        )
        SELECT hospitalid, COUNT(DISTINCT patientunitstayid) as patient_count
        FROM severalepi
        GROUP BY hospitalid
        ORDER BY patient_count DESC
    """)
    
    print("\nHospital Distribution (based on patients with blood cultures and ICU stay >= 48 hours):")
    print("--------------------------------------------------------------------------------")
    print(f"Total number of hospitals: {len(hospital_counts)}")
    print(f"Average qualifying patients per hospital: {hospital_counts['patient_count'].mean():.2f}")
    print(f"Median qualifying patients per hospital: {hospital_counts['patient_count'].median():.2f}")
    print(f"Min qualifying patients in a hospital: {hospital_counts['patient_count'].min()}")
    print(f"Max qualifying patients in a hospital: {hospital_counts['patient_count'].max()}")
    
    # Count hospitals with >50 qualifying patients
    large_hospitals = len(hospital_counts[hospital_counts['patient_count'] > 50])
    print(f"Number of hospitals with >50 qualifying patients: {large_hospitals}")

if __name__ == "__main__":
    analyze_cohort_counts() 