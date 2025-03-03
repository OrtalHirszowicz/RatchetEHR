
# Database Setup
DB_NAME = 'mimic'
PG_USERNAME = 'postgres'
PG_PASSWORD = 'postgres'

# Schemas
OMOP_CDM_SCHEMA = 'omop' # schema holding standard OMOP tables
CDM_AUX_SCHEMA = 'mimiciii' # schema to hold auxilliary tables not tied to a particular schema
CDM_VERSION = 'v5.x.x' # set to 'v5.x.x' if on v5

# SQL Paths
SQL_PATH_COHORTS = 'sql/Cohorts' # path to SQL scripts that generate cohorts
SQL_PATH_FEATURES = 'sql/Features' # path to SQL scripts that generate features

# Cache
DEFAULT_SAVE_LOC = '/bigdata/omerg/RatchetEHR/tmp/tmp' # where to save temp files
#DEFAULT_SAVE_LOC = 'tmp'
TASK = "bsi"
