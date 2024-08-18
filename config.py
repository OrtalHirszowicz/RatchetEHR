
# Database Setup
DB_NAME = #to fill - database name for postgres
PG_USERNAME = #to fill - username for postgres
PG_PASSWORD = #to fill - password name for postgres

# Schemas
OMOP_CDM_SCHEMA = 'omop' # schema holding standard OMOP tables
CDM_AUX_SCHEMA = 'mimiciii' # schema to hold auxilliary tables not tied to a particular schema
CDM_VERSION = 'v5.x.x' # set to 'v5.x.x' if on v5

# SQL Paths
SQL_PATH_COHORTS = # path to SQL scripts that generate cohorts
SQL_PATH_FEATURES = # path to SQL scripts that generate features

# Cache
DEFAULT_SAVE_LOC = # where to save temp files
TASK = "bsi"
