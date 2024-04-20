# %%
#from https://www.nber.org/research/data/icd-9-cm-and-icd-10-cm-and-icd-10-pcs-crosswalk-or-general-equivalence-mappings

import pandas as pd
import dbutils as dbutils
import config

# %%
# database connection parameters
database_name = config.DB_NAME
config_path = 'postgresql://{database_name}'.format(
    database_name = database_name
)
connect_args = {"host": '/var/run/postgresql/', 'user': config.PG_USERNAME, 
                'password': config.PG_PASSWORD, 'database': config.DB_NAME} # connect_args to pass to sqlalchemy create_engine function

# schemas 
cdm_schema_name = config.OMOP_CDM_SCHEMA # the name of the schema housing your OMOP CDM tables


# set up database, reset schemas as needed
schema_name = 'icd9_to_icd10'
reset_schema = False
db = dbutils.Database(config_path, schema_name, connect_args, cdm_schema_name)

data = pd.read_csv("../Tables/icd9toicd10cmgem.csv")
data = data[['icd9cm', 'icd10cm']]
data.to_sql('icd9_to_icd10', db.engine)

