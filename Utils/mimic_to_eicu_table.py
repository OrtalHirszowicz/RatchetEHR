# %%
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
schema_name = 'mimic_to_eicu_converter'
reset_schema = True
db = dbutils.Database(config_path, schema_name, connect_args, cdm_schema_name)

#%%
data = pd.read_csv("../Tables/measurements_used.txt", sep = ' -- ')
data['MIMIC-III'] = data['MIMIC-III'].astype(str)
data_2 = pd.read_csv("../Tables/mimic_name_to_general.txt", sep = ' -- ')
data_2['MIMIC-III'] = data_2['MIMIC-III'].astype(str)
data = data.set_index('MIMIC-III').join(data_2.set_index('MIMIC-III'))
data.to_sql('mimic_to_eicu_converter', db.engine)

# schema_name = 'medical_hist_converter'
# reset_schema = False
# db = dbutils.Database(config_path, schema_name, connect_args, cdm_schema_name)

# data = pd.read_csv("../Tables/medical_hist_converter.csv", sep = ',')
# data.to_sql('medical_hist_converter', db.engine)

# schema_name = 'drug_converter'
# reset_schema = False
# db = dbutils.Database(config_path, schema_name, connect_args, cdm_schema_name)

# data = pd.read_csv("../Tables/drug_converter.csv", sep = ' -- ')
# data.to_sql('drug_converter', db.engine)

# %%
