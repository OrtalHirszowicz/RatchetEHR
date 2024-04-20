
# %%

#from https://github.com/spiros/chronological-map-phenotypes

import re
from os import listdir
from os.path import join
import dbutils as dbutils
import config
import pandas as pd


map_pheno_path = r'../chronological-map-phenotypes/secondary_care'
pattern = re.compile('ICD_.*')
relevant_files = [f for f in listdir(map_pheno_path) if pattern.match(f)]
#print(relevant_files)
# %%
df = None
for file in relevant_files:
    curr_df = pd.read_csv(join(map_pheno_path, file))
    if df is None:
        df = curr_df
        continue
    df = pd.concat([df, curr_df])

df['ICD10code'] = df['ICD10code'].map(lambda x: x.replace('.', ''))
# %%
# from IPython.display import display
# display(df)
# %%
database_name = config.DB_NAME
config_path = 'postgresql://{database_name}'.format(
    database_name = database_name
)
connect_args = {"host": '/var/run/postgresql/', 'user': config.PG_USERNAME, 
                'password': config.PG_PASSWORD, 'database': config.DB_NAME} # connect_args to pass to sqlalchemy create_engine function

# schemas 
cdm_schema_name = config.OMOP_CDM_SCHEMA # the name of the schema housing your OMOP CDM tables


# set up database, reset schemas as needed
schema_name = 'public'
reset_schema = False
db = dbutils.Database(config_path, schema_name, connect_args, cdm_schema_name)

df.to_sql('icd10_pheno', db.engine)


#%%