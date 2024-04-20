import sys
sys.path.append('..')

from Utils.dbutils import Database
from jinja2 import Template
import pandas as pd
import datetime as dt
import time
import config

class Cohort(object): 
    def __init__(
        self,
        schema_name='',
        database_table_name=None,
        cohort_table_name=None,
        cohort_generation_script=None,
        database_generation_script = None,
        database_generation_kwargs = None,
        cohort_generation_kwargs=None,
        first=None,
        verbose=True,
        outcome_col_name='y'
    ):  

        self._cohort = []
        self._built = False
        self._first = first
        
        self._schema_name = schema_name
        self._cohort_table_name = cohort_table_name
        self._database_table_name = database_table_name
        self._cohort_generation_script = cohort_generation_script
        self._cohort_generation_kwargs = cohort_generation_kwargs 
        self._database_generation_kwargs = database_generation_kwargs
        self._database_generation_script = database_generation_script
        self._outcome_col = outcome_col_name
        self.cohort_generation_sql = None
        self.use_last_years = False
        
        self._dtype = {}
        self._dtype['example_id'] = int
        self._dtype['person_id'] = int
        self._dtype['person_source_value'] = str
        self._dtype['start_date'] = str
        self._dtype['end_date'] = str


    def get_num_examples(self):  # noqa
        return len(self._cohort)

    def is_built(self):  # noqa
        return self._built

    def build(self, db, replace=False, use_dataset = False):  # noqa
        if use_dataset and (replace or self._database_table_name not in db.get_all_tables(
                schema=self._schema_name).values):
            t = time.time()
            if replace:
                print('Regenerating Table (replace=True)')
            else:
                 print(
                    'Table not found in schema {}, regenerating'.format(
                    self._schema_name
                    )
                )
                    
            with open(self._database_generation_script, 'r') as f:
                database_generation_sql_raw = f.read()
                if self._database_generation_kwargs is not None:
                    self.database_generation_sql = database_generation_sql_raw.format(
                            **dict(self._database_generation_kwargs, **{'cdm_schema':config.OMOP_CDM_SCHEMA})
                        )
                else:
                    self.cohort_generation_sql = database_generation_sql_raw
                        
                db.build_table('{}.{}'.format(
                    self._schema_name,
                    self._database_table_name
                ), self.database_generation_sql)
                    #print(self.cohort_generation_sql)
                print('Regenerated Database in {} seconds'.format(
                    time.time() - t
                ))
        else:
            print('Database already exists, set replace=True to rebuild')

        if replace or self._cohort_table_name not in db.get_all_tables(
                schema=self._schema_name
            ).values:
                t = time.time()
                if replace:
                    print('Regenerating Table (replace=True)')
                else:
                    print(
                        'Table not found in schema {}, regenerating'.format(
                            self._schema_name
                        )
                    )
                
                with open(self._cohort_generation_script, 'r') as f:
                    cohort_generation_sql_raw = f.read()
                if self._cohort_generation_kwargs is not None:
                    self.cohort_generation_sql = cohort_generation_sql_raw.format(
                        **dict(self._cohort_generation_kwargs, **{'cdm_schema':config.OMOP_CDM_SCHEMA})
                    )
                else:
                    self.cohort_generation_sql = cohort_generation_sql_raw
                    
                db.build_table('{}.{}'.format(
                    self._schema_name,
                    self._cohort_table_name
                ), self.cohort_generation_sql)
                #print(self.cohort_generation_sql)
                print('Regenerated Cohort in {} seconds'.format(
                    time.time() - t
                ))
        else:
            print('Table already exists, set replace=True to rebuild')
        
        # If $first is set, then get only $first members of the cohort
        if self._first is not None:
            new_table_name = '{}.{}___first_{}'.format(
                self._schema_name,
                self._cohort_table_name,
                self._first
            )

            sql = """
                create table {new_table}
                as (
                    select * from {cohort_table}
                    order by example_id
                    limit {first}
                )
            """.format(
                    new_table=new_table_name,
                    cohort_table=self.table_name,
                    first=self._first
                )
            
            db.build_table(new_table_name, sql)

            self.table_name = new_table_name

        if not self._built:
            
            cohort_table = '{schema}.{table}'.format(
                schema=self._schema_name,
                table=self._cohort_table_name
            )
            
            col_names = [
                'example_id',
                'person_id',
                'start_date',
                'end_date'
            ] + [self._outcome_col] + (['last_years'] if self.use_last_years else [])
            
            sql = """
                select {columns}
                from {table}
            """.format(
                columns=','.join(col_names),
                table=cohort_table
            )
            
            self._cohort = db.query(sql)
            
            for date_col in ['start_date', 'end_date']:
                self._cohort[date_col] = pd.to_datetime(self._cohort[date_col])
            self._cohort = self._cohort.astype(
                {k:v for k,v in self._dtype.items() if k in self._cohort.columns}
            )
            self._end_date = pd.to_datetime(max(self._cohort[date_col]))

        db.meta.reflect()
        self._built = True
