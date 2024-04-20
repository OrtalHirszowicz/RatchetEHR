#feature generator
import sys
sys.path.append('..')

import time
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import itertools
import os
from itertools import chain
from os import listdir
from os.path import isfile, join
import pickle
import gc

import config 

SHOULD_DEBUG = True
SHOULD_CREATE_FILES_1 = True

class Feature():

    def __init__(
        self,
        feature_name,
        feature_sql_file,
        feature_sql_params,
        from_sql_file = True,
        temporal=True, 
        type = "",
        with_feature_end_date = False,
    ):
        self.feature_name = feature_name
        self.is_temporal = temporal
        self.with_feature_end_date = with_feature_end_date
        
        self.params = feature_sql_params
        if from_sql_file:
            with open(feature_sql_file, 'r') as f:
                raw_sql = f.read()
            self._sql_raw = raw_sql
            
        self._feature_sql_file = feature_sql_file
        
        self.from_sql_file = from_sql_file

        self.type = type
        


    def __str__(self):
        return "Temporal feature extracted from {}".format(
            self._feature_sql_file
        )
    
class TimeDelta():
    def __init__(
        self,
        hours = 0,
        minutes = 0,
        seconds = 0,
    ):
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds

    def build_deltatime(self):
        return timedelta(hours=self.hours, minutes=self.minutes, seconds=self.seconds)

    def convert_to_seconds(self):
        return self.seconds + 60 * self.minutes + 60 * 60 * self.hours

class FeatureSet():
    
    def __init__(
        self,
        db,
        task_name, 
        #Columns for the features (each feature should have the following columns)
        id_col = 'person_id',
        time_col = 'feature_start_date',
        end_time_col = 'feature_end_date',
        feature_name_col = 'feature_name',
        feature_value_col = 'feature_value',
        unique_id_col = 'example_id', 
        feature_set_file_name = None
        ):
        
        self.feature_set_file_name = feature_set_file_name
        self._db = db

        self.end_date_posfix = "_with_end_date"
        
        self.id_col = id_col
        self.time_col = time_col
        self.end_time_col = end_time_col
        self.feature_name_col = feature_name_col
        self.feature_value_col = feature_value_col
        self.unique_id_col = unique_id_col
        
        self._temporal_features = []
        self._nontemporal_features = []
        
        self._temporal_feature_names = []
        self._temporal_feature_names_set = set()
        
        self._nontemporal_feature_names = []
        self._nontemporal_feature_names_set = set()
        

        self.postprocess_func = None

        self.numeric_features = []
        self.non_numeric_features = []

        self.task_name = task_name

    def add(self, feature):
        if feature.is_temporal:
            self._temporal_features.append(feature)
        else:
            self._nontemporal_features.append(feature)
        if feature.type == "Measurement":
            self.numeric_features.append(feature.feature_name)
        if not feature.is_temporal:
            self.non_numeric_features.append(feature.feature_name)

    def add_default_features(self, default_features, schema_name=None, cohort_name=None, temporal=True, database_name = None, from_sql_file = True, type = "", with_feature_end_date = False):
        fns = [
            (f, './sql/Features/{}.sql'.format(f))
            for f in default_features
        ]
        for feature_name, fn in fns:
            feature = Feature(
                feature_name,
                fn,
                {
                    'cdm_schema':config.OMOP_CDM_SCHEMA,
                    'cohort_table':'{}.{}'.format(
                        schema_name,
                        cohort_name
                    ),
                    'database_table' : '{}.{}'.format(
                        schema_name,
                        database_name
                    )
                },
                temporal=temporal,
                from_sql_file = from_sql_file, 
                type = type,
                with_feature_end_date=with_feature_end_date
            )
            self.add(feature)

            
    def get_feature_names(self):
        return self._temporal_feature_names +  self._nontemporal_feature_names

    def get_num_features(self):
        return len (
            self._temporal_feature_names +  self._nontemporal_feature_names
        )

    def generate_measurement_features(self, cohort, feature_names):
        raw_sql = \
        '''
            select
                b.example_id,
                a.person_id,
                c."General" as feature_name,
                CASE
                    WHEN lower(unit_source_value) = 'mg/l' THEN (a.value_as_number * 0.1)::text
                    WHEN lower(unit_source_value) = 'g/dl' THEN (a.value_as_number * 100)::text
                    WHEN lower(unit_source_value) = 'mmol/l' THEN (a.value_as_number * 18)::text
                    WHEN lower(unit_source_value) = 'meq/l' THEN (a.value_as_number * 18)::text
                    WHEN lower(unit_source_value) = 'deg. f' THEN (a.value_as_number / 33.8)::text
                    ELSE a.value_as_number::text 
                END as feature_value,
                measurement_datetime as feature_start_date
            from 
                {cdm_schema}.measurement as a
            inner join
                {cohort_table} as b
            using 
                (person_id)
            inner JOIN
                mimic_to_eicu_converter AS c  
            ON 
                (measurement_source_value = "MIMIC-III")
            where
                unit_source_value not in ('#/hpf') and measurement_datetime is not NULL and value_as_number is not NULL and measurement_source_value in ({feature_names})
                and measurement_datetime BETWEEN b.start_date AND b.end_date '''.format(
                cdm_schema = config.OMOP_CDM_SCHEMA,
                cohort_table='{}.{}'.format(
                    cohort._schema_name,
                    cohort._cohort_table_name
                ),
                feature_names = ', '.join(["'{x}'".format(x = x) for x in feature_names])
        )
        return [raw_sql]
        
    def generate_observation_features(self, params, cohort):
        raw_sql = \
        '''
        select
            b.example_id,
            a.person_id,
            '{feature_name}' as feature_name,
            a.value_as_string as feature_value,
            observation_datetime as feature_start_date
        from 
            {cdm_schema}.observation as a
        inner join
            {cohort_table} as b
        using 
            (person_id)
        where
            observation_datetime is not NULL and value_as_string is not NULL and observation_concept_id = {observation_concept_id} and observation_datetime BETWEEN b.start_date AND b.end_date
        '''.format (
            cdm_schema = config.OMOP_CDM_SCHEMA,
            cohort_table = '{}.{}'.format(
                cohort._schema_name,
                cohort._cohort_table_name
            ), 
            feature_name = params['name'],
            observation_concept_id = params['observation_concept_id']
        )
        # if SHOULD_DEBUG:
        #     print(raw_sql)
        return [raw_sql]
    

    def store_features_measurements(self, cohort, from_cached, cache_file):
        if not from_cached:
            measurement_features = [f.feature_name for f in self._temporal_features if f.from_sql_file == False and f.type == "Measurement"]
            raw_sqls = []
            if len(measurement_features) > 0:
                raw_sqls = [self.generate_measurement_features(cohort, measurement_features)]
            raw_sqls += [self.generate_observation_features(f.feature_name, cohort) 
                for f in self._temporal_features if f.from_sql_file == False and f.type == "Observation"]
            if (len(raw_sqls) + len([1 for f in self._temporal_features + self._nontemporal_features if f.from_sql_file == True and not f.with_feature_end_date])== 0):
                return
            joined_sql =  "{} order by {} asc".format(
                " union all ".join(
                        #Getting the information about a single feature
                        [f._sql_raw.format(
                            cdm_schema=config.OMOP_CDM_SCHEMA,
                            cohort_table='{}.{}'.format(
                                cohort._schema_name,
                                cohort._cohort_table_name
                            ),
                            database_table='{}.{}'.format(
                                cohort._schema_name,
                                cohort._database_table_name
                            )
                        )
                        for f in self._temporal_features + self._nontemporal_features if f.from_sql_file == True and not f.with_feature_end_date] +
                        list(itertools.chain.from_iterable(raw_sqls))
                ),
                ",".join([self.id_col, self.unique_id_col,      ## Order by unique_id
                        self.time_col, self.feature_name_col])    
            )
            print(joined_sql)
            # if SHOULD_DEBUG:
            #     print(joined_sql)
            copy_sql = """
                copy 
                    ({query})
                to 
                    stdout 
                with 
                    csv {head}
            """.format(
                query=joined_sql,
                head="HEADER"
            )
            t = time.time()
            conn = self._db.engine.raw_connection()
            cur = conn.cursor()
            store = open(cache_file,'wb')
            cur.copy_expert(copy_sql, store)
            store.seek(0)
            print('Data loaded to buffer in {0:.2f} seconds'.format(
                time.time()-t
            ))
                
        #Saving some unique values for each column by the following order: concept_name, feature_start_date and example_id
        
    def store_features_with_end_date(self, cohort, from_cached, cache_file):
        if not from_cached:
            joined_sql =  "{} order by {} asc".format(
                " union all ".join(
                        #Getting the information about a single feature
                        [f._sql_raw.format(
                            cdm_schema=config.OMOP_CDM_SCHEMA,
                            cohort_table='{}.{}'.format(
                                cohort._schema_name,
                                cohort._cohort_table_name
                            ),
                            database_table='{}.{}'.format(
                                cohort._schema_name,
                                cohort._database_table_name
                            )
                        )
                        for f in self._temporal_features + self._nontemporal_features if f.from_sql_file == True and f.with_feature_end_date]
                ),
                ",".join([self.id_col, self.unique_id_col,      ## Order by unique_id
                        self.time_col, self.end_time_col, self.feature_name_col])    
            )
            print(joined_sql)
            copy_sql = """
                copy 
                    ({query})
                to 
                    stdout 
                with 
                    csv {head}
            """.format(
                query=joined_sql,
                head="HEADER"
            )
            t = time.time()
            conn = self._db.engine.raw_connection()
            cur = conn.cursor()
            store = open(cache_file + self.end_date_posfix,'wb')
            cur.copy_expert(copy_sql, store)
            store.seek(0)
            print('Data loaded to buffer in {0:.2f} seconds'.format(
                time.time()-t
            ))

    def parse_date(self, str_date):
        return datetime.strptime(str_date, '%Y-%m-%d %H:%M:%S')

    def get_dict_path(self, person_id, post_fix = ""):
        return config.DEFAULT_SAVE_LOC + '/Dictionary' + self.task_name +'/' + str(person_id) + post_fix

    def get_temp_dict(self):
        feature_set_path = config.DEFAULT_SAVE_LOC + '/' + self.feature_set_file_name
        with open(feature_set_path, 'rb') as pickle_file:
            featureSetInfo = pickle.load(pickle_file)
        return {val: key for key, val in featureSetInfo.numeric_feature_to_index.items()}

    def get_temp_dict_codes(self):
        feature_set_path = config.DEFAULT_SAVE_LOC + '/' + self.feature_set_file_name
        with open(feature_set_path, 'rb') as pickle_file:
            featureSetInfo = pickle.load(pickle_file)
        return {val: key for key, val in featureSetInfo.feature_codes_to_id.items()}
    
    def fill_na (self, chunk):
        chunk[['example_id', 'person_id']].fillna(0, inplace = True)
        chunk[['feature_name']].fillna("", inplace = True)
        chunk[['feature_value']].fillna('0', inplace = True)
        return chunk.dropna(subset = ['feature_start_date'])
        #return chunk

    def build_bit_vec_features (self, cohort, time_delta : TimeDelta, cache_file='/tmp/store.csv', nontemporal_cache_file='/tmp/store_ntmp.csv', 
            from_cached=False, features_to_generate_without_file = [], use_prebuilt_features = False):
        gc.collect()
        dict_path = config.DEFAULT_SAVE_LOC + '/Dictionary' + self.task_name +'/' 
        self.persons_ids = [f for f in listdir(dict_path) if isfile(join(dict_path, f))]
        self.person_ids = self.persons_ids
        if len([f for f in self._temporal_features + self._nontemporal_features if f.from_sql_file == True and f.with_feature_end_date]) == 0:
            return
        self.store_features_with_end_date(cohort, from_cached, cache_file)
        store = open(cache_file + self.end_date_posfix ,'rb')

        chunksize = int(2e6) 
        col_to_dtype = {'example_id' : int  , 'person_id' : int, 'feature_name' : str , 'feature_value' : str, 'feature_start_date': int, 'feature_end_date': int}
        max_value = 0
        time_window = time_delta.convert_to_seconds()
        feature_codes = set()
        if not use_prebuilt_features:
            for chunk in pd.read_csv(store, chunksize=chunksize, dtype= col_to_dtype):
                feature_codes = feature_codes.union(set(chunk[self.feature_value_col]))
            self.feature_codes_to_id = {code: i for i, code in enumerate(sorted(list(feature_codes)))}

        if use_prebuilt_features:
            self.feature_codes_to_id = {code: i for i, code in self.get_temp_dict_codes().items()}

        store = open(cache_file + self.end_date_posfix ,'rb')
        for chunk in pd.read_csv(store, chunksize=chunksize, dtype= col_to_dtype):
            chunk[self.feature_value_col] = chunk[self.feature_value_col].map(self.feature_codes_to_id)
            curr_max_value = np.max(chunk[self.feature_value_col]) + 1
            if curr_max_value > max_value:
                max_value = curr_max_value

        def get_tensor_shape(person_id, num_vals):
            dict_file = self.get_dict_path(person_id)
            with open(dict_file, 'rb') as pickle_file:
                try:
                    val = pickle.load(pickle_file)
                except:
                    pass
            return torch.zeros(size = (val[0].shape[0], num_vals))

        if use_prebuilt_features:
            max_value = len(self.feature_codes_to_id)

        print("Building BOW")
        one_hot_tensors = {person_id: get_tensor_shape(person_id, max_value) for person_id in self.person_ids}
        store = open(cache_file + self.end_date_posfix ,'rb')
        person_ids = set(self.person_ids)
        for chunk in pd.read_csv(store, chunksize=chunksize, dtype= col_to_dtype):
            chunk[self.feature_value_col] = chunk[self.feature_value_col].map(self.feature_codes_to_id).fillna(-1).astype('int32')
            curr_ids = set(chunk[self.unique_id_col])
            curr_ids = curr_ids.intersection(person_ids)
            chunk[self.time_col] = (chunk[self.time_col] // time_window).astype(int)
            chunk[self.end_time_col] = (chunk[self.end_time_col] // time_window).astype(int)
            for p_id in curr_ids:
                curr_chunk = chunk[chunk[self.unique_id_col] == p_id]
                curr_tensor = one_hot_tensors[p_id]
                for index, row in curr_chunk.iterrows():
                    if row[self.feature_value_col] == -1:
                        continue
                    curr_tensor[row[self.time_col] : row[self.end_time_col], row[self.feature_value_col]] = 1
                one_hot_tensors[p_id] = curr_tensor
        
        for p_id in one_hot_tensors:
            dict_file = self.get_dict_path(p_id)
            with open(dict_file, 'rb') as pickle_file:
                val = pickle.load(pickle_file)
            with open(dict_file, 'wb') as pickle_file:
                pickle.dump((val[0], val[1], one_hot_tensors[p_id]), pickle_file)

        

    def build(self, cohort, time_delta : TimeDelta, cache_file='/tmp/store.csv', nontemporal_cache_file='/tmp/store_ntmp.csv', 
            from_cached=False, features_to_generate_without_file = [], use_prebuilt_features = False):

        self.store_features_measurements(cohort, from_cached, cache_file)
        store = open(cache_file,'rb')

        t = time.time()

        store = open(cache_file, 'rb')
        chunksize = int(2e5) 
        self.numeric_feature_names = set()
        self.feature_names = set()
        person_ids = set()
        window_initial_time_for_person = {}

        time_window = time_delta.build_deltatime()

        curr_id = 0
        curr_id_numeric = 0
        counter = 0
        col_to_dtype = {'example_id' : int  , 'person_id' : int, 'feature_name' : str , 'feature_value' : str}
        #Getting information about persons and different features
        for chunk in pd.read_csv(store, chunksize=chunksize, dtype= col_to_dtype):
            chunk_without_null = self.fill_na(chunk)
            curr_person_ids = set(chunk_without_null[self.unique_id_col].values)
            person_ids = (person_ids.union(curr_person_ids))
            person_ids_iterator = curr_person_ids.difference(window_initial_time_for_person.keys())
            window_initial_time_for_person.update({person_id: 
                                                self.parse_date(chunk_without_null[chunk_without_null[self.unique_id_col] == person_id]['feature_start_date'].values[0]) 
                            for person_id in person_ids_iterator})
                    
            feature_names = set(chunk['feature_name'].values)
            if not use_prebuilt_features:
                curr_numeric_features = (feature_names).intersection(self.numeric_features)
                curr_numeric_features = curr_numeric_features.union(
                    set([f for f in feature_names if "Numeric" in f]))
                self.numeric_feature_names = self.numeric_feature_names.union(curr_numeric_features)
    
            curr_non_numeric_features = [f for f in feature_names if "Numeric" not in f]
            self.feature_names = self.feature_names.union(curr_non_numeric_features)
            counter += 1
            del curr_non_numeric_features
            
        gc.collect()

        print("Counter = ", counter)
        print(len(person_ids))
        self.feature_to_index = {val: i for i, val in enumerate(sorted(list(self.feature_names)))}
        self.index_to_feature = {i: val for val, i in self.feature_to_index.items()}
        self.num_non_numeric_features = len(self.index_to_feature)
        if not use_prebuilt_features:
            self.numeric_feature_to_index = {val: i for i, val in enumerate(sorted(list(self.numeric_feature_names)))}
            self.index_to_numeric_feature = {i: val for val, i in self.numeric_feature_to_index.items()}
            self.num_numeric_features = len(self.numeric_feature_to_index)
        else:
            self.index_to_numeric_feature = {i: val for i, val in self.get_temp_dict().items()}
            self.numeric_feature_to_index =  {val: key for key, val in self.index_to_numeric_feature.items()}
            #Getting the maximal index and adding 1.
            self.num_numeric_features = max(self.numeric_feature_to_index.values()) + 1

            
        #Adding information for the tensors
        new_store = open(cache_file, 'rb')


        self.window_times_for_person = {}
        self.person_ids = set()

        def get_window_info (window_values):
            window_values["numeric_features_ids"] = window_values["feature_name"].replace(self.numeric_feature_to_index).fillna(-1)
            numeric_window_values = window_values[window_values["numeric_features_ids"] != -1]
            window_info = {(feature_id, True): subdf["feature_value"].values for feature_id, subdf in numeric_window_values.groupby("numeric_features_ids")}
            return window_info

        def convert_intervals (x):
            curr_intervals = set(x['intervals'])
            x['intervals'].replace({y: i for i, y in enumerate(sorted(list(curr_intervals)))}, inplace = True)
            return x

    
        if SHOULD_DEBUG:
            print('Four: {0:.2f} seconds'.format(
                time.time()-t
            ))

        def save_info(x):
            key_variant = x[0]
            val_variant = x[1]['Time variant info']
            val_invariant = x[1]['Time invariant info']
            dict_file = self.get_dict_path(key_variant)
            if os.path.isfile(dict_file):
                with open(dict_file, 'rb') as pickle_file:
                    curr_val = pickle.load(pickle_file)
                    curr_val_variant = curr_val[0]
                    curr_val_invariant = curr_val[1]
                    curr_val_variant += val_variant
                    curr_val_invariant.update(val_invariant)
                    val_invariant = curr_val_invariant
                    val_variant = curr_val_variant
            with open(dict_file, 'wb') as pickle_file:
                pickle.dump((val_variant, val_invariant), pickle_file)

        def get_time_invariant_info(person_id, chunk):
            relevant_chunk = chunk[chunk[self.unique_id_col] == person_id]
            return {(self.feature_to_index[row['feature_name']], False): int(float(row['feature_value'])) for _, row in 
                relevant_chunk.iterrows()}
            
        chunksize = int(2e5)

        for chunk in pd.read_csv(new_store, chunksize = chunksize, dtype= col_to_dtype):
            chunk_with_date = self.fill_na(chunk)
            persons_ids = set(chunk_with_date[self.unique_id_col].values)
            relevant_window_initial_time_for_person = dict(filter(lambda elem: elem[0] in persons_ids, window_initial_time_for_person.items()))
            chunk_with_date['feature_start_date_normalized'] = chunk_with_date['feature_start_date'].map(self.parse_date) - \
                                        chunk_with_date[self.unique_id_col].replace(relevant_window_initial_time_for_person)
            chunk_with_date['intervals'] = chunk_with_date['feature_start_date_normalized'] // time_window
            chunk_with_date.groupby(self.unique_id_col).apply(convert_intervals).reset_index(level=[0], drop=True)
            groups_by_person_id = {name: val.groupby('intervals') for name, val in chunk_with_date.groupby(self.unique_id_col)}
            persons_in_dict = persons_ids.intersection(self.window_times_for_person.keys())
            self.window_times_for_person.update({person_id: self.window_times_for_person[person_id] + \
                                                list(groups_by_person_id[person_id].first()['feature_start_date_normalized'].values) 
                                            for person_id in persons_in_dict})
            persons_not_in_dict = persons_ids.difference(self.window_times_for_person.keys())
            self.window_times_for_person.update({person_id: 
                                            list(groups_by_person_id[person_id].first()['feature_start_date_normalized'].values)
                                            for person_id in persons_not_in_dict})
            if SHOULD_CREATE_FILES_1:
                window_groups = chunk_with_date.groupby([self.unique_id_col, 'intervals'])
                window_groups_new = {person_id: [] for (person_id, _), _ in window_groups}
                [window_groups_new[curr_person_id].append((interval, curr_df))
                for ((curr_person_id, interval), curr_df) in window_groups]
                window_groups = window_groups_new
                sorted_window_groups = {person_id: sorted(window_groups[person_id], key = lambda x: x[0]) for 
                                    person_id in window_groups.keys()}
                new_tensors_info_for_person = {person_id: { 'Time variant info':
                                            [get_window_info(group) for _, group in sorted_window_groups[person_id]] }for person_id in sorted_window_groups.keys()}
                chunk_with_date['numeric_features_ids'] = chunk_with_date["feature_name"].map(self.numeric_feature_to_index).fillna(-1)
                chunk_with_date = chunk_with_date[chunk_with_date["numeric_features_ids"] != -1]
                window_groups = chunk_with_date.groupby([self.unique_id_col, 'numeric_features_ids', 'intervals'], sort = True)
                new_tensors_info_for_person = {person_id: {'Time variant info': [{} for i in range((curr_num + 1))] } for person_id, curr_num in chunk_with_date.groupby([self.unique_id_col])['intervals'].max().iteritems()}
                for (person_id, feature_id, interval), subdf in window_groups: 
                    new_tensors_info_for_person[person_id]['Time variant info'][interval][(feature_id, True)] = np.median(np.asarray(subdf["feature_value"].values).astype(float))
                gc.collect()
                [info.update(
                        {'Time invariant info': get_time_invariant_info(person_id, chunk[chunk['feature_start_date'].isnull()])}) for person_id, info in 
                        new_tensors_info_for_person.items()]
                [save_info(x) for x in new_tensors_info_for_person.items()]
                del window_groups, new_tensors_info_for_person, chunk_with_date,
            del persons_ids, groups_by_person_id, persons_not_in_dict
            gc.collect()

        
        if SHOULD_DEBUG: 
            print('Six: {0:.2f} seconds'.format(
                    time.time()-t
                ))
            

        dict_path = config.DEFAULT_SAVE_LOC + '/Dictionary' + self.task_name +'/' 
        self.persons_ids = [f for f in listdir(dict_path) if isfile(join(dict_path, f))]

        print('Seven: {0:.2f} seconds'.format(
            time.time()-t
            ))      
        
        EMBEDDING_SIZE = 300
        def create_sparse_tensor(person_id):
            dict_file = self.get_dict_path(person_id)
            with open(dict_file, 'rb') as pickle_file:
                tensor_info = pickle.load(pickle_file)
            curr_tensor_info = tensor_info[0]
            num_windows = len(curr_tensor_info)
            curr_tensor_info_ordered = [list(x.items()) for x in curr_tensor_info]
            rows = np.repeat(range(len(curr_tensor_info)), [len(x) for x in curr_tensor_info])
            curr_tensor_info_ordered = list(chain.from_iterable(curr_tensor_info_ordered))
            cols = np.array([int(feature_idx_and_is_numeric[0]) for feature_idx_and_is_numeric, val in curr_tensor_info_ordered])
            values = np.array([val for feature_idx_and_is_numeric, val in curr_tensor_info_ordered])
            is_numeric = np.array([feature_idx_and_is_numeric[1] for feature_idx_and_is_numeric, val in curr_tensor_info_ordered])
            if (len(is_numeric) == 0):
                with open(dict_file, 'wb') as pickle_file:
                    self.person_ids_to_remove.append(int(person_id))
                    pickle.dump(None, pickle_file)
                return
            curr_tensor_numeric = torch.sparse_coo_tensor(size = (num_windows, self.num_numeric_features), 
                                                        indices = [rows[is_numeric], cols[is_numeric]], 
                                                        values = (np.array(values[is_numeric], dtype = "float32")))
            non_numeric_values = []
            if (len(non_numeric_values) == 0):
                curr_tensor = torch.empty(size = (num_windows, self.num_non_numeric_features, EMBEDDING_SIZE), 
                                        layout=torch.sparse_coo)
            else:
                curr_tensor = torch.sparse_coo_tensor(size = (num_windows, self.num_non_numeric_features, EMBEDDING_SIZE), 
                                                            indices = [np.repeat(rows[~is_numeric], EMBEDDING_SIZE), np.repeat(cols[~is_numeric], EMBEDDING_SIZE), list(range(EMBEDDING_SIZE))*len(rows[~is_numeric]) ], 
                                                            values = non_numeric_values)
            curr_tensor = [(k[0], v) for k, v in tensor_info[1].items()]
            curr_tensor = torch.sparse_coo_tensor(size = (self.num_non_numeric_features, ), indices = np.expand_dims(np.array([x[0] for x in curr_tensor]), axis = 0), values=[x[1] for x in curr_tensor])
            with open(dict_file, 'wb') as pickle_file:
                pickle.dump((curr_tensor_numeric, curr_tensor), pickle_file)
            del curr_tensor, non_numeric_values, curr_tensor_numeric, is_numeric, values, cols, rows, dict_file, tensor_info, curr_tensor_info, num_windows, curr_tensor_info_ordered, 
            gc.collect()

        #
        if SHOULD_CREATE_FILES_1:
            self.person_ids_to_remove = []
            [create_sparse_tensor(p) for p in self.persons_ids]
            self.persons_ids = [person_id for person_id in self.persons_ids 
                                 if not (person_id in self.person_ids_to_remove)]

        #self.person_ids = self.persons_ids

        print('Tensors generated in {0:.2f} seconds'.format(
                        time.time()-t
                    ))

        