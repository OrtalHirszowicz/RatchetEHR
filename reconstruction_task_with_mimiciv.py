#Reconstruction 
# %% [markdown]
# ## Imports and GPU setup

# %%
#from tkinter import N
from sklearn.model_selection import train_test_split
import Utils.dbutils as dbutils
import Generators.CohortGenerator as CohortGenerator
import Generators.FeatureGenerator as FeatureGenerator
from Utils.model_params_init import get_model_params
import config

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

import os.path
import _pickle as pickle

import warnings
from skorch.exceptions import SkorchWarning
warnings.filterwarnings('ignore', category=SkorchWarning)

from Utils.experiments_utils import ExperimentConducterReconstruction

from Utils.feature_set_info import FeatureSetInfo
from Utils.data_getter import DataGetter
import hyper_params

import math

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14

# %%
NUM_DEVICE = 1
assert(torch.cuda.is_available())
torch.cuda.set_device(NUM_DEVICE)
torch.cuda.empty_cache()
print("Num device: ", NUM_DEVICE)

# %%
#torch.cuda.memory_summary(device=None, abbreviated=False)

# %% [markdown]
# Some parameters:

# %%
SHOULD_USE_CACHE = True
NUM_MESUREMENTS = 100 #600
SHOULD_UPLOAD_SAVED_FEATURESET_INFO = True
SHOULD_DO_GRID_SEARCH = False
embedding_dim = 300 # size of embedding, must be multiple of number of heads
NUM_HOURS_FOR_WINDOW = 2
MODEL_NAME = "ReconstructionTransformerMIMICIV"
#_19 -> 10 features, _20 -> 20 features
NUM_MOST_IMPORTANT_FEATURES = -1 #587 #200 #587 (20 --> 45, 60 --> 74, 10 --> 38)
SHOULD_USE_WEIGHTS = False
NUM_EXPERIMENTS = 1 #2#10
TASK = 'reconstruction_mimiciv_' + str(NUM_MESUREMENTS)
HIDDEN_SIZE = {100: 1, 20: 3, 10: 2}
NUM_ATTENTION_HEADS = HIDDEN_SIZE[NUM_MESUREMENTS]
SHOULD_UPDATE_DATA =  not SHOULD_UPLOAD_SAVED_FEATURESET_INFO
X_val_MAX_SIZE = -1 #100
X_test_MAX_SIZE = -1 #100
SHOULD_USE_CACHE = False
NUM_MESUREMENTS = 100 #00 #100 #600
SHOULD_UPLOAD_SAVED_FEATURESET_INFO = True
SHOULD_DO_GRID_SEARCH = False
embedding_dim = 300 # size of embedding, must be multiple of number of heads
NUM_HOURS_FOR_WINDOW = hyper_params.NUM_HOURS_FOR_WINDOW
MODEL_NAME = "Transformer"
ADDITIONAL_NAME_FOR_EXPERIMENT = hyper_params.CURR_TASK + ("" if hyper_params.CONDUCT_TRANSFER_LEARNING else '_no_transfer') + ("" if not hyper_params.USE_RNN else "_rnn") + ("" if not hyper_params.USE_LSTM else "_lstm") + \
    hyper_params.ADDITIONAL_NAME
NUM_MOST_IMPORTANT_FEATURES = -1 #600 #587
FEATURESET_FILE_NAME = 'featureset_mimiciv_bsi_100_2h_100_2_Transformermimiciv'
LOCATION_WEIGHTS = hyper_params.LOCATION_WEIGHTS


DF_PRECENTAGE = hyper_params.DF_PRECENTAGE 
# %% [markdown]
# ## Cohort, Outcome and Feature Collection
# 
# ### 1. Set up a connection to the OMOP CDM database
# 
# Parameters for connection to be specified in ./config.py

# %%
schema_name = '"' + TASK  + '_test"' # all created tables will be created using this schema
cohort_name = '"' + '__' + TASK + '_cohort"'
reset_schema = True # if true, rebuild all data from scratch

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
db = dbutils.Database(config_path, schema_name, connect_args, schema_name)
if reset_schema:
    db.execute(
        'drop schema if exists {} cascade'.format(schema_name)
    )
db.execute(
    'create schema if not exists {}'.format(schema_name)
)

# %% [markdown]
# ### 2. Generate the Cohort as per the given SQL file

# %%
cohort_script_path = config.SQL_PATH_COHORTS + '/' + hyper_params.CURR_TASK + '_cohort.sql'
database_script_path = config.SQL_PATH_COHORTS + '/BSI_database.sql'

# cohort parameters  
params = {
          'cohort_table_name'     : cohort_name,
          'schema_name'           : schema_name,
          'aux_data_schema'       : config.CDM_AUX_SCHEMA,
          'min_hours_in_ICU'      : 48
         }

cohort = CohortGenerator.Cohort(
    schema_name=schema_name,
    cohort_table_name=cohort_name,
    cohort_generation_script=cohort_script_path,
    cohort_generation_kwargs=params,
    outcome_col_name='y'
)
cohort.use_last_years =True
cohort.build(db, replace=reset_schema)


# %%
# Build the Feature Set by executing SQL queries and reading into tensors
# The tensors are located in featureSet.tensors_for_person. A dictionary where each key is a person_id and each value is 
# the person's tensor.
feature_set_path = config.DEFAULT_SAVE_LOC + '/featureset_' + TASK + '_' + str(NUM_MESUREMENTS) + '_' + str(NUM_HOURS_FOR_WINDOW) \
                    + '_' + MODEL_NAME + 'mimiciv'
cache_data_path = config.DEFAULT_SAVE_LOC + '/cache_data_bsi_test_' + str(NUM_MESUREMENTS) + 'mimiciv'

if SHOULD_UPLOAD_SAVED_FEATURESET_INFO and os.path.isfile(feature_set_path):
    with open(feature_set_path, 'rb') as pickle_file:
        featureSetInfo = pickle.load(pickle_file)
else:
    featureSet = FeatureGenerator.FeatureSet(db, task_name = TASK,
    feature_set_file_name = FEATURESET_FILE_NAME)
    temporal_features_list_observation = [{"name": 'language', "observation_concept_id": 40758030}]
    featureSet.add_default_features(
        [],
        schema_name,
        cohort_name,
        from_sql_file = False,
        type = "Measurement"
    )
    eicu_measurements = ['measurements_mimiciv']
    featureSet.add_default_features(
        eicu_measurements,
        schema_name,
        cohort_name,
        from_sql_file = True,
        type = "Measurement"
    )
    non_temportal_feature_list = ['age_mimiciv',  'gender_mimiciv',  'first_care_unit_mimiciv']
    featureSet.add_default_features(
        non_temportal_feature_list,
        schema_name,
        cohort_name,
        temporal = False,
        type = "General"
    )

    non_temportal_feature_list = ['diagnosis_mimiciv', 'medical_history_mimiciv', 'drug_mimiciv']# 'procedures', 'drug']
    featureSet.add_default_features(
        non_temportal_feature_list,
        schema_name,
        cohort_name,
        temporal = True,
        type = "Diagnosis",
        with_feature_end_date = True
    )
    time_delta = FeatureGenerator.TimeDelta(hours = NUM_HOURS_FOR_WINDOW)#(hours = 2)
    #numeric_names = pd.read_csv("./Tables/mimic_name_to_general.txt", sep = ' -- ')
    #featureSet.numeric_features += list(numeric_names["General"].values)
    # featureSet.postprocess_func = post_process
    featureSet.build(cohort, time_delta = time_delta, from_cached=SHOULD_USE_CACHE, cache_file=cache_data_path,
                    use_prebuilt_features = True)
    featureSet.build_bit_vec_features(cohort, time_delta = time_delta, from_cached=SHOULD_USE_CACHE, cache_file=cache_data_path,
                    use_prebuilt_features = True)              
    featureSetInfo = FeatureSetInfo(featureSet, task_name=TASK)
    with open(feature_set_path, 'wb') as pickle_file:
        pickle.dump(featureSetInfo, pickle_file)

#%%
def get_dict_path(person_id, task_name = TASK):
    return config.DEFAULT_SAVE_LOC + '/Dictionary' + task_name +'/' + str(person_id)



# %%
#featureSetInfo.person_ids.remove('141651.zip')
person_indices  = featureSetInfo.person_ids
person_indices = list(map(int, person_indices))
person_indices = [(0, x) for x in person_indices]
idx_to_person = {i : id for i, id in enumerate(sorted(list(person_indices)))}
person_to_idx = {id : i for i, id in enumerate(sorted(list(person_indices)))}
dict_file = get_dict_path(person_indices[0][1]) + '_postprocessed_tensor'
with open(dict_file, 'rb') as pickle_file:
    curr_tensor_info = pickle.load(pickle_file)
    max_visits = min(curr_tensor_info[0].shape[0], hyper_params.MAX_VISITS)
unique_id = featureSetInfo.unique_id_col
curr_cohort = cohort._cohort[np.isin(cohort._cohort[unique_id].values, list(featureSetInfo.person_ids))]
curr_cohort = curr_cohort.sort_values(by = unique_id)
outcomes_filt = curr_cohort['y']#.values
if 'last_years' in curr_cohort:
    is_last_years = curr_cohort['last_years'].values
n_visits = {p[1]: math.ceil(np.max(np.array(featureSetInfo.window_times_for_person[p[1]]) - featureSetInfo.window_times_for_person[p[1]][0]).astype('timedelta64[h]').astype(int) / NUM_HOURS_FOR_WINDOW) for p in person_indices }



# %%

# split data into train, validate and test sets
validation_precentage = 0.5
test_val_precentage = hyper_params.TEST_VAL_PRECENTAGE

if 1-DF_PRECENTAGE > 0:
    person_indices, _ = train_test_split(sorted(person_indices), test_size = (1-DF_PRECENTAGE), 
    stratify=outcomes_filt)


from Utils.data_utils import get_data
# %%
### load the data from memory into visits_data

visits_data = DataGetter([TASK])

# %%
dataset_dict = {
    'person_indices': person_indices, #The ids of all the persons
    'outcomes_filt': outcomes_filt, # A pandas Series defined such that outcomes_filt.iloc[i] is the outcome of the ith patient
    'idx_to_person': idx_to_person,
    'n_visits': n_visits,
    'visits_data': visits_data,
    'num_invariant_features': featureSetInfo.num_non_numeric_features,
}

not_good_features_file_path = config.DEFAULT_SAVE_LOC + 'not_good_features_file_path_mimiciv'
with open(not_good_features_file_path, 'rb') as pickle_file:
    not_good_features = pickle.load(pickle_file)
    not_good_features += [0, 2, 3, 5, 8, 10, 15, 16, 17, 18, 21, 22, 24, 25, 26, 27, 28, 29, 30, 32, 36, 37, 39, 43, 44, 46, 47, 50, 57, 62, 64, 65, 66, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 84, 87, 95, 96, 97, 99, 100, 101, 102, 103, 104, 105, 106, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404]
    #[0, 2, 3, 8, 10, 15, 16, 17, 18, 21, 22, 24, 25, 26, 27, 28, 29, 30, 32, 36, 37, 39, 43, 44, 46, 47, 50, 57, 62, 64, 65, 66, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 84, 87, 95, 96, 97, 99, 100, 101, 102, 103, 104, 105, 106, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404]
    not_good_features = sorted(list(set(not_good_features)))
    num_numeric_features = len(featureSetInfo.numeric_feature_to_index)
    good_features = list(set(range(num_numeric_features)).difference(not_good_features))
    dataset_dict['not_good_features'] = not_good_features
dataset_dict['is_last_years'] = is_last_years

# %%
### Some default parameters:
ft_epochs = hyper_params.FT_EPOCHS# 200 #200 #400

# %%


# using the same split as before, create train/validate/test batches for the deep model
# `mbsz` might need to be decreased based on the GPU's memory and the number of features being used
mbsz = hyper_params.MBSZ
# Pick a name for the model (mn_prefix) that will be used when saving checkpoints
# Then, set some parameters for SARD. The values below reflect a good starting point that performed well on several tasks
mn_prefix = TASK + '_experiment_prefix'
n_heads = NUM_ATTENTION_HEADS #2
#assert embedding_dim % n_heads == 0

model_params =  get_model_params(embedding_dim, n_heads, featureSetInfo)

# %%
## test_scores = []

experiment_params = {
    'embedding_dim': embedding_dim,
    'num_most_important_features': NUM_MOST_IMPORTANT_FEATURES,
    'dataset_dict': dataset_dict,
    'max_visits': max_visits,
    'lambda_param': hyper_params.LAMBDA_PARAM,
    'lr': hyper_params.LR,#0.0005,
    'weight_decay': hyper_params.WEIGHT_DECAY,
    'ft_epochs': ft_epochs,
    'model_params': model_params,
    'mbsz': mbsz,
    'model_name': MODEL_NAME + ADDITIONAL_NAME_FOR_EXPERIMENT,
    'verbose': True
}

max_score = float('inf')
max_net = None
test_scores = []

for i in range(NUM_EXPERIMENTS):
    import gc

    gc.collect()

    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)
    print("Experiment number ", i)
    X_train, y_train, X_val, y_val, X_test, y_test, dataset_dict = \
        get_data(visits_data, person_indices, dataset_dict, 
        test_val_precentage, validation_precentage, 
            max_visits, n_visits, curr_cohort, fix_imbalance = False, need_to_clean_data = False) #source_visits_data = TODO: Fill it!
    if X_val_MAX_SIZE != -1:
        X_val = X_val[:min(X_val_MAX_SIZE, len(X_val))]
        y_val = y_val[:min(X_val_MAX_SIZE, len(X_val))]
    if X_test_MAX_SIZE != -1:
        X_test = X_test[:min(X_test_MAX_SIZE, len(X_test))]
        y_test = y_test[:min(X_test_MAX_SIZE, len(X_test))]
    experiment_params['X_train'] = X_train
    experiment_params['y_train'] = y_train
    experiment_params['X_val'] = X_val
    experiment_params['y_val'] = y_val
    experiment_params['X_test'] = X_test
    experiment_params['y_test'] = y_test
    experiment_params['dataset_dict'] = dataset_dict
    # experiment_params['max_values_variant'] = dataset_dict['max_values_variant']
    # experiment_params['max_values_invariant'] = dataset_dict['max_values_invariant']
    conducter = ExperimentConducterReconstruction(experiment_params)
    curr_score, transformer_net = conducter.conduct_experiment(i, task_name = TASK, bert_weights=LOCATION_WEIGHTS,
                                                               feature_set_info = featureSetInfo)
    test_scores.append(curr_score)
    if curr_score < max_score:
        max_score = curr_score
        del max_net
        max_net = transformer_net
    else:
        del transformer_net
    del X_train, y_train, X_val, y_val, X_test, y_test, dataset_dict
    gc.collect()
    torch.cuda.empty_cache()
    
    
#Saving the best model parameters:
torch.save(max_net.module.state_dict(), 
           config.DEFAULT_SAVE_LOC + "/SavedModels/" + TASK + '/best_best_model_' +  MODEL_NAME + ADDITIONAL_NAME_FOR_EXPERIMENT) 

# %%
import seaborn as sns
test_scores = pd.DataFrame(test_scores, columns = ['ROC-AUC score'])
test_scores['Experiments'] = range(len(test_scores['ROC-AUC score'].values))

# %%
ax = sns.barplot(x = 'Experiments', y = 'ROC-AUC score', data = pd.DataFrame(test_scores))

# %%
ax = sns.boxplot( y = 'ROC-AUC score', data = pd.DataFrame(test_scores))

# %%
print("Some information on the sets:")
print("Training size: ", len(X_train))
print("Validation size: ", len(X_val))
print("Test size: ", len(X_test))

# %%
print(config.DEFAULT_SAVE_LOC + "/SavedModels/" + TASK + '/best_best_model_' +  MODEL_NAME + ADDITIONAL_NAME_FOR_EXPERIMENT)

# %%


# %%
# import collections

# {i : numeric_feature_to_index[i] for i in sorted(numeric_feature_to_index.keys())}

# %%




# %%
