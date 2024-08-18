# %%
# %%
# %%
# %%
import sys
from pathlib import Path
from os.path import dirname, abspath
d = dirname(abspath(Path.cwd()))
sys.path.insert(0, d)

# %%
# %%
# %%
# %%
#bsi_tl_recon*
# %% [markdown]
# ## Imports and GPU setup

print("Random Forest")

# %%
from copy import deepcopy
from Utils.SkorchUtils.datasets import MyDatasetSingle
from sklearn.model_selection import train_test_split
import Utils.dbutils as dbutils
from Utils.feature_set_info import FeatureSetInfo
import Generators.CohortGenerator as CohortGenerator
import Generators.FeatureGenerator as FeatureGenerator
import config

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

import os.path
import pickle

from Utils.experiments_utils import update_experiment_results

import warnings
from skorch.exceptions import SkorchWarning
warnings.filterwarnings('ignore', category=SkorchWarning)

from Utils.data_utils import get_data
import config

from Utils.experiments_utils import ExperimentConducterTransferLearning

from Utils.data_getter import DataGetter
from Utils.model_params_init import get_model_params

from collections import OrderedDict

# from torch.optim import Optimizer
import seaborn as sns
import gc

import hyper_params

import math

import math
from torch.utils.data import DataLoader

import random

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 13

# %%
assert(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(hyper_params.DEVICE_NUM)

print("Folder Num: ", hyper_params.ADDITIONAL_NAME)


#%% 
# %% [markdown]
# Some parameters:

# %%
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
SHOULD_USE_WEIGHTS = False
NUM_EXPERIMENTS = hyper_params.NUM_EXPERIMENTS
TASK = 'mimiciv_' + hyper_params.CURR_TASK + '_' + str(NUM_MESUREMENTS) + '_' +str(NUM_HOURS_FOR_WINDOW) +'h'
CONVERT_TWO = {10: '_19', 20: '_3_20', 100: '_100_2_1', 200: '_100_1', 1:'_1_1'}
MORTALITY_TRANSFORMER_INIT_WEGITHS_LOCATION = None
FEATURESET_FILE_NAME = None
HIDDEN_SIZE = {100: 1, 20: 1, 10: 2, 1:1}
NUM_ATTENTION_HEADS = HIDDEN_SIZE[NUM_MESUREMENTS]
SHOULD_UPDATE_DATA = not SHOULD_UPLOAD_SAVED_FEATURESET_INFO

PERFORM_RANDOM_SEARCH = False

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
reset_schema = False # if true, rebuild all data from scratch

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
db = dbutils.Database(config_path, schema_name, connect_args, cdm_schema_name)
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
cohort_script_path = config.SQL_PATH_COHORTS + '/' + hyper_params.CURR_TASK + '_mimiciv_cohort.sql'

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
                    use_prebuilt_features = False)
    featureSet.build_bit_vec_features(cohort, time_delta = time_delta, from_cached=SHOULD_USE_CACHE, cache_file=cache_data_path,
                    use_prebuilt_features = False)              
    featureSetInfo = FeatureSetInfo(featureSet, task_name=TASK)
    with open(feature_set_path, 'wb') as pickle_file:
        pickle.dump(featureSetInfo, pickle_file)



# %% [markdown]
# ### 4. Process the collected data and calculate indices needed for the deep model

# %%
def get_dict_path(person_id):
    return config.DEFAULT_SAVE_LOC + '/Dictionary' + TASK +'/' + str(person_id)

# %%
person_indices  = featureSetInfo.person_ids
orig_person_indices = list(map(int, person_indices))
unique_id = featureSetInfo.unique_id_col
person_indices = set(orig_person_indices).intersection(set(cohort._cohort[unique_id].values))
person_indices = [(0, x) for x in person_indices]
idx_to_person = {i : id for i, id in enumerate(sorted(list(person_indices)))}
person_to_idx = {id : i for i, id in enumerate(sorted(list(person_indices)))}
#visits_data = featureSetInfo.post_process_transformer
dict_file = get_dict_path(person_indices[0][1]) + '_postprocessed_tensor'
with open(dict_file, 'rb') as pickle_file:
    curr_tensor_info = pickle.load(pickle_file)
    max_visits = curr_tensor_info[0].shape[0]
n_visits = {int(p): math.ceil(np.max(np.array(featureSetInfo.window_times_for_person[int(p)])).astype('timedelta64[h]').astype(int) / NUM_HOURS_FOR_WINDOW) for p in featureSetInfo.person_ids }
if hyper_params.USE_INIT_DATA:
    person_indices = [(0, p) for p, val in n_visits.items() if val > (hyper_params.NUM_HOURS_PREDICTION // NUM_HOURS_FOR_WINDOW)]
person_indices = sorted(person_indices, key = lambda x: x[1])
curr_cohort = cohort._cohort[np.isin(cohort._cohort[unique_id].values, [x[1] for x in person_indices])]
curr_cohort = curr_cohort.sort_values(by = unique_id)
outcomes_filt = curr_cohort['y'].values
if hyper_params.USE_TEST_GROUP:
    is_last_years = curr_cohort['last_years'].values
# %%
one_label_precentage = np.sum(outcomes_filt) / len(outcomes_filt)
print("Precentage of 1 label: ", one_label_precentage)
print("For now uses Weighted random sampler")


# %%
if SHOULD_UPDATE_DATA:
    def update_data(person_id):
        dict_file = get_dict_path(person_id) + "_postprocessed_tensor"
        with open(dict_file, 'rb') as pickle_file:
            val = pickle.load(pickle_file)
        dict_file = get_dict_path(person_id) + '_transformer'
        with open(dict_file, 'wb') as pickle_file:
            pickle.dump((val[0], val[1].to_dense(), val[2]), pickle_file)
        
    [update_data(person_id[1]) for person_id in person_indices]

# %%
### load the data from memory into visits_data


# def get_data_transformer(person_id):
#     dict_file = get_dict_path(person_id) + '_postprocessed_tensor'
#     with open(dict_file, 'rb') as pickle_file:
#         val = pickle.load(pickle_file)
#     return (val[0].to_dense(), val[1], val[2])

# source_visits_data = OrderedDict({person_id: get_data_transformer(person_id) for person_id in sorted(person_indices)})

#TODO: Ortal - add this

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
    num_numeric_features = len(featureSetInfo.numeric_feature_to_index)
    good_features = list(set(range(num_numeric_features)).difference(not_good_features))
    dataset_dict['not_good_features'] = not_good_features

# %%
#Filtering out samples without enough data:
if hyper_params.CURR_TASK != 'mortality':
    total_n_visits = deepcopy(n_visits)
    dataset = MyDatasetSingle(hyper_params.MAX_VISITS, total_n_visits, dataset_dict['visits_data'], TASK, person_indices, y = None,
    mbsz = hyper_params.MBSZ, dataset_dict = dataset_dict, feature_set_info=featureSetInfo, should_mask_input= False,
    if_clean_data= False)
    features_info_counter = None
    num_numeric_features = len(featureSetInfo.numeric_feature_to_index)
    info_precentages = {}


    for x in DataLoader(dataset = dataset, batch_size = hyper_params.MBSZ, pin_memory=True, num_workers=hyper_params.NUM_WORKERS):
        curr_person_indices = x[0][0][1].tolist()
        curr_info = x[0][1]
        for i in range(curr_info.shape[0]):
            curr_person_idx = curr_person_indices[i]
            curr_info_person = curr_info[i, :total_n_visits[curr_person_idx], :]
            if curr_info_person.shape[0] * curr_info_person.shape[1] != 0: 
                info_precentages[curr_person_idx] = torch.sum(1 * (curr_info_person != 0)).item() / (curr_info_person.shape[0] * curr_info_person.shape[1])
            else:
                info_precentages[curr_person_idx] = 0
    dataset_dict['info_precentages'] = info_precentages
    dataset_dict['weight'] = dataset_dict['info_precentages']
    not_relevant_person_indices = [p_id for p_id, p in info_precentages.items() if p <= hyper_params.STAY_INFO_PRECENTAGE_LOW_THRESHOLD]
    not_relevant_person_indices_eICU = set([(0, p) for p in not_relevant_person_indices]).intersection(set(person_indices))
    # not_relevant_person_indices_val_test = [p_id for p_id, p in info_precentages.items() if p < hyper_params.STAY_INFO_PRECENTAGE_LOW_THRESHOLD_VAL_TEST]
    # not_relevant_person_indices_val_test = set([(0, p) for p in not_relevant_person_indices_val_test]).intersection(set(person_indices))
    #not_relevant_person_indices_eICU = not_relevant_person_indices_eICU.union(not_relevant_person_indices_val_test)
    not_relevant_person_indices_eICU = [person_indices.index(p) for p in not_relevant_person_indices_eICU]
    person_indices = [(a, b) for a, b in list(np.delete(np.array(person_indices), not_relevant_person_indices_eICU, axis = 0))]
    outcomes_filt = list(np.delete(np.array(outcomes_filt), not_relevant_person_indices_eICU))
    dataset_dict['person_indices'] = person_indices
    dataset_dict['outcomes_filt'] = outcomes_filt
    if hyper_params.USE_TEST_GROUP:
        is_last_years = list(np.delete(np.array(is_last_years), not_relevant_person_indices_eICU))
        dataset_dict['is_last_years'] = is_last_years

# %%
one_label_precentage = np.sum(outcomes_filt) / len(outcomes_filt)
print("Precentage of 1 label after cleaning: ", one_label_precentage)
print("For now uses Weighted random sampler")


# %%
# split data into train, validate and test sets
validation_precentage = 0.5
test_val_precentage = hyper_params.TEST_VAL_PRECENTAGE
if 1-DF_PRECENTAGE > 0:
    person_indices, _ = train_test_split(sorted(dataset_dict['person_indices'], key = lambda x: x[1]), test_size = (1-DF_PRECENTAGE), 
    stratify=outcomes_filt)
    dataset_dict['person_indices'] =  person_indices
# %%
### Some default parameters:
ft_epochs = hyper_params.FT_EPOCHS #508
#ft_epochs = ft_epochs * int ((1-test_val_precentage) * len(person_indices)) #800 #* 2 #400
#print(ft_epochs)

# %%


# using the same split as before, create train/validate/test batches for the deep model
# `mbsz` might need to be decreased based on the GPU's memory and the number of features being used
mbsz = hyper_params.MBSZ #64 #64 * 2
# Pick a name for the model (mn_prefix) that will be used when saving checkpoints
# Then, set some parameters for SARD. The values below reflect a good starting point that performed well on several tasks
mn_prefix = 'bsi_experiment_prefix'
n_heads = NUM_ATTENTION_HEADS #2
#assert embedding_dim % n_heads == 0

print(len(featureSetInfo.numeric_feature_to_index))


# %%
X_train, y_train, X_val, y_val, X_test, y_test, new_dataset_dict = \
        get_data(visits_data, dataset_dict['person_indices'], dataset_dict, 
        test_val_precentage, validation_precentage, 
            max_visits, n_visits, curr_cohort, fix_imbalance = False, need_to_clean_data = False, featureSetInfo = featureSetInfo)



# %%
TAKE_MEDIAN = True
TAKE_LAST = False

total_n_visits = deepcopy(n_visits)

    

# %%
def build_dataset(X, y, is_train = False):
    dataset = MyDatasetSingle(max_visits, total_n_visits, dataset_dict['visits_data'], TASK, X, y, 
                clf = None, mbsz = mbsz, dataset_dict = dataset_dict, feature_set_info=featureSetInfo, 
                should_mask_input=False)
    num_samples = 128

    dataset = DataLoader(dataset, pin_memory = True, batch_size=num_samples, num_workers = hyper_params.NUM_WORKERS)
    y = []
    all_data = []
    person_indices = []
    for t in dataset:
        if TAKE_MEDIAN:
            all_data.append(torch.cat([torch.median(t[0][1][i, min(hyper_params.MAX_VISITS - 8, max(0, total_n_visits[p.item()] - 8)) :total_n_visits[p.item()] + 1, :], dim = 0).values.unsqueeze(0) for i, p in enumerate(t[0][0]) if total_n_visits[p.item()] > 0]))
        elif TAKE_LAST:
            all_data.append(torch.cat([t[0][1][i, min(total_n_visits[p.item()] - 1, hyper_params.MAX_VISITS - 1), :].unsqueeze(0) for i, p in enumerate(t[0][0]) if total_n_visits[p.item()] > 0]))
        curr_y = list(t[1].detach().cpu().numpy())
        curr_person_indices = list(t[0][0].detach().cpu().numpy())
        y += [x for x, p in zip(curr_y, curr_person_indices) if total_n_visits[p] > 0]
        person_indices += [p for p in person_indices if total_n_visits[p] > 0]
    new_X = torch.cat(all_data, dim = 0).detach().cpu().numpy()
    new_y = y
    return new_X, new_y


# %%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()
params = {
    "n_estimators": [100, 500, 50, 20, 1000],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [2, 10, 20, 50, 100],
    "min_samples_split": [2, 10, 50, 100],
    "max_features": ["sqrt", "log2"],
    "class_weight": [None, "balanced", "balanced_subsample"]
}
N_ITERS = 10000
clf = RandomizedSearchCV(random_forest, params, scoring = 'roc_auc', verbose = 3, n_iter = N_ITERS, n_jobs = 8)
if PERFORM_RANDOM_SEARCH:
    new_X_train, new_y_train = build_dataset(X_val + X_train, y_val + y_train, True)
    new_X_test, new_y_test = build_dataset(X_test, y_test)
    clf.fit(new_X_train, new_y_train)
    print("################################################")
    print("ROC-AUC Score: ", roc_auc_score(new_y_test, clf.predict_proba(new_X_test)[:, 1]))
    print("Best params:", clf.best_params_)



# %%
scores = []
auc_pr_scores = []
features_codes = sorted(list(set(range(len(featureSetInfo.feature_codes_to_id) + featureSetInfo.num_non_numeric_features+ len(featureSetInfo.numeric_feature_to_index))).difference(set(dataset_dict['not_good_features']))))
idx_to_numeric_feature = {val: x for x, val in featureSetInfo.numeric_feature_to_index.items()}
numeric_features = [idx_to_numeric_feature[y] for y in features_codes if y < len(featureSetInfo.numeric_feature_to_index)]
num_not_icd_codes = featureSetInfo.num_numeric_features + featureSetInfo.num_non_numeric_features
idx_to_code_feature = {val: x for x, val in featureSetInfo.feature_codes_to_id.items()}
other_features = [idx_to_code_feature[y - num_not_icd_codes] for y in features_codes if y - num_not_icd_codes >= 0]
feature_names = numeric_features + other_features
X_train, y_train, X_val, y_val, X_test, y_test, new_dataset_dict = \
    get_data(visits_data, dataset_dict['person_indices'], dataset_dict,
    test_val_precentage, validation_precentage, 
        max_visits, total_n_visits, curr_cohort, fix_imbalance = False, need_to_clean_data = False, featureSetInfo = featureSetInfo)
new_X_train, new_y_train = build_dataset(X_train + X_val, y_train + y_val, True)
new_X_test, new_y_test = build_dataset(X_test, y_test)
for i in range(NUM_EXPERIMENTS):
    #Done using 10,000 random searches
    params = {
        'n_estimators': 100, 'min_samples_split': 100, 'max_features': 'sqrt', 
'max_depth': 100, 'criterion': 'log_loss', 'class_weight': 'balanced'
    }
    clf = RandomForestClassifier(**params)
    clf.fit(new_X_train, new_y_train)
    #print(clf.feature_importances_)
    #print("Important featuers: ")
    not_important_features_2 = [features_codes[i] for i, val in enumerate(clf.feature_importances_) if val <= 0.01]
    important_feature_names = [feature_names[i] for i, val in enumerate(clf.feature_importances_) if val > 0]
    print(not_important_features_2)
    y_test_pred = clf.predict_proba(new_X_test)[:, 1]
    curr_score = roc_auc_score(new_y_test, y_test_pred)
    curr_auc_pr_score = average_precision_score(new_y_test, y_test_pred)
    print("Current ROC-AUC score on the test set:  ", curr_score)
    print("Current AUC PR score on the test set:  ", curr_auc_pr_score)
    scores.append(curr_score)
    auc_pr_scores.append(curr_auc_pr_score)

    update_experiment_results(X_test, y_test_pred, new_y_test, "Random Forest")


#%%
print("AUC ROC Score")
print("Mean: ", np.array(scores).mean())
print("STD: ", np.array(scores).std())

print("AUC PR Score")
print("Mean: ", np.array(auc_pr_scores).mean())
print("STD: ", np.array(auc_pr_scores).std())




