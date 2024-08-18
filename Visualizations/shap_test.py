# %%
# %%
# %%
# %%
#bsi_tl_recon*
# %% [markdown]
# ## Imports and GPU setup

# %%
import sys, os
sys.path.insert(0, os.path.dirname(__file__) + '/../')
from copy import deepcopy
from Models.shap_models import ModelWrapper
from Utils.SkorchUtils.datasets import MyDatasetSingle
from sklearn.model_selection import train_test_split
import Utils.dbutils as dbutils
from Utils.experiments_utils import save_norm_info
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

import warnings
from skorch.exceptions import SkorchWarning
warnings.filterwarnings('ignore', category=SkorchWarning)

from Utils.data_utils import get_data
import config

from Utils.data_getter import DataGetter

import shap
from Models.shap_models import ModelWrapper

import hyper_params

import math

import math
from torch.utils.data import DataLoader

from Utils.model_params_init import get_model_params

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 13

# %%
assert(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(hyper_params.DEVICE_NUM)

print("Folder Num: ", hyper_params.ADDITIONAL_NAME)

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
TASK = 'eicu_' + hyper_params.CURR_TASK + '_' + str(NUM_MESUREMENTS) + '_' +str(NUM_HOURS_FOR_WINDOW) +'h'
MIMIC_TASK = hyper_params.CURR_TASK +'_' + str(NUM_MESUREMENTS) + '_' +str(NUM_HOURS_FOR_WINDOW) +'h'
CONVERT_TWO = {10: '_19', 20: '_3_20', 100: '_100_2_1', 200: '_100_1', 1:'_1_1'}
FEATURESET_FILE_NAME = 'featureset_reconstruction_eicu_' + str(NUM_MESUREMENTS) + '_' + str(NUM_MESUREMENTS) + '_2_ReconstructionTransformer'
HIDDEN_SIZE = {100: 1, 20: 1, 10: 2, 1:1}
NUM_ATTENTION_HEADS = HIDDEN_SIZE[NUM_MESUREMENTS]
SHOULD_UPDATE_DATA = not SHOULD_UPLOAD_SAVED_FEATURESET_INFO
MIMIC_SHOULD_UPLOAD_SAVED_FEATURESET_INFO = True
SHOULD_USE_CACHE_MIMIC = False
MODEL_LOC = "../outputs/Experiments/eICU_all_0_mimic_ensemble_1/best_model_Transformerbsi_2_0"


PERFORM_RANDOM_SEARCH = True
DF_PRECENTAGE = 1 #0.01 #0.01 #0.2
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
cohort_script_path = config.SQL_PATH_COHORTS + '/' + hyper_params.CURR_TASK + '_eicu_cohort.sql'

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
cohort.build(db, replace=reset_schema)

mimic_cohort_script_path = config.SQL_PATH_COHORTS + '/' + hyper_params.CURR_TASK + '_cohort.sql'
mimic_schema_name = '"' + MIMIC_TASK + '_test' +'"' # all created tables will be created using this schema
mimic_cohort_name = '"' + '__' + MIMIC_TASK + '_cohort' + '"'

mimic_db = dbutils.Database(config_path, mimic_schema_name, connect_args, cdm_schema_name)
if reset_schema:
    mimic_db.execute(
        'drop schema if exists {} cascade'.format(mimic_schema_name)
    )
mimic_db.execute(
    'create schema if not exists {}'.format(mimic_schema_name)
)

mimic_params = {
          'cohort_table_name'     : mimic_cohort_name,
          'schema_name'           : mimic_schema_name,
          'aux_data_schema'       : config.CDM_AUX_SCHEMA,
          'min_hours_in_ICU'      : 48
         }

mimic_cohort = CohortGenerator.Cohort(
    schema_name=mimic_schema_name,
    cohort_table_name=mimic_cohort_name,
    cohort_generation_script=mimic_cohort_script_path,
    cohort_generation_kwargs=mimic_params,
    outcome_col_name='y'
)
mimic_cohort.build(mimic_db, replace= reset_schema)

# %%
# Build the Feature Set by executing SQL queries and reading into tensors
# The tensors are located in featureSet.tensors_for_person. A dictionary where each key is a person_id and each value is 
# the person's tensor.
feature_names_path = config.DEFAULT_SAVE_LOC + '/feature_names'

if os.path.isfile(feature_names_path):
    with open(feature_names_path, 'rb') as pickle_file:
        df_feature_names = pickle.load(pickle_file)

else:
    get_measurments_vals_sql_query = \
    '''
    SELECT count(distinct person_id), measurement_source_value FROM omop.measurement
    where value_as_number is not null
    group by measurement_source_value
    order by count desc
    '''
    #print(get_measurments_vals_sql_query)
    conn = db.engine.raw_connection()
    sql_query = pd.read_sql_query(get_measurments_vals_sql_query, conn)
    df_feature_names = pd.DataFrame(sql_query, columns = ['count', 'measurement_source_value'])

    with open(feature_names_path, 'wb') as pickle_file:
        pickle.dump(df_feature_names, pickle_file)

# %%
features_to_generate_without_file = df_feature_names['measurement_source_value'].values
features_to_generate_without_file = features_to_generate_without_file[:NUM_MESUREMENTS]
temporal_features_list_measurements = list(features_to_generate_without_file)

# %%
temporal_features_list_measurements += ['Temperature', 'Arterial BP [Systolic]', 'Arterial BP [Diastolic]',
'GcsApacheIIScore', 'GCS Total', 'C-Reactive Protein', 'C Reactive Protein (CRP)', 
'HCO3 (serum)', 'Hematocrit', 'Hemoglobin', 'Potassium', 'Sodium', 
'pH', 'pO2', 'Prothrombin time', 'WBC (4-11,000)', 'WBC   (4-11,000)', 'Lactate', 'Urine Out Foley', 'Foley',
'RDW', 'RBC', 'Asparate Aminotransferase (AST)', 'MCHC', 'Bilirubin, Total', 
'Neutrophils', 'Albumin', 'Creatinine', 'Hematocrit', 'Alkaline Phosphatase', 'MCV', 'Alanine Aminotransferase (ALT)', 
'ALT', 'Lymphocytes']
temporal_features_list_measurements = list(set(temporal_features_list_measurements))

# %% [markdown]
# ### 3. Generate and build a feature set for each patient in the cohort using some default features


# %%
# Build the Feature Set by executing SQL queries and reading into tensors
# The tensors are located in featureSet.tensors_for_person. A dictionary where each key is a person_id and each value is 
# the person's tensor.
feature_set_path = config.DEFAULT_SAVE_LOC + '/featureset_' + TASK + '_' + str(NUM_MESUREMENTS) + '_' + str(NUM_HOURS_FOR_WINDOW) \
                    + '_' + MODEL_NAME
cache_data_path = config.DEFAULT_SAVE_LOC + '/cache_data_bsi_test_' + str(NUM_MESUREMENTS)

if SHOULD_UPLOAD_SAVED_FEATURESET_INFO and os.path.isfile(feature_set_path):
    with open(feature_set_path, 'rb') as pickle_file:
        featureSetInfo = pickle.load(pickle_file)
else:
    featureSet = FeatureGenerator.FeatureSet(db, task_name = TASK,
    feature_set_file_name = FEATURESET_FILE_NAME)
    temporal_features_list_observation = [{"name": 'language', "observation_concept_id": 40758030}]
    featureSet.add_default_features(
        temporal_features_list_measurements,
        schema_name,
        cohort_name,
        from_sql_file = False,
        type = "Measurement"
    )
    eicu_measurements = ['eicu_measurements']
    featureSet.add_default_features(
        eicu_measurements,
        schema_name,
        cohort_name,
        from_sql_file = True,
        type = "Measurement"
    )
    non_temportal_feature_list = ['age_eicu',  'gender_eicu',  'first_care_unit_eicu']
    featureSet.add_default_features(
        non_temportal_feature_list,
        schema_name,
        cohort_name,
        temporal = False,
        type = "General"
    )

    non_temportal_feature_list = ['diagnosis_eicu', 'medical_history_eicu', 'drug_eicu']# 'procedures', 'drug']
    featureSet.add_default_features(
        non_temportal_feature_list,
        schema_name,
        cohort_name,
        temporal = True,
        type = "Diagnosis",
        with_feature_end_date = True
    )
    time_delta = FeatureGenerator.TimeDelta(hours = NUM_HOURS_FOR_WINDOW)#(hours = 2)
    numeric_names = pd.read_csv("./Tables/mimic_name_to_general.txt", sep = ' -- ')
    featureSet.numeric_features += list(numeric_names["General"].values)
    # featureSet.postprocess_func = post_process
    featureSet.build(cohort, time_delta = time_delta, from_cached=SHOULD_USE_CACHE, cache_file=cache_data_path,
                features_to_generate_without_file = features_to_generate_without_file,
                    use_prebuilt_features = True)
    featureSet.build_bit_vec_features(cohort, time_delta = time_delta, from_cached=SHOULD_USE_CACHE, cache_file=cache_data_path,
                features_to_generate_without_file = features_to_generate_without_file,
                    use_prebuilt_features = True)              
    featureSetInfo = FeatureSetInfo(featureSet, task_name=TASK)
    with open(feature_set_path, 'wb') as pickle_file:
        pickle.dump(featureSetInfo, pickle_file)


# %%
# Build the Feature Set by executing SQL queries and reading into tensors
# The tensors are located in featureSet.tensors_for_person. A dictionary where each key is a person_id and each value is 
# the person's tensor.
mimic_feature_set_path = config.DEFAULT_SAVE_LOC + '/featureset_' + MIMIC_TASK + '_' + str(NUM_MESUREMENTS) + '_' + str(NUM_HOURS_FOR_WINDOW) \
                    + '_' + MODEL_NAME
mimic_cache_data_path = config.DEFAULT_SAVE_LOC + '/mimic_cache_data_bsi_test_' + str(NUM_MESUREMENTS)

if MIMIC_SHOULD_UPLOAD_SAVED_FEATURESET_INFO and os.path.isfile(mimic_feature_set_path):
    with open(mimic_feature_set_path, 'rb') as pickle_file:
        mimic_featureSetInfo = pickle.load(pickle_file)
else:
    mimic_featureSet = FeatureGenerator.FeatureSet(mimic_db, task_name = MIMIC_TASK,
    feature_set_file_name = FEATURESET_FILE_NAME)
    temporal_features_list_observation = [{"name": 'language', "observation_concept_id": 40758030}]
    mimic_featureSet.add_default_features(
        temporal_features_list_measurements,
        mimic_schema_name,
        mimic_cohort_name,
        from_sql_file = False,
        type = "Measurement"
    )
    non_temportal_feature_list = ['age',  'gender',  'first_care_unit']
    mimic_featureSet.add_default_features(
        non_temportal_feature_list,
        mimic_schema_name,
        mimic_cohort_name,
        temporal = False,
        type = "General"
    )

    non_temportal_feature_list = ['diagnosis', 'medical_history', 'drug']# 'procedures', 'drug']
    mimic_featureSet.add_default_features(
        non_temportal_feature_list,
        mimic_schema_name,
        mimic_cohort_name,
        temporal = True,
        type = "Diagnosis",
        with_feature_end_date = True
    )
    time_delta = FeatureGenerator.TimeDelta(hours = NUM_HOURS_FOR_WINDOW)#(hours = 2)
    numeric_names = pd.read_csv("./Tables/mimic_name_to_general.txt", sep = ' -- ')
    mimic_featureSet.numeric_features += list(numeric_names["General"].values)
    # featureSet.postprocess_func = post_process
    mimic_featureSet.build(mimic_cohort, time_delta = time_delta, from_cached=SHOULD_USE_CACHE_MIMIC, cache_file=mimic_cache_data_path,
                features_to_generate_without_file = features_to_generate_without_file,
                    use_prebuilt_features = True)
    mimic_featureSet.build_bit_vec_features(mimic_cohort, time_delta = time_delta, from_cached=SHOULD_USE_CACHE_MIMIC, cache_file=mimic_cache_data_path,
                features_to_generate_without_file = features_to_generate_without_file,
                    use_prebuilt_features = True)              
    mimic_featureSetInfo = FeatureSetInfo(mimic_featureSet, task_name=MIMIC_TASK)
    with open(mimic_feature_set_path, 'wb') as pickle_file:
        pickle.dump(mimic_featureSetInfo, pickle_file)


# %% [markdown]
# ### 4. Process the collected data and calculate indices needed for the deep model

# %%
def get_dict_path(person_id):
    return config.DEFAULT_SAVE_LOC + '/Dictionary' + TASK +'/' + str(person_id)

# %%
person_indices  = featureSetInfo.person_ids
person_indices = list(map(int, person_indices))
unique_id = featureSetInfo.unique_id_col
person_indices = set(person_indices).intersection(set(cohort._cohort[unique_id].values))
person_indices = [(0, x) for x in person_indices]
idx_to_person = {i : id for i, id in enumerate(sorted(list(person_indices)))}
person_to_idx = {id : i for i, id in enumerate(sorted(list(person_indices)))}
#visits_data = featureSetInfo.post_process_transformer
dict_file = get_dict_path(person_indices[0][1]) + '_postprocessed_tensor'
with open(dict_file, 'rb') as pickle_file:
    curr_tensor_info = pickle.load(pickle_file)
    max_visits = curr_tensor_info[0].shape[0]
n_visits = {p[1]: math.ceil(np.max(np.array(featureSetInfo.window_times_for_person[p[1]])).astype('timedelta64[h]').astype(int) / NUM_HOURS_FOR_WINDOW) for p in person_indices }
if hyper_params.USE_INIT_DATA:
    person_indices = [(0, p) for p, val in n_visits.items() if val > (hyper_params.NUM_HOURS_PREDICTION // NUM_HOURS_FOR_WINDOW)]
person_indices = sorted(person_indices, key = lambda x: x[1])
curr_cohort = cohort._cohort[np.isin(cohort._cohort[unique_id].values, [x[1] for x in person_indices])]
curr_cohort = curr_cohort.sort_values(by = unique_id)
outcomes_filt = curr_cohort['y'].values

#%% 
mimic_person_indices  = mimic_featureSetInfo.person_ids
mimic_person_indices = list(map(int, mimic_person_indices))
mimic_person_indices = [(1, x) for x in mimic_person_indices]
#assert(len(set(person_indices).intersection(set(mimic_featureSetInfo.person_ids))) == 0)
mimic_n_visits = {p[1]: math.ceil(np.max(np.array(mimic_featureSetInfo.window_times_for_person[p[1]])).astype('timedelta64[h]').astype(int) / NUM_HOURS_FOR_WINDOW) for p in mimic_person_indices }
if hyper_params.USE_INIT_DATA:
    mimic_person_indices = [(1, p) for p, val in mimic_n_visits.items() if val > (hyper_params.NUM_HOURS_PREDICTION // NUM_HOURS_FOR_WINDOW)]
mimic_curr_cohort = mimic_cohort._cohort[np.isin(mimic_cohort._cohort[unique_id].values, [x[1] for x in mimic_person_indices])]
mimic_person_indices = sorted(mimic_person_indices, key = lambda x: x[1])
mimic_curr_cohort = mimic_curr_cohort.sort_values(by = unique_id)
mimic_outcomes_filt = mimic_curr_cohort['y'].values

# %%
one_label_precentage = np.sum(outcomes_filt) / len(outcomes_filt)
print("Precentage of 1 label: ", one_label_precentage)
print("For now uses Weighted random sampler")

# %%
one_label_precentage = np.sum(mimic_outcomes_filt) / len(mimic_outcomes_filt)
print("Mimic Precentage of 1 label: ", one_label_precentage)

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


visits_data = DataGetter([TASK, MIMIC_TASK])

 # %%
dataset_dict = {
    'person_indices': person_indices, #The ids of all the persons
    'outcomes_filt': outcomes_filt, # A pandas Series defined such that outcomes_filt.iloc[i] is the outcome of the ith patient
    'idx_to_person': idx_to_person,
    'n_visits': n_visits,
    'visits_data': visits_data,
    'num_invariant_features': featureSetInfo.num_non_numeric_features,
    'mimic_n_visits': mimic_n_visits
}

#%%

if hyper_params.USE_MIMIC_AS_TEST_DATA:
    dataset_dict['mimic_test_data'] = (mimic_person_indices, mimic_outcomes_filt)

elif hyper_params.USE_COMBINATION_AS_TRAIN_SET:
    dataset_dict['person_indices'] = person_indices + mimic_person_indices
    person_indices += mimic_person_indices
    n_visits.update(mimic_n_visits)
    dataset_dict['n_visits'] = n_visits
    outcomes_filt = np.concatenate((outcomes_filt, mimic_outcomes_filt))
    dataset_dict['outcomes_filt'] = outcomes_filt
    curr_cohort = pd.concat([curr_cohort, mimic_curr_cohort])
else:
    dataset_dict['person_indices'] = mimic_person_indices
    dataset_dict['n_visits'] = mimic_n_visits
    dataset_dict['outcomes_filt'] = mimic_outcomes_filt
    curr_cohort = mimic_curr_cohort
    dataset_dict['mimic_test_data'] = (person_indices, outcomes_filt)
    dataset_dict['mimic_n_visits'] = n_visits

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

model_params =  get_model_params(embedding_dim, n_heads, featureSetInfo)

# %%
X_train, y_train, X_val, y_val, X_test, y_test, new_dataset_dict = \
        get_data(visits_data, dataset_dict['person_indices'], dataset_dict, 
        test_val_precentage, validation_precentage, 
            max_visits, n_visits, curr_cohort, fix_imbalance = False, need_to_clean_data = False, featureSetInfo = featureSetInfo)

positive_indices = np.where(np.array(y_train) == 1)[0]
positive_X_train = [X_train[idx] for idx in positive_indices]

dataset = MyDatasetSingle(max_visits, dataset_dict['n_visits'], dataset_dict['visits_data'], TASK, X_train, y_train, 
            clf = None, mbsz = mbsz, dataset_dict = dataset_dict, feature_set_info=featureSetInfo)
save_norm_info(dataset_dict = dataset_dict, data_loader = dataset)
dataset.set_normalize_data(True)
# %%
# %%
num_samples = 100 #200 #100 #100
dataset = DataLoader(dataset, pin_memory = True, batch_size=num_samples, num_workers = hyper_params.NUM_WORKERS)
for t in dataset:
    data = t[0][1]
    person_indices = t[0][0]
    break
# %%
model = ModelWrapper(model_params = model_params, dataset_dict=dataset_dict, model_name = MODEL_NAME + ADDITIONAL_NAME_FOR_EXPERIMENT, 
    model_loc = MODEL_LOC, task_name = TASK, max_visits=max_visits, should_del_model=False, num_samples = num_samples)
model.set_person_indices(person_indices[:num_samples // 2])
#with torch.no_grad():
    # explainer = shap.KernelExplainer(model, data = data[:num_samples // 2, : , :].numpy()) 
    # model.set_person_indices(torch.cat((person_indices[num_samples // 2:], person_indices[:num_samples // 2]), dim = 0))
    # shap_values = explainer.shap_values(data[num_samples // 2:, : , :].numpy().reshape(-1, data[num_samples // 2:, : , :].shape[-1]))
explainer = shap.GradientExplainer(model, data = data[:num_samples // 2, : , :]) 
model.set_person_indices(torch.cat((person_indices[num_samples // 2:], person_indices[:num_samples // 2]), dim = 0))
shap_values = explainer.shap_values(data[num_samples // 2:, : , :])
# shap_vals_constrained_features = []
# for i in range(data.shape[-1]):
#     curr_data = data.clone()[num_samples // 2:, : , :]
#     curr_data[:, :, i] = data[:num_samples // 2, :, i]
#     shap_vals_constrained_features.append(explainer.shap_values(curr_data))

expected_value = explainer.explainer.expected_value

shap_res = {"shap_values": shap_values,
"explain_data": data,
"expected_value": expected_value,
"person_indices": person_indices,
"data": data,
"dataset_dict": dataset_dict,
"model_pred": np.argmax(model(data).detach().cpu().numpy(), axis = 1)
}

# %%



# %%

# %%


# %%
import pickle

cache_data_path = config.DEFAULT_SAVE_LOC + '/shap_results'
with open(cache_data_path, "wb") as pickle_file:
    pickle.dump(shap_res, pickle_file)



