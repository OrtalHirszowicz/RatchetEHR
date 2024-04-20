
import torch
import config
from os.path import exists
import pickle

def get_interaction(min_feature_idx, max_feature_idx, shap_results):
    num_samples = shap_results["num_samples"]
    data = shap_results["explain_data"]
    all_data = data.clone()
    #Create coalition - replacing players i, j with S_{i j} which will be feature min _index:
    all_data[:, :, min_feature_idx] += all_data[:, :, max_feature_idx]
    all_data[:, :, max_feature_idx] = torch.zeros_like(all_data[:, :, max_feature_idx])
    curr_data = all_data[num_samples // 2:, : , :]
    background_data = all_data[:num_samples // 2]
    shap_results["explainer"].data = background_data
    phi_1 = shap_results["explainer"].shap_values(curr_data)[:, :, min_feature_idx]
    phi_2 = shap_results["shap_vals_constrained_features"][min_feature_idx][:, :, max_feature_idx]
    phi_3 = shap_results["shap_vals_constrained_features"][max_feature_idx][:, :, min_feature_idx]
    return phi_1 - (phi_2 + phi_3)

def get_interactions(first_feature_idx, second_feature_idx):
    max_feature_idx = max((first_feature_idx, second_feature_idx))
    min_feature_idx = min((first_feature_idx, second_feature_idx))

    shap_data_path = config.DEFAULT_SAVE_LOC + '/shap_results'
    if exists(shap_data_path):
        with open(shap_data_path, "rb") as pickle_file:
            shap_results = pickle.load(pickle_file)
    if (min_feature_idx, max_feature_idx) not in shap_results["interactions"]:
        shap_results["interactions"][(min_feature_idx, max_feature_idx)] = get_interaction(min_feature_idx, max_feature_idx)
    
    with open(shap_data_path, "wb") as pickle_file:
        pickle.dump(shap_results, pickle_file)
    return shap_results["interactions"][(min_feature_idx, max_feature_idx)]