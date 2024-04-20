#transformer utils
import torch
import pandas as pd
import config
import pickle
import os

def get_dict_path(task_name, person_id):
    return config.DEFAULT_SAVE_LOC + '/Dictionary' + task_name +'/' + str(person_id)

def postprocess_tensor(person_id, max_visits, task_name):
    dict_file = get_dict_path(task_name, person_id)
    if not os.path.isfile(dict_file):
        return 
    with open(dict_file, 'rb') as pickle_file:
        t = pickle.load(pickle_file)
    curr_t = t[0].coalesce()
    dense_tensor = torch.full(curr_t.shape, fill_value = float('nan'))
    indices = curr_t.indices()
    dense_tensor[indices[0, :], indices[1, :]] = curr_t.values()
    t_arr = dense_tensor.numpy()
    #Interpolation of signals from the input (the signals are measurements)
    for feature_idx in range(t_arr.shape[1]):
        curr_signal = pd.DataFrame(t_arr[:, feature_idx])
        curr_signal_interpolated = curr_signal.interpolate()
        dense_tensor[:, feature_idx] = torch.from_numpy(curr_signal_interpolated.values).to(dense_tensor).squeeze(dim = 1)
    #Filling the left NaN values - features that have no information (signal should be only zeros)
    if (max_visits - dense_tensor.shape[0] == 0):
        return (dense_tensor, t[1])
    padding = torch.zeros(size=(max_visits - dense_tensor.shape[0], dense_tensor.shape[1]))
    padded = torch.cat((dense_tensor,padding), dim = 0)
    dict_file = get_dict_path(task_name, person_id) + '_postprocessed_tensor'
    with open(dict_file, 'wb') as pickle_file:
        if len(t) > 2:
            pickle.dump((torch.nan_to_num(padded).to_sparse(), t[1], t[2]), pickle_file)
        else:
            pickle.dump((torch.nan_to_num(padded).to_sparse(), t[1]), pickle_file)


def get_shape(person_id, loc, task_name):
    dict_file = get_dict_path(task_name, person_id)
    if not os.path.isfile(dict_file):
        return 0
    with open(dict_file, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data[loc].shape[0]

def post_process(person_ids, task_name):
    max_visits = max([get_shape(person_id, 0, task_name) for person_id in person_ids] + [get_shape(person_id, 1, task_name) for person_id in person_ids]) + 10
    i = [postprocess_tensor(person_id, max_visits, task_name) for person_id in person_ids]
