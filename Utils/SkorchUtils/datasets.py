import torch
from Models.visit_transformer import NumLastVisits
import hyper_params
import random
import numpy as np
from Utils.data_utils import update_summary_statistics, clean_data_person

# Custom dataset class creation 
class MyDatasetReconstructionSingle(torch.utils.data.Dataset):
    # pylint: disable=anomalous-backslash-in-string
    
    def __init__(
            self,
            max_visits,
            n_visits,
            visits_data,
            task_name,
            X,
            y=None,
            clf = None,
            dataset_dict = None,
            feature_set_info = None,
            mbsz = 32
    ):
        self.max_visits = max_visits
        self.n_visits = n_visits
        self.X = X
        self.y = y

        self.normalize_data = False

        self.task_name = task_name
        self.visits_data = visits_data
        self.clf = clf
        self._len = len(self.X)
        if (y is not None):
            assert(len(X) == len(y))

        self.i = 0
        self.num_last_visits = NumLastVisits().get()

        self.seed_base = random.randint(0, 5000)

        self.dataset_dict = dataset_dict

        self.summary_statistics = update_summary_statistics({}, feature_set_info=feature_set_info)

    def create_tensor(self, x, p):
        return x[max(0, self.n_visits[p] - self.max_size) : max(0, self.n_visits[p] - self.max_size)  + self.max_size, :]

    def get_batch(self, p, num_last_windows_with_zeros = 0):
        with torch.no_grad():
            curr_tensor = self.visits_data[p]
            curr_tensor = (clean_data_person(torch.clone(curr_tensor[0]), self.summary_statistics, self.n_visits, p[1] ), curr_tensor[1], curr_tensor[2])
            max_size = hyper_params.MAX_VISITS #max(hyper_params.MAX_VISITS, curr_tensor[0].shape[1])
            self.max_size = max_size
            reshaped = self.create_tensor(curr_tensor[0], p[1])
            reshaped_non_variant_info = curr_tensor[1].to_dense()
            torch.zeros(size = (reshaped.shape[0], reshaped.shape[1], reshaped_non_variant_info.shape[0]))
            seq_len = min(self.max_visits, self.n_visits[p[1]])
            reshaped_non_variant = torch.zeros(size = (reshaped.shape[0], reshaped_non_variant_info.shape[0]))
            reshaped_non_variant[:seq_len, :] = reshaped_non_variant_info.unsqueeze(dim = 0).repeat(seq_len, 1)
            x = self.create_tensor(curr_tensor[2], p[1])
            reshaped_one_hot_info = x if (max_size - x.shape[0]) <= 0 else torch.cat((x, torch.zeros(size = ( max_size - x.shape[0], x.shape[1]))), dim = 0)
            reshaped_res = torch.cat((reshaped, reshaped_non_variant, reshaped_one_hot_info), dim = 1)
            #reshaped_res = reshaped_res / reshaped_res.max(0, keepdim = True)[0]
            if 'not_good_features' in self.dataset_dict:
                good_features = list(set(range(reshaped_res.shape[1])).difference(self.dataset_dict['not_good_features']))
                reshaped_res = reshaped_res[:, good_features]
            if self.normalize_data:
                for i in range(reshaped_res.shape[1]):
                    if self.dataset_dict['norm_info']['std'][i] != 0.0:
                        reshaped_res[:self.n_visits[p[1]], i] -= self.dataset_dict['norm_info']['mean'][i]
                        reshaped_res[:self.n_visits[p[1]], i] /= self.dataset_dict['norm_info']['std'][i]
        return reshaped_res

    def __len__(self):
        return self._len

    def len(self):
        return self._len
    
    def set_normalize_data(self, normalize_data):
        self.normalize_data = normalize_data

    def __getitem__(self, i):
        if (self.y is None):
            return (self.X[i], torch.nan_to_num(self.get_batch(self.X[i]))), []
        p = self.X[i]
        curr_batch = self.get_batch(p)
        reshaped_res = curr_batch.clone()
        if hyper_params.ZERO_VISITS:
            num_visits = min(self.n_visits[p[1]], hyper_params.MAX_VISITS)
            num_samples = int(hyper_params.SAMPLE_PRECENTAGE * num_visits)
            visits_to_zero = random.sample(range(num_visits), num_samples)
            if self.normalize_data:
                reshaped_res[visits_to_zero, :] = torch.zeros(size = ( len(visits_to_zero), reshaped_res.shape[1]))
            res = (p[1], torch.nan_to_num(reshaped_res) ,np.pad(np.array(visits_to_zero), (0, hyper_params.MAX_VISITS - len(visits_to_zero))).astype(np.float64)), np.nan_to_num(curr_batch).astype(np.float64)
        if hyper_params.ZERO_FEATURES:
            num_features = reshaped_res.shape[1]
            num_samples = int(hyper_params.FEATURE_SAMPLE_PRECENTAGE * num_features)
            features_to_zero = random.sample(range(num_features), num_samples)
            if self.normalize_data:
                reshaped_res[:, features_to_zero] = torch.zeros(size = (  reshaped_res.shape[0], len(features_to_zero)))
            res = (p[1], torch.nan_to_num(reshaped_res) ,np.pad(np.array(features_to_zero), (0, num_features - len(features_to_zero))).astype(np.float64)), np.nan_to_num(curr_batch).astype(np.float64)
        # curr_batch = curr_batch[visits_to_zero, :]
        # curr_batch = torch.nn.functional.pad(curr_batch, pad = (0, 0, 0, int(hyper_params.SAMPLE_PRECENTAGE * hyper_params.MAX_VISITS) + 1 - curr_batch.shape[0]))
        
        return res 


    # Custom dataset class creation 
class MyDatasetSingle(torch.utils.data.Dataset):
    # pylint: disable=anomalous-backslash-in-string
    
    def __init__(
            self,
            max_visits,
            n_visits,
            visits_data,
            task_name,
            X,
            y=None,
            clf = None,
            dataset_dict = None,
            feature_set_info = None,
            mbsz = 32,
            should_mask_input = True,
            if_clean_data = True,
    ):
        self.max_visits = hyper_params.MAX_VISITS
        self.n_visits = n_visits
        self.X = X
        self.y = y
        self.should_mask_input = should_mask_input
        self.normalize_data = False

        self.summary_statistics = update_summary_statistics({}, feature_set_info=feature_set_info)

        self.task_name = task_name
        self.visits_data = visits_data
        self.clf = clf
        self.dataset_dict = dataset_dict
        self._len = len(self.X)
        if (y is not None):
            assert(len(X) == len(y))

        self.num_last_visits = NumLastVisits().get()
        self.if_clean_data = if_clean_data

    def set_normalize_data(self, normalize_data):
        self.normalize_data = normalize_data

    def get_labels(self):
        return self.y

    def create_tensor(self, x, p):
        if hyper_params.USE_INIT_DATA:
            return x[:hyper_params.MAX_VISITS, :]
        return x[max(0, self.n_visits[p] - self.max_size) : max(0, self.n_visits[p] - self.max_size)  + self.max_size, :]


    def get_batch(self, p, num_last_windows_with_zeros = 0):
        with torch.no_grad():
            curr_tensor = self.visits_data[p]
            max_size = hyper_params.MAX_VISITS #max(hyper_params.MAX_VISITS, curr_tensor[0].shape[1])
            self.max_size = max_size
            curr_tensor_numeric = curr_tensor[0]
            if self.if_clean_data:
                curr_tensor_numeric = clean_data_person(curr_tensor[0], self.summary_statistics, self.n_visits, p[1] )
            if len(curr_tensor) > 2:
                curr_tensor = (curr_tensor_numeric, curr_tensor[1], curr_tensor[2])
            else:
                curr_tensor = (curr_tensor_numeric, curr_tensor[1])
            reshaped = self.create_tensor(curr_tensor[0], p[1])
            reshaped_non_variant_info = curr_tensor[1].to_dense()
            torch.zeros(size = (reshaped.shape[0], reshaped.shape[1], reshaped_non_variant_info.shape[0]))
            seq_len = min(self.max_visits, self.n_visits[p[1]])
            reshaped_non_variant = torch.zeros(size = (reshaped.shape[0], reshaped_non_variant_info.shape[0]))
            reshaped_non_variant[:seq_len, :] = reshaped_non_variant_info.unsqueeze(dim = 0).repeat(seq_len, 1)
            if len(curr_tensor) > 2:
                x = self.create_tensor(curr_tensor[2], p[1])
                reshaped_one_hot_info = x if (max_size - x.shape[0]) <= 0 else torch.cat((x, torch.zeros(size = ( max_size - x.shape[0], x.shape[1]))), dim = 0)
                reshaped_res = torch.cat((reshaped, reshaped_non_variant, reshaped_one_hot_info), dim = 1)
            else:
                reshaped_res = torch.cat((reshaped, reshaped_non_variant), dim = 1)
            if 'not_good_features' in self.dataset_dict: # and self.dataset_dict['not_good_features'] is not None:
                good_features = list(set(range(reshaped_res.shape[1])).difference(self.dataset_dict['not_good_features']))
                reshaped_res = reshaped_res[:, good_features]
            if self.normalize_data:
                for i in range(reshaped_res.shape[1]):
                    if self.dataset_dict['norm_info']['std'][i] != 0.0:
                        reshaped_res[:self.n_visits[p[1]], i] -= self.dataset_dict['norm_info']['mean'][i]
                        reshaped_res[:self.n_visits[p[1]], i] /= self.dataset_dict['norm_info']['std'][i]
                    reshaped_res[min(max_size-1, self.n_visits[p[1]]):, i] = 0
                    reshaped_res[min(max_size-1, self.n_visits[p[1]]):, i] = 0
            #     # for i in self.dataset_dict['not_good_features']:
            #     #     reshaped_res[:, i] = torch.zeros((reshaped.shape[0]))
            # #reshaped_res = reshaped_res / reshaped_res.max(0, keepdim = True)[0]
            reshaped_res = torch.nan_to_num(reshaped_res)
        return reshaped_res

    def __len__(self):
        return self._len

    def len(self):
        return self._len

    def __getitem__(self, i):
        curr_x = self.X[i]
        if (self.y is None):
            return (curr_x, torch.nan_to_num(self.get_batch(curr_x))), []
        p = curr_x
        curr_batch = self.get_batch(p)
        reshaped_res = curr_batch.clone()
        if self.should_mask_input and self.normalize_data and self.normalize_data:
            num_visits = min(self.n_visits[p[1]], hyper_params.MAX_VISITS)
            num_samples = int(hyper_params.SAMPLE_PRECENTAGE * num_visits) 
            visits_to_zero = random.sample(range(num_visits), num_samples)
            reshaped_res[visits_to_zero, :] = torch.zeros(size = ( len(visits_to_zero), reshaped_res.shape[1]))
        if hyper_params.ZERO_FEATURES and self.should_mask_input:
            num_features = reshaped_res.shape[1]
            num_samples = int(hyper_params.SAMPLE_PRECENTAGE * num_features)
            features_to_zero = random.sample(range(num_features), num_samples)
            if self.normalize_data:
                reshaped_res[:, features_to_zero] = torch.zeros(size = (  reshaped_res.shape[0], len(features_to_zero)))
        res = (p[1], torch.nan_to_num(reshaped_res)), self.y[i]
        return res 

# import torch
# from Models.visit_transformer import NumLastVisits
# import hyper_params
# import random
# import numpy as np
# from Utils.data_utils import update_summary_statistics, clean_data_person

# # Custom dataset class creation 
# class MyDatasetReconstructionSingle(torch.utils.data.Dataset):
#     # pylint: disable=anomalous-backslash-in-string
    
#     def __init__(
#             self,
#             max_visits,
#             n_visits,
#             visits_data,
#             task_name,
#             X,
#             y=None,
#             clf = None,
#             dataset_dict = None,
#             feature_set_info = None,
#             mbsz = 32
#     ):
#         self.max_visits = max_visits
#         self.n_visits = n_visits
#         self.X = X
#         self.y = y

#         self.normalize_data = False

#         self.task_name = task_name
#         self.visits_data = visits_data
#         self.clf = clf
#         self._len = len(self.X)
#         if (y is not None):
#             assert(len(X) == len(y))

#         self.i = 0
#         self.num_last_visits = NumLastVisits().get()

#         self.seed_base = random.randint(0, 5000)

#         self.dataset_dict = dataset_dict

#         self.summary_statistics = update_summary_statistics({}, feature_set_info=feature_set_info)

#     def create_tensor(self, x, p):
#         return x[max(0, self.n_visits[p] - self.max_size) : max(0, self.n_visits[p] - self.max_size)  + self.max_size, :]

#     def get_batch(self, p, num_last_windows_with_zeros = 0):
#         with torch.no_grad():
#             curr_tensor = self.visits_data[p]
#             curr_tensor = (clean_data_person(torch.clone(curr_tensor[0]), self.summary_statistics, self.n_visits, p[1] ), curr_tensor[1], curr_tensor[2])
#             max_size = hyper_params.MAX_VISITS #max(hyper_params.MAX_VISITS, curr_tensor[0].shape[1])
#             self.max_size = max_size
#             reshaped = self.create_tensor(curr_tensor[0], p[1])
#             reshaped_non_variant_info = curr_tensor[1].to_dense()
#             torch.zeros(size = (reshaped.shape[0], reshaped.shape[1], reshaped_non_variant_info.shape[0]))
#             seq_len = min(self.max_visits, self.n_visits[p[1]])
#             reshaped_non_variant = torch.zeros(size = (reshaped.shape[0], reshaped_non_variant_info.shape[0]))
#             reshaped_non_variant[:seq_len, :] = reshaped_non_variant_info.unsqueeze(dim = 0).repeat(seq_len, 1)
#             x = self.create_tensor(curr_tensor[2], p[1])
#             reshaped_one_hot_info = x if (max_size - x.shape[0]) <= 0 else torch.cat((x, torch.zeros(size = ( max_size - x.shape[0], x.shape[1]))), dim = 0)
#             reshaped_res = torch.cat((reshaped, reshaped_non_variant, reshaped_one_hot_info), dim = 1)
#             #reshaped_res = reshaped_res / reshaped_res.max(0, keepdim = True)[0]
#             if self.normalize_data:
#                 reshaped_res[:self.n_visits[p[1]], :] = torch.Tensor(self.dataset_dict['norm_info'].transform(reshaped_res[:self.n_visits[p[1]], :].detach().cpu().numpy()))
#                 # for i in range(reshaped_res.shape[1]):
#                 #     if self.dataset_dict['norm_info']['std'][i] != 0.0:
#                 #         reshaped_res[:self.n_visits[p[1]], i] -= self.dataset_dict['norm_info']['mean'][i]
#                 #         reshaped_res[:self.n_visits[p[1]], i] /= self.dataset_dict['norm_info']['std'][i]
#         return reshaped_res

#     def __len__(self):
#         return self._len

#     def len(self):
#         return self._len
    
#     def set_normalize_data(self, normalize_data):
#         self.normalize_data = normalize_data

#     def __getitem__(self, i):
#         if (self.y is None):
#             return (self.X[i], torch.nan_to_num(self.get_batch(self.X[i]))), []
#         p = self.X[i]
#         curr_batch = self.get_batch(p)
#         reshaped_res = curr_batch.clone()
#         num_visits = min(self.n_visits[p[1]], hyper_params.MAX_VISITS)
#         num_samples = int(hyper_params.SAMPLE_PRECENTAGE * num_visits)
#         random.seed(self.seed_base + i * 3 + 2)
#         visits_to_zero = random.sample(range(num_visits), num_samples)
#         if self.normalize_data:
#             reshaped_res[visits_to_zero, :] = torch.zeros(size = ( len(visits_to_zero), reshaped_res.shape[1]))
#         # curr_batch = curr_batch[visits_to_zero, :]
#         # curr_batch = torch.nn.functional.pad(curr_batch, pad = (0, 0, 0, int(hyper_params.SAMPLE_PRECENTAGE * hyper_params.MAX_VISITS) + 1 - curr_batch.shape[0]))
#         res = (p[1], torch.nan_to_num(reshaped_res) ,np.pad(np.array(visits_to_zero), (0, hyper_params.MAX_VISITS - len(visits_to_zero))).astype(np.float64)), np.nan_to_num(curr_batch).astype(np.float64)
#         return res 


#     # Custom dataset class creation 
# class MyDatasetSingle(torch.utils.data.Dataset):
#     # pylint: disable=anomalous-backslash-in-string
    
#     def __init__(
#             self,
#             max_visits,
#             n_visits,
#             visits_data,
#             task_name,
#             X,
#             y=None,
#             clf = None,
#             dataset_dict = None,
#             feature_set_info = None,
#             mbsz = 32,
#             should_mask_input = True,
#             if_clean_data = True,
#     ):
#         self.max_visits = hyper_params.MAX_VISITS
#         self.n_visits = n_visits
#         self.X = X
#         self.y = y
#         self.should_mask_input = should_mask_input
#         self.normalize_data = False

#         self.summary_statistics = update_summary_statistics({}, feature_set_info=feature_set_info)

#         self.task_name = task_name
#         self.visits_data = visits_data
#         self.clf = clf
#         self.dataset_dict = dataset_dict
#         self._len = len(self.X)
#         if (y is not None):
#             assert(len(X) == len(y))

#         self.num_last_visits = NumLastVisits().get()
#         self.if_clean_data = if_clean_data

#     def set_normalize_data(self, normalize_data):
#         self.normalize_data = normalize_data

#     def get_labels(self):
#         return self.y

#     def create_tensor(self, x, p):
#         if hyper_params.USE_INIT_DATA:
#             return x[:hyper_params.MAX_VISITS, :]
#         return x[max(0, self.n_visits[p] - self.max_size) : max(0, self.n_visits[p] - self.max_size)  + self.max_size, :]


#     def get_batch(self, p, num_last_windows_with_zeros = 0):
#         with torch.no_grad():
#             curr_tensor = self.visits_data[p]
#             max_size = hyper_params.MAX_VISITS #max(hyper_params.MAX_VISITS, curr_tensor[0].shape[1])
#             self.max_size = max_size
#             curr_tensor_numeric = curr_tensor[0]
#             if self.if_clean_data:
#                 curr_tensor_numeric = clean_data_person(curr_tensor[0], self.summary_statistics, self.n_visits, p[1] )
#             curr_tensor = (curr_tensor_numeric, curr_tensor[1], curr_tensor[2])
#             reshaped = self.create_tensor(curr_tensor[0], p[1])
#             reshaped_non_variant_info = curr_tensor[1].to_dense()
#             torch.zeros(size = (reshaped.shape[0], reshaped.shape[1], reshaped_non_variant_info.shape[0]))
#             seq_len = min(self.max_visits, self.n_visits[p[1]])
#             reshaped_non_variant = torch.zeros(size = (reshaped.shape[0], reshaped_non_variant_info.shape[0]))
#             reshaped_non_variant[:seq_len, :] = reshaped_non_variant_info.unsqueeze(dim = 0).repeat(seq_len, 1)
#             x = self.create_tensor(curr_tensor[2], p[1])
#             reshaped_one_hot_info = x if (max_size - x.shape[0]) <= 0 else torch.cat((x, torch.zeros(size = ( max_size - x.shape[0], x.shape[1]))), dim = 0)
#             reshaped_res = torch.cat((reshaped, reshaped_non_variant, reshaped_one_hot_info), dim = 1)
#             if 'not_good_features' in self.dataset_dict: # and self.dataset_dict['not_good_features'] is not None:
#                 for i in self.dataset_dict['not_good_features']:
#                     reshaped_res[:, i] = torch.zeros((reshaped.shape[0]))
#             if self.normalize_data:
#                 # for i in range(reshaped_res.shape[1]):
#                 #     if self.dataset_dict['norm_info']['std'][i] != 0.0:
#                 #         reshaped_res[:self.n_visits[p[1]], i] -= self.dataset_dict['norm_info']['mean'][i]
#                 #         reshaped_res[:self.n_visits[p[1]], i] /= self.dataset_dict['norm_info']['std'][i]
#                 #     reshaped_res[min(max_size-1, self.n_visits[p[1]]):, i] = 0
#                 #     reshaped_res[min(max_size-1, self.n_visits[p[1]]):, i] = 0
#                 if self.n_visits[p[1]] > 0:
#                     reshaped_res[:self.n_visits[p[1]], :] = torch.Tensor(self.dataset_dict['norm_info'].transform(reshaped_res[:self.n_visits[p[1]], :].detach().cpu().numpy()))
#             #reshaped_res = reshaped_res / reshaped_res.max(0, keepdim = True)[0]
#             #reshaped_res = torch.nan_to_num(reshaped_res)
#         return reshaped_res

#     def __len__(self):
#         return self._len

#     def len(self):
#         return self._len

#     def __getitem__(self, i):
#         curr_x = self.X[i]
#         if (self.y is None):
#             return (curr_x, torch.nan_to_num(self.get_batch(curr_x))), []
#         p = curr_x
#         curr_batch = self.get_batch(p)
#         reshaped_res = curr_batch.clone()
#         if self.should_mask_input and self.normalize_data and self.normalize_data:
#             num_visits = min(self.n_visits[p[1]], hyper_params.MAX_VISITS)
#             num_samples = int(hyper_params.SAMPLE_PRECENTAGE * num_visits) 
#             visits_to_zero = random.sample(range(num_visits), num_samples)
#             reshaped_res[visits_to_zero, :] = torch.zeros(size = ( len(visits_to_zero), reshaped_res.shape[1]))
#         res = (p[1], torch.nan_to_num(reshaped_res)), self.y[i]
#         return res 