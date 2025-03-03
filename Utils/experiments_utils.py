import collections
from copy import deepcopy
import os
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from Utils.SkorchUtils.criterions import L1Loss
import config
import torch
import gc
import numpy as np
import hyper_params

from Models.rnn_model import VisitSimple
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import StepLR

if hyper_params.TRANSFORMER_TYPE not in ["DETR", "LRP"]:
    from Models.transformers_models import VTClassifer
    from Models.visit_transformer import VisitTransformer
from skorch.callbacks import EpochScoring, Callback
from Utils.SkorchUtils.datasets import MyDatasetReconstructionSingle, MyDatasetSingle
from Utils.SkorchUtils.classifiers import MyNeuralNetClassifier
from torch.optim import AdamW
from Models.optimizers import mAdamW, ChildTuningAdamW
from skorch.callbacks import EarlyStopping, ProgressBar, Checkpoint

from torch.utils.data import DataLoader


from skorch.helper import predefined_split
from skorch.dataset import ValidSplit

from sklearn.model_selection import train_test_split

import random

def update_experiment_results(person_indices, y_pred_test, y_test, experiment_name):
    if hyper_params.USE_TEST_GROUP:
        test_group_file_loc = hyper_params.LOCATION_OF_RESULTS_ON_TEST_GROUP
        if not os.path.exists(test_group_file_loc):
            with open(test_group_file_loc, 'wb') as f:
                pickle.dump(collections.defaultdict(dict), f)
        with open(test_group_file_loc, 'rb') as f:
            curr_res = pickle.load(f)
            curr_res[experiment_name] = {
                "Person indices": person_indices,
                "Estimated score": y_pred_test,
                "Ground truth": y_test
            } 
        with open(test_group_file_loc, 'wb') as f:
            pickle.dump(curr_res, f)


def save_norm_info(dataset_dict, data_loader, if_mimic = False, is_test=False):
    sums = None
    means_len = 0
    std_sums = None
    for x in DataLoader(dataset = data_loader, batch_size = 128, pin_memory=True, num_workers=hyper_params.NUM_WORKERS):
        try:
            if sums is None:
                sums = np.zeros(shape = (x[0][1].shape[-1]))
                std_sums = np.zeros(shape = (x[0][1].shape[-1]))
            curr_len = sum([dataset_dict['mimic_n_visits' if if_mimic else 'n_visits'][p.item()] for p in x[0][0]])
            means_len += curr_len
            for i in range(x[0][1].shape[-1]):
                sums[i] += torch.sum(x[0][1][:,:, i]).item() 
                std_sums[i] += torch.sum(x[0][1][:,:, i] ** 2).item() 
            del x
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
    means = sums / means_len
    stds = np.nan_to_num(np.sqrt(np.abs((std_sums / means_len) - (means ** 2))))
    stds[27:] = 0
    if not is_test:
        dataset_dict['norm_info'] = {}
    # print(f"means: {means.__repr__()}")
    # print(f"stds: {stds.__repr__()}")

    if is_test:
        train_test_multiply = []
        for i in range(26):
            train_test_multiply.append(dataset_dict['norm_info']['mean'][i] / means[i])
        
        # print(f"mimic_eicu_multiply: {train_test_multiply}")
        return
        

    # Hack means to normalize values according to mimic_means/eicu_means
    # eicu_multiply = {0: 1.1543457508087158,
    #                 1: 0.4182162582874298,
    #                 2: 0.6439023017883301,
    #                 3: 1.304754614830017,
    #                 4: 0.2815670669078827,
    #                 5: 0.7608580589294434,
    #                 6: 0.7266958951950073,
    #                 7: 0.7319362759590149,
    #                 8: 1.061784029006958,
    #                 9: 3.5780935287475586,
    #                 10: 0.7582082152366638,
    #                 11: 0.35705775022506714,
    #                 12: 0.7509816884994507,
    #                 13: 0.7581446766853333,
    #                 14: 0.8129744529724121,
    #                 15: 1.133150339126587,
    #                 16: 0.7859081625938416,
    #                 17: 1.2323776483535767,
    #                 18: 0.7411289215087891,
    #                 19: 2.851663589477539,
    #                 20: 0.7586720585823059,
    #                 21: 0.6318545341491699,
    #                 22: 0.811314582824707,
    #                 23: 5.703190326690674,
    #                 24: 0.8561076521873474,
    #                 25: 0.8793057799339294}

    # eicu_multiply = {0: 2.125330924987793,
    #                 1: 1.1330621242523193,
    #                 2: 1.0537651777267456,
    #                 3: 2.4396607875823975,
    #                 4: 0.8294560313224792,
    #                 5: 0.8699194192886353,
    #                 6: 0.9715360999107361,
    #                 7: 0.9771843552589417,
    #                 8: 1.0438992977142334,
    #                 9: 1.4898715019226074,
    #                 10: 0.8932877779006958,
    #                 11: 0.9925414323806763,
    #                 12: 1.004792332649231,
    #                 13: 1.0112175941467285,
    #                 14: 1.0341627597808838,
    #                 15: 1.0957410335540771,
    #                 16: 1.015944480895996,
    #                 17: 8.334455490112305,
    #                 18: 0.978984534740448,
    #                 19: 1.1381853818893433,
    #                 20: 1.003302812576294,
    #                 21: 1.39203941822052,
    #                 22: 0.9594324827194214,
    #                 23: 17.681446075439453,
    #                 24: 0.975580632686615,
    #                 25: 1.0982935428619385}

    # eicu_multiply = {0: 37.049468994140625,
    #                 1: 19.17434310913086,
    #                 2: 18.616552352905273,
    #                 3: 42.14234161376953,
    #                 4: 237.5410919189453,
    #                 5: 12.40732192993164,
    #                 6: 13.88382625579834,
    #                 7: 13.96271800994873,
    #                 8: 20.650697708129883,
    #                 9: 126.07775115966797,
    #                 10: 16.64463996887207,
    #                 11: 15.178378105163574,
    #                 12: 14.363238334655762,
    #                 13: 14.455086708068848,
    #                 14: 16.43892478942871,
    #                 15: 21.663076400756836,
    #                 16: 14.486641883850098,
    #                 17: 119.11432647705078,
    #                 18: 14.007542610168457,
    #                 19: 45.59672546386719,
    #                 20: 14.306382179260254,
    #                 21: 186.80601501464844,
    #                 22: 13.685796737670898,
    #                 23: 252.69956970214844,
    #                 24: 18.178443908691406,
    #                 25: 20.60381317138672}

    # full_list = list(eicu_multiply.values()) + list([1])*31
    mimic_means = [84.5626897,
                    1.26641865,
                    67.4690172,
                    121.72434,
                    2.5912688,
                    1.26879189,
                    25.4866527,
                    8.3880105,
                    1.15286064,
                    205.380628,
                    1.25769147,
                    5.06000139,
                    27.6281699,
                    77.2428593,
                    1.74605908,
                    12.6292338,
                    3.44685305,
                    5.25999196,
                    13.3463771,
                    17.5751611,
                    117.796738,
                    15.7226882,
                    24.9041986,
                    108.722238,
                    5.58675452,
                    78.1548415,
                    50.7245765,
                    0.235339059,
                    0.229098847,
                    0.194495326,
                    0.359920277,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0]
    
    mimic_stds = [442.482656,
                1.50404805,
                116.053069,
                701.511706,
                19.7642807,
                1.56572871,
                12.2000955,
                4.02150758,
                0.908097806,
                827.246839,
                1.46468748,
                9.73698876,
                12.1153675,
                34.2262221,
                1.81738303,
                9.61587548,
                1.57577731,
                28.37154,
                6.27917662,
                8.86548465,
                50.7389416,
                18.3196585,
                24.5045973,
                7386.76516,
                3.03902182,
                70.2873613,
                27.6980806,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0]
    
    eicu_multiply = {}
    for i in range(len(mimic_means)):
        eicu_multiply[i] = mimic_means[i] / means[i] if means[i] != 0 else 1

    # print(f"eicu_multiply is: {eicu_multiply}")

    means_value = {}
    stds_value = {}
    for i in range(len(mimic_means)):
        means_value[i] = mimic_means[i] / eicu_multiply[i] if eicu_multiply[i] != 0 else 1
        stds_value[i] = mimic_stds[i] / eicu_multiply[i] if eicu_multiply[i] != 0 else 0
    
    if not is_test:
        dataset_dict['norm_info']['mean'] = means if hyper_params.SHOULD_FINETUNE else means
        dataset_dict['norm_info']['std'] = list(stds_value.values()) if hyper_params.SHOULD_FINETUNE else stds 
    # return mimic_means, mimic_stds
    return means, stds


class Monitor:
    def __init__(self, monitor, epochs_to_save_after_high = 0, train_val_max_diff = 0.1) -> None:
        self.monitor = monitor
        self.time_before_monitor_true = -1
        self.epochs_to_save_after_high = epochs_to_save_after_high
        self.train_val_max_diff = train_val_max_diff

    def __call__(self, net):
        res = net.history[-1, self.monitor] and net.history[-1, 'roc_auc'] - net.history[-1, 'val_roc_auc'] <= self.train_val_max_diff
        if res:
            self.time_before_monitor_true = 0
        else:
            self.time_before_monitor_true += 1
        return self.time_before_monitor_true >= 0 and self.time_before_monitor_true <= self.epochs_to_save_after_high

class ExperimentConducterTransferLearning:
    def __init__(self, param_dict) -> None:
        self.embedding_dim = param_dict['embedding_dim']
        self.num_most_important_features = param_dict['num_most_important_features']
        self.dataset_dict = param_dict['dataset_dict']
        self.max_visits = param_dict['max_visits']
        self.lambda_param = param_dict['lambda_param'] #0.01
        self.lr = param_dict['lr'] #0.001
        self.X_val = param_dict['X_val']
        self.y_val = param_dict['y_val']
        self.ft_epochs = param_dict['ft_epochs']
        self.X_train = param_dict['X_train']
        self.y_train = param_dict['y_train']
        self.X_test = param_dict['X_test']
        self.y_test = param_dict['y_test']
        self.model_params = param_dict['model_params']
        self.mbsz = param_dict['mbsz']
        self.model_name = param_dict['model_name']
        self.verbose = param_dict['verbose']
        self.weight_decay = param_dict['weight_decay']
        self.num_transformer_blocks_to_freeze = param_dict['num_transformer_blocks_to_freeze']
        self.lower_lr = param_dict['lower_lr']
        self.lower_weight_decay = param_dict['lower_weight_decay']

    def conduct_experiment(self, num, task_name, ft_epochs,
        bert_weights, visit_transformer_type = 'VisitTransformer', use_sampler = False, feature_set_info = None):
        # using the same split as before, create train/validate/test batches for the deep model
        # `mbsz` might need to be decreased based on the GPU's memory and the number of features being used
        mbsz = self.mbsz
        #assert embedding_dim % n_heads == 0

        model_params = self.model_params
        
        val_auc_arr = []

        dataset = MyDatasetSingle(self.max_visits, self.dataset_dict['n_visits'], self.dataset_dict['visits_data'], task_name, self.X_train, self.y_train, 
            clf = None, mbsz = mbsz, dataset_dict = self.dataset_dict, feature_set_info=feature_set_info)
        save_norm_info(dataset_dict = self.dataset_dict, data_loader = dataset)
        dataset.set_normalize_data(True)
        train_dataset = dataset
        nets = []
        for i in range(hyper_params.ENSEMBLE_SIZE):
            not_good_features = None
            if hyper_params.ENSEMBLE_SIZE > 1:
                not_good_features_orig = self.dataset_dict['not_good_features']
                num_numeric_features = self.dataset_dict['num_numeric_features']
                good_features = list(set(range(num_numeric_features)).difference(not_good_features_orig))
                not_good_features = sorted(not_good_features_orig + random.sample(good_features, len(good_features) // 3))
                self.dataset_dict['not_good_features'] = not_good_features
                X_train_new, _, y_train_new, _ = train_test_split(self.X_train, self.y_train, test_size=0.3, stratify=self.y_train)
                dataset = MyDatasetSingle(self.max_visits, self.dataset_dict['n_visits'], self.dataset_dict['visits_data'], task_name, X_train_new, y_train_new, 
                    clf = None, mbsz = mbsz, dataset_dict = self.dataset_dict, feature_set_info=feature_set_info)
                dataset.set_normalize_data(True)
            dataset_dict = deepcopy(self.dataset_dict)
            if hyper_params.K_FOLD_SIZE == None:
                # valid_ds = MyDatasetSingle(self.max_visits, self.dataset_dict['n_visits'], self.dataset_dict['visits_data'], task_name, 
                #     self.X_val, self.y_val, clf = None, mbsz = mbsz, should_mask_input=False, dataset_dict = dataset_dict, feature_set_info=feature_set_info)
                valid_ds = MyDatasetSingle(self.max_visits, self.dataset_dict['n_visits'], self.dataset_dict['visits_data'], task_name, 
                    self.X_test, self.y_test, clf = None, mbsz = mbsz, should_mask_input=False, dataset_dict = dataset_dict, feature_set_info=feature_set_info)
                valid_ds.set_normalize_data(True)
                splitter = None#predefined_split(valid_ds)
            else:
                splitter = ValidSplit(hyper_params.K_FOLD_SIZE, stratified = True)
            
            lr_scheduler_callback = LRScheduler(policy=StepLR, step_size=4, gamma=0.7)
            callbacks=[
                ('train_acc', EpochScoring(scoring='accuracy', name = 'accuracy', on_train=True, lower_is_better=False)),
                ('train_f1', EpochScoring(scoring='f1', name = 'f1', on_train=True, lower_is_better=False)),
                ('train_roc_auc', EpochScoring(scoring='roc_auc', name = 'roc_auc', on_train=True, lower_is_better=False)), 
                ('train_auc_pr', EpochScoring(scoring='average_precision', name = 'auc_pr', on_train=True, lower_is_better=False)), 
                ('checkpoint', Checkpoint(monitor= hyper_params.MONITOR_TYPE, f_params = config.DEFAULT_SAVE_LOC + "/SavedModels/" + config.TASK + 
                                                        '/best_model_' + self.model_name + '_' + str(i) + '_' + str(hyper_params.SEED_NUMBER))),
                ('lr_scheduler', lr_scheduler_callback),
                ('test_loss', EpochScoring(scoring='neg_log_loss', name='test_loss', lower_is_better=True, on_train=False)),
            ]
            if not hyper_params.USE_TEST_GROUP:
                callbacks += [
                    ('val_roc_auc', EpochScoring('roc_auc', name='val_roc_auc', lower_is_better=False)),
                    ('val_accuracy', EpochScoring('accuracy', name='val_accuracy', lower_is_better=False,)),
                    ('val_f1', EpochScoring(scoring='f1', name = 'val_f1', lower_is_better=False)),
                    ('val_auc_pr', EpochScoring(scoring='average_precision', name = 'val_auc_pr', lower_is_better=False)),
                    ('progress_bar', ProgressBar(postfix_keys = [])),
                ]
            else:
                callbacks += [
                    ('progress_bar', ProgressBar(postfix_keys = ['train_loss'])),
                ]
            # if hyper_params.SHOULD_USE_VAL_SET:
            #     callbacks.append(('early_stopping', EarlyStopping(monitor = 'val_roc_auc', patience = hyper_params.EARLY_STOPPING_EPOCHS, lower_is_better=False)))
            transformer_net = MyNeuralNetClassifier(bert_weights = bert_weights, task_name = task_name, dataset_dict=dataset_dict, module = None,
                                            optimizer = ChildTuningAdamW if hyper_params.OPTIMIZER != "AdamW" else AdamW, use_sampler = use_sampler,   
                                            callbacks = callbacks, train_split = splitter, optimizer__bert_lr = hyper_params.BERT_LR
                                            ,max_epochs = self.ft_epochs, dataset = dataset, iterator_train__batch_size = mbsz, #iterator_train__sampler = sampler,
                                            iterator_train__pin_memory = True,  iterator_valid__batch_size = mbsz, #iterator_test__sampler = sampler,
                                            iterator_valid__pin_memory = True, iterator_valid__num_workers = hyper_params.NUM_WORKERS, iterator_train__num_workers = hyper_params.NUM_WORKERS,
                                            verbose = self.verbose, criterion__lambda_param = self.lambda_param, optimizer__lr = self.lr, **model_params)
            # import ipdb; ipdb.set_trace()
            nets.append((transformer_net, dataset_dict , dataset))
            if hyper_params.ENSEMBLE_SIZE > 1:
                self.dataset_dict['not_good_features'] = not_good_features_orig
        torch.autograd.set_detect_anomaly(True)
        if not hyper_params.TEST_ONLY:
            for i, net in enumerate(nets):
                net[2].dataset_dict = net[1]
                import ipdb; ipdb.set_trace()
                history = net[0].fit(X = net[2], y = np.array(self.y_train))
                print(f"losses list: {net[0].losses_list}")
                import ipdb; ipdb.set_trace()
            ##Drawing AUC curve 
            plt.clf()
            training_roc_auc_arr = [x['roc_auc'] for x in history.history]
            plt.plot(range(len(training_roc_auc_arr)), training_roc_auc_arr, label = 'Train')
            if not hyper_params.USE_TEST_GROUP:
                val_auc_arr = [x['val_roc_auc'] for x in history.history]
                plt.plot(range(len(val_auc_arr)), val_auc_arr, label = 'Validation')
            plt.title('AUC Plots - Transformer Model')
            plt.xlabel('Epochs')
            plt.ylabel('AUC')
            x = plt.legend()
            plt.savefig(config.DEFAULT_SAVE_LOC + '/training_figures/' + self.model_name + '/auc_experiment_num_' + str(num))

            ##Drawing AUC PR curve 
            plt.clf()
            training_roc_auc_arr = [x['auc_pr'] for x in history.history]
            plt.plot(range(len(training_roc_auc_arr)), training_roc_auc_arr, label = 'Train')
            if not hyper_params.USE_TEST_GROUP:
                val_auc_arr = [x['val_auc_pr'] for x in history.history]
                plt.plot(range(len(val_auc_arr)), val_auc_arr, label = 'Validation')
            plt.title('AUC PR Plots - Transformer Model')
            plt.xlabel('Epochs')
            plt.ylabel('AUC PR')
            x = plt.legend()
            plt.savefig(config.DEFAULT_SAVE_LOC + '/training_figures/' + self.model_name + '/auc_pr_experiment_num_' + str(num))


            ##Drawing Accuracy curve
            plt.clf()
            training_accruacy_arr = [x['accuracy'] for x in history.history]
            plt.plot(range(len(training_accruacy_arr)), training_accruacy_arr, label = 'Train')
            if not hyper_params.USE_TEST_GROUP:
                val_accruacy_arr = [x['val_accuracy'] for x in history.history]
                plt.plot(range(len(val_accruacy_arr)), val_accruacy_arr, label = 'Validation')
            plt.title('Accuracy Plots - Transformer Model')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            x = plt.legend()
            plt.savefig(config.DEFAULT_SAVE_LOC + '/training_figures/' + self.model_name + '/accuracy_experiment_num_' + str(num))


            del transformer_net

        gc.collect()
        torch.cuda.empty_cache()

        #Test set evaluation
        if hyper_params.SEED_NUMBER is not None:
            import random
            seed_num = hyper_params.SEED_NUMBER
            print("Seed: ", seed_num)
            torch.manual_seed(seed_num)
            random.seed(seed_num)
            np.random.seed(seed_num)
            torch.use_deterministic_algorithms(True)
            torch.cuda.manual_seed(seed_num)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed_num)
        for i in range(len(nets)):
            params = [{'params': [], 'mode': None}]
            transformer_net = MyNeuralNetClassifier(bert_weights = bert_weights, task_name = task_name, dataset_dict=nets[i][1],  optimizer = ChildTuningAdamW if hyper_params.OPTIMIZER != "AdamW" else AdamW,  callbacks = callbacks
                    ,max_epochs = self.ft_epochs, dataset = dataset, iterator_train__batch_size = mbsz, module = None,
                    iterator_train__pin_memory = True, iterator_train__num_workers = hyper_params.NUM_WORKERS, iterator_valid__batch_size = mbsz, 
                    iterator_valid__pin_memory = True, iterator_valid__num_workers = hyper_params.NUM_WORKERS, 
                    verbose = self.verbose, criterion__lambda_param = self.lambda_param, optimizer__lr = self.lr, optimizer__params = params, **model_params )
            
            transformer_net.initialize()
            # transformer_net.load_params(f_params = "/bigdata/omerg/RatchetEHR/tmp/tmp/SavedModels/mimiciv_bsi_100_2h/best_best_model_Transformerbsi_finetune")
            # transformer_net.load_params(f_params = "/bigdata/omerg/RatchetEHR/tmp/tmp/SavedModels/bsi/best_model_Transformerbsi_0_None")
            if hyper_params.TEST_ONLY:
                transformer_net.load_params(f_params = bert_weights)
            else:
                transformer_net.load_params(f_params = config.DEFAULT_SAVE_LOC + "/SavedModels/" + config.TASK + 
                                                        '/best_model_' + self.model_name + '_' + str(i) + '_' + str(hyper_params.SEED_NUMBER))
        
            transformer_net.module.eval()

            nets[i] = (transformer_net, nets[i][1], nets[i][2])
        mimic_pr_score = 0
        with torch.no_grad(): 
            y_pred_test = []
            y_pred_val = []
            for net in nets:
                # import ipdb; ipdb.set_trace()
                if hyper_params.TEST_ON_TRAIN:
                    self.X_test = self.X_train
                    self.y_test = self.y_train
                dataset = MyDatasetSingle(self.max_visits, self.dataset_dict['n_visits'], self.dataset_dict['visits_data'], task_name, self.X_test, self.y_test,
                    mbsz = mbsz, should_mask_input=False, dataset_dict = nets[i][1], feature_set_info=feature_set_info)
                save_norm_info(dataset_dict = self.dataset_dict, data_loader = dataset, is_test=True)
                dataset = MyDatasetSingle(self.max_visits, self.dataset_dict['n_visits'], self.dataset_dict['visits_data'], task_name, self.X_test,
                    mbsz = mbsz, should_mask_input=False, dataset_dict = nets[i][1], feature_set_info=feature_set_info)
                dataset.set_normalize_data(True)
                dataset_val = MyDatasetSingle(self.max_visits, self.dataset_dict['n_visits'], self.dataset_dict['visits_data'], task_name, self.X_val, 
                    mbsz = mbsz, should_mask_input=False, dataset_dict = nets[i][1], feature_set_info=feature_set_info)
                dataset.set_normalize_data(True)
                dataset_val.set_normalize_data(True)
                # STopping to calculate loss
                # import ipdb; ipdb.set_trace()
                y_pred_test.append(np.expand_dims(net[0].predict_proba( dataset) [:, 1], 1))
                y_pred_val.append(np.expand_dims(net[0].predict_proba(dataset_val)[:, 1], 1))
            y_pred_test = np.mean(np.concatenate(y_pred_test, axis = 1), axis = 1)
            y_pred_val = np.mean(np.concatenate(y_pred_val, axis = 1), axis = 1)
            print(f"Y real: {self.y_test}")
            print(f"Y prediction: {y_pred_test}")
            score = roc_auc_score(self.y_val, y_pred_val)
            print("Current ROC-AUC score on the val set: ", score)
            pr_score = average_precision_score(self.y_val, y_pred_val)
            print("Current AUC PR score on the val set: ", pr_score)
            score = roc_auc_score(self.y_test, y_pred_test)
            print("Current ROC-AUC score on the test set: ", score)
            pr_score = average_precision_score(self.y_test, y_pred_test)
            print("Current AUC PR score on the test set: ", pr_score)
            mimic_score = -1

            if hyper_params.USE_TEST_GROUP:
                update_experiment_results(self.X_test, y_pred_test, self.y_test, hyper_params.experiment_name)
            
        if 'mimic_test_data' in self.dataset_dict and self.dataset_dict['mimic_test_data'] is not None:
            mimic_person_indices = self.dataset_dict['mimic_test_data'][0]
            mimic_y = np.array(self.dataset_dict['mimic_test_data'][1])
            if hasattr( transformer_net.module, "change_n_visits"):
                transformer_net.module.change_n_visits(self.dataset_dict['mimic_n_visits'])
            if hyper_params.MIMIC_PRECENTAGE_IN_TRAINING_DATA > 0:
                mimic_X_train, mimic_X_val_test, mimic_y_train, mimic_y_val_test = train_test_split(mimic_person_indices, mimic_y, test_size = 1 - hyper_params.MIMIC_PRECENTAGE_IN_TRAINING_DATA, 
                    stratify=mimic_y)
                mimic_X_val, mimic_X_test, mimic_y_val, mimic_y_test = train_test_split(mimic_X_val_test, mimic_y_val_test, test_size = 1 - hyper_params.MIMIC_PRECENTAGE_IN_TRAINING_DATA, 
                    stratify=mimic_y_val_test)
                datasets = {}
                for name, X, y in [("train" ,mimic_X_train, mimic_y_train), ("val", mimic_X_val, mimic_y_val), ("test", mimic_X_test, None)]:
                    datasets[name] = MyDatasetSingle(self.max_visits, self.dataset_dict['mimic_n_visits'], self.dataset_dict['visits_data'], task_name, 
                        X, y = y, mbsz = mbsz, should_mask_input=False, dataset_dict = self.dataset_dict, feature_set_info=feature_set_info)
                    datasets[name].set_normalize_data(True)

                save_norm_info(dataset_dict = self.dataset_dict, data_loader = datasets["train"], if_mimic = True)

                splitter = predefined_split(datasets["val"])
                
                transformer_net.train_split = splitter
                transformer_net.warm_start = True
                for param_group in transformer_net.optimizer_.param_groups:
                    param_group['lr'] = 1e-4
                i = 0
                for name, cb in transformer_net.callbacks_:
                    if name == 'checkpoint':
                        break
                    i += 1
                transformer_net.callbacks_[i] = ('checkpoint', Checkpoint(monitor= hyper_params.MONITOR_TYPE, f_params = config.DEFAULT_SAVE_LOC + "/SavedModels/" + config.TASK + 
                                                        '/best_model_' + self.model_name + '_other_dataset'))
                transformer_net.fit(datasets["train"], mimic_y_val)
                with torch.no_grad():
                    y_pred_test_mimic = transformer_net.predict_proba( datasets["test"]) [:, 1]
                    mimic_score = roc_auc_score(np.array(mimic_y_test), y_pred_test_mimic)
                    print("Current ROC-AUC score on the mimic test set: ", mimic_score)
                    mimic_pr_score = average_precision_score(np.array(mimic_y_test), y_pred_test_mimic)
                    print("Current AUC PR score on MIMIC-III: ", mimic_pr_score)
            else:
                with torch.no_grad():
                    #self.dataset_dict['not_good_features'] = None
                    dataset = MyDatasetSingle(self.max_visits, self.dataset_dict['mimic_n_visits'], self.dataset_dict['visits_data'], task_name, 
                        mimic_person_indices, y = mimic_y, mbsz = mbsz, should_mask_input=False, dataset_dict = self.dataset_dict, feature_set_info=feature_set_info)
                    #save_norm_info(dataset_dict = self.dataset_dict, data_loader = dataset, if_mimic = True)
                    dataset.set_normalize_data(True)
                    y_pred_test_mimic = transformer_net.predict_proba( X = dataset) [:, 1]
                    mimic_score = roc_auc_score(mimic_y, y_pred_test_mimic)
                    print("Current ROC-AUC score on MIMIC-III: ", mimic_score)
                    mimic_pr_score = average_precision_score(mimic_y, y_pred_test_mimic)
                    print("Current AUC PR score on MIMIC-III: ", mimic_pr_score)

                    if hyper_params.CALC_EXTERNAL:
                        if hasattr( transformer_net.module, "change_n_visits"):
                            transformer_net.module.change_n_visits(self.dataset_dict['n_visits'])
                        #self.dataset_dict['not_good_features'] = None
                        dataset = MyDatasetSingle(self.max_visits, self.dataset_dict['n_visits'], self.dataset_dict['visits_data'], task_name, 
                            self.dataset_dict['external_person_indices'], y = self.dataset_dict['external_outcome_filt'], mbsz = mbsz, should_mask_input=False, dataset_dict = self.dataset_dict, feature_set_info=feature_set_info)
                        #save_norm_info(dataset_dict = self.dataset_dict, data_loader = dataset, if_mimic = True)
                        dataset.set_normalize_data(True)
                        y_pred_test_external = transformer_net.predict_proba( X = dataset) [:, 1]
                        external_score = roc_auc_score(self.dataset_dict['external_outcome_filt'], y_pred_test_external)
                        print("Current ROC-AUC score on External: ", external_score)
                        external_pr_score = average_precision_score(self.dataset_dict['external_outcome_filt'], y_pred_test_external)
                        print("Current AUC PR score on External: ", external_pr_score)
        
        
        return mimic_score, score, pr_score, mimic_pr_score, transformer_net

                
        clf.train();



# %%
class ExperimentConducterReconstruction:
    def __init__(self, param_dict) -> None:
        self.embedding_dim = param_dict['embedding_dim']
        self.num_most_important_features = param_dict['num_most_important_features']
        self.dataset_dict = param_dict['dataset_dict']
        self.max_visits = param_dict['max_visits']
        self.lambda_param = param_dict['lambda_param'] #0.01
        self.lr = param_dict['lr'] #0.001
        self.X_val = param_dict['X_val']
        self.y_val = param_dict['y_val']
        self.ft_epochs = param_dict['ft_epochs']
        self.X_train = param_dict['X_train']
        self.y_train = param_dict['y_train']
        self.X_test = param_dict['X_test']
        self.y_test = param_dict['y_test']
        self.model_params = param_dict['model_params']
        self.mbsz = param_dict['mbsz']
        self.model_name = param_dict['model_name']
        self.verbose = param_dict['verbose']
        self.weight_decay = param_dict['weight_decay']

    def conduct_experiment(self, num, task_name, feature_set_info = None, bert_weights = None):
        # using the same split as before, create train/validate/test batches for the deep model
        # `mbsz` might need to be decreased based on the GPU's memory and the number of features being used
        mbsz = self.mbsz
        #assert embedding_dim % n_heads == 0

        model_params = self.model_params

        
        lr = self.lr #0.001
        callbacks=[
            #('early_stopping', EarlyStopping(monitor = 'loss', patience = 20, lower_is_better=True, threshold=1e-8)),
            ('checkpoint', Checkpoint(monitor= 'train_loss_best', f_params = config.DEFAULT_SAVE_LOC + "/SavedModels/" + config.TASK + 
                                                        '/best_model_' + self.model_name)),
            ('progress_bar', ProgressBar()), 
            ]

        dataset = MyDatasetReconstructionSingle(self.max_visits, self.dataset_dict['n_visits'], self.dataset_dict['visits_data'], task_name, 
            self.X_train, self.y_train, mbsz = mbsz, dataset_dict = self.dataset_dict, feature_set_info=feature_set_info)
        save_norm_info(dataset_dict = self.dataset_dict, data_loader = dataset)
        dataset.set_normalize_data(True)
        valid_ds = MyDatasetReconstructionSingle(self.max_visits, self.dataset_dict['n_visits'], self.dataset_dict['visits_data'], task_name, 
            self.X_val, self.y_val, mbsz = mbsz, dataset_dict = self.dataset_dict, feature_set_info=feature_set_info)
        valid_ds.set_normalize_data(True)
        splitter = predefined_split(valid_ds)

        transformer_net = MyNeuralNetClassifier(task_name = task_name, dataset_dict=self.dataset_dict,  optimizer = mAdamW, module = None, optimizer__lr = lr, callbacks = callbacks,
                                         max_epochs = self.ft_epochs, dataset = dataset, train_split=splitter,
                                         iterator_valid__batch_size = mbsz, iterator_valid__timeout = 60*10, 
                                         iterator_valid__pin_memory = True, iterator_valid__num_workers = hyper_params.NUM_WORKERS,
                                         iterator_train__batch_size = mbsz, iterator_train__timeout = 60*10, 
                                         iterator_train__pin_memory = True, iterator_train__num_workers = hyper_params.NUM_WORKERS, 
                                         optimizer__weight_decay = self.weight_decay, bert_weights=bert_weights,
                                        verbose = self.verbose, criterion = L1Loss, **model_params)
        transformer_net.fit(dataset, None)


        del transformer_net
        gc.collect()
        torch.cuda.empty_cache()

        params = [{'params': [], 'mode': None}]
        transformer_net = MyNeuralNetClassifier(task_name = task_name, dataset_dict=self.dataset_dict,  optimizer = mAdamW, module = None, optimizer__lr = lr, callbacks = callbacks,
                                         max_epochs = self.ft_epochs, dataset = dataset, train_split=splitter,
                                         iterator_valid__batch_size = mbsz, iterator_valid__timeout = 60*10, 
                                         iterator_valid__pin_memory = True, iterator_valid__num_workers = hyper_params.NUM_WORKERS,
                                         iterator_train__batch_size = mbsz, iterator_train__timeout = 60*10, 
                                         iterator_train__pin_memory = True, iterator_train__num_workers = hyper_params.NUM_WORKERS, 
                                         optimizer__weight_decay = self.weight_decay, optimizer__params = params,
                                        verbose = self.verbose, criterion = L1Loss, **model_params)
        transformer_net.initialize()
        transformer_net.load_params(f_params = config.DEFAULT_SAVE_LOC + "/SavedModels/" + config.TASK + 
                                                        '/best_model_' + self.model_name)
        
        transformer_net.module.eval()
        def get_batch(x, dataset):
            return torch.cat([x[1] for x in DataLoader(dataset = dataset, batch_size = hyper_params.MBSZ, pin_memory=True, num_workers=hyper_params.NUM_WORKERS)], dim = 0)
            # torch.cuda.empty_cache()
            # person_range = [person_idx.item() if torch.is_tensor(person_idx) else person_idx for person_idx in x]
            # persons_data = {p: self.dataset_dict['visits_data'][p].to_dense() for p in person_range}
            # tensors_for_visits = [persons_data[p][0].unsqueeze(dim = 0).cuda() for p in person_range]
            # reshaped = torch.cat(tensors_for_visits, dim =0).cpu()
            # return reshaped
    
        with torch.no_grad(): 
            dataset = MyDatasetReconstructionSingle(self.max_visits, self.dataset_dict['n_visits'], self.dataset_dict['visits_data'], task_name, self.X_test, 
                self.y_test, mbsz = mbsz, dataset_dict = self.dataset_dict, feature_set_info=feature_set_info)
            dataset.set_normalize_data(True)
            y_pred_test =transformer_net.predict(dataset)
            score = torch.nn.MSELoss()(torch.from_numpy(y_pred_test), get_batch(self.X_test, dataset)).item()
            output_string = "Current ROC-AUC score on the test set: " + str(score) + "\n"

            dataset = MyDatasetReconstructionSingle(self.max_visits, self.dataset_dict['n_visits'], self.dataset_dict['visits_data'], task_name, self.X_val, 
                self.y_val,  mbsz = mbsz, dataset_dict= self.dataset_dict, feature_set_info=feature_set_info)
            dataset.set_normalize_data(True)
            y_pred_val =transformer_net.predict(dataset)
            val_score = torch.nn.MSELoss()(torch.from_numpy(y_pred_val), get_batch(self.X_val, dataset)).item()
            output_string += "Current ROC-AUC score on the validation set: " + str(val_score) + "\n"
            print(output_string)
            with open(config.DEFAULT_SAVE_LOC + "/ModelOutput/" + self.model_name, 'a') as f:
                f.write(output_string)

        torch.cuda.empty_cache()
        
        
        return score, transformer_net

