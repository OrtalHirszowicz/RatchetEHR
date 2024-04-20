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


def save_norm_info(dataset_dict, data_loader, if_mimic = False):
    sums = None
    means_len = 0
    std_sums = None
    for x in DataLoader(dataset = data_loader, batch_size = 128, pin_memory=True, num_workers=hyper_params.NUM_WORKERS):
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
    means = sums / means_len
    stds = np.nan_to_num(np.sqrt(np.abs((std_sums / means_len) - (means ** 2))))
    dataset_dict['norm_info'] = {}
    dataset_dict['norm_info']['mean'] = means
    dataset_dict['norm_info']['std'] = stds
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
                splitter = predefined_split(valid_ds)
            else:
                splitter = ValidSplit(hyper_params.K_FOLD_SIZE, stratified = True)
            callbacks=[
                ('train_acc', EpochScoring(scoring='accuracy', name = 'accuracy', on_train=True, lower_is_better=False)),
                ('train_f1', EpochScoring(scoring='f1', name = 'f1', on_train=True, lower_is_better=False)),
                ('train_roc_auc', EpochScoring(scoring='roc_auc', name = 'roc_auc', on_train=True, lower_is_better=False)), 
                ('train_auc_pr', EpochScoring(scoring='average_precision', name = 'auc_pr', on_train=True, lower_is_better=False)), 
                ('checkpoint', Checkpoint(monitor= hyper_params.MONITOR_TYPE, f_params = config.DEFAULT_SAVE_LOC + "/SavedModels/" + config.TASK + 
                                                        '/best_model_' + self.model_name + '_' + str(i) + '_' + str(hyper_params.SEED_NUMBER))),
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
            nets.append((transformer_net, dataset_dict , dataset))
            if hyper_params.ENSEMBLE_SIZE > 1:
                self.dataset_dict['not_good_features'] = not_good_features_orig
        torch.autograd.set_detect_anomaly(True)
        for i, net in enumerate(nets):
            net[2].dataset_dict = net[1]
            history = net[0].fit(X = net[2], y = np.array(self.y_train))
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
        

        for i in range(len(nets)):
            params = [{'params': [], 'mode': None}]
            transformer_net = MyNeuralNetClassifier(bert_weights = bert_weights, task_name = task_name, dataset_dict=nets[i][1],  optimizer = ChildTuningAdamW if hyper_params.OPTIMIZER != "AdamW" else AdamW,  callbacks = callbacks
                    ,max_epochs = self.ft_epochs, dataset = dataset, iterator_train__batch_size = mbsz, module = None,
                    iterator_train__pin_memory = True, iterator_train__num_workers = hyper_params.NUM_WORKERS, iterator_valid__batch_size = mbsz, 
                    iterator_valid__pin_memory = True, iterator_valid__num_workers = hyper_params.NUM_WORKERS, 
                    verbose = self.verbose, criterion__lambda_param = self.lambda_param, optimizer__lr = self.lr, optimizer__params = params, **model_params )
            
            transformer_net.initialize()
            
            transformer_net.load_params(f_params = config.DEFAULT_SAVE_LOC + "/SavedModels/" + config.TASK + 
                                                        '/best_model_' + self.model_name + '_' + str(i) + '_' + str(hyper_params.SEED_NUMBER))
        
            transformer_net.module.eval()

            nets[i] = (transformer_net, nets[i][1], nets[i][2])
        mimic_pr_score = 0
        with torch.no_grad(): 
            y_pred_test = []
            y_pred_val = []
            for net in nets:
                dataset = MyDatasetSingle(self.max_visits, self.dataset_dict['n_visits'], self.dataset_dict['visits_data'], task_name, self.X_test, 
                    mbsz = mbsz, should_mask_input=False, dataset_dict = nets[i][1], feature_set_info=feature_set_info)
                dataset.set_normalize_data(True)
                dataset_val = MyDatasetSingle(self.max_visits, self.dataset_dict['n_visits'], self.dataset_dict['visits_data'], task_name, self.X_val, 
                    mbsz = mbsz, should_mask_input=False, dataset_dict = nets[i][1], feature_set_info=feature_set_info)
                dataset.set_normalize_data(True)
                dataset_val.set_normalize_data(True)
                y_pred_test.append(np.expand_dims(net[0].predict_proba( dataset) [:, 1], 1))
                y_pred_val.append(np.expand_dims(net[0].predict_proba(dataset_val)[:, 1], 1))
            y_pred_test = np.mean(np.concatenate(y_pred_test, axis = 1), axis = 1)
            y_pred_val = np.mean(np.concatenate(y_pred_val, axis = 1), axis = 1)

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

