import hyper_params
from Models.transformers_models import VTClassifer
from Models.visit_transformer import VisitTransformer
import torch
from Utils.SkorchUtils.classifiers import MyNeuralNetClassifier
import config
import gc
import torch.nn as nn

from Models.optimizers import ChildTuningAdamW

import numpy as np

class ModelWrapper (nn.Module):
    def __init__(self, model_params, dataset_dict, model_name, task_name, max_visits, chunk_size = 10, should_del_model = True, model_loc = None, num_samples = None):
        super(ModelWrapper, self).__init__()
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        self.model_params = model_params
        self.clf = None
        self.dataset_dict = dataset_dict
        self.model_name = model_name
        self.task_name = task_name
        self.max_visits = max_visits
        self.should_del_model = should_del_model
        self.model_loc = model_loc
        self.init_model()

    def set_person_indices(self, person_indices):
        self.person_indices = person_indices

    def init_model(self):
        model_params = {k[len("module__"):]: v for k, v in self.model_params.items()}
        base_model = VisitTransformer(
            **model_params
        )
        base_model.set_data(
            self.dataset_dict
        )
        clf = VTClassifer(
            base_model, **model_params
        ).cuda()

        transformer_net = MyNeuralNetClassifier(clf, module = None,optimizer = ChildTuningAdamW, use_sampler = False, optimizer__mode = hyper_params.OPTIM_TYPE,  dataset_dict = self.dataset_dict,  #iterator_train__sampler = sampler,
                                         iterator_train__pin_memory = True, iterator_train__num_workers = hyper_params.NUM_WORKERS, iterator_valid__batch_size = hyper_params.MBSZ, #iterator_test__sampler = sampler,
                                         iterator_valid__pin_memory = True, iterator_valid__num_workers = hyper_params.NUM_WORKERS, criterion__clf = clf, optimizer__params = clf.parameters(), **self.model_params)
        # clf.load_state_dict(torch.load(config.DEFAULT_SAVE_LOC + "/SavedModels/" + config.TASK + 
        #                                                         '/best_model_' + MODEL_NAME + ADDITIONAL_NAME_FOR_EXPERIMENT))
        transformer_net.initialize()
        if self.model_loc is not None:
            transformer_net.load_params(f_params = self.model_loc)
        else:
            transformer_net.load_params(f_params = config.DEFAULT_SAVE_LOC + "/SavedModels/" + config.TASK + 
                                                            '/best_model_' + self.model_name)
        self.clf = transformer_net.module

        self.clf.eval()
        self.clf.bert.eval()

        del transformer_net
        gc.collect()
        torch.cuda.empty_cache()

    def forward (self, input):
        if type(input) == type(np.array([1])):
            input = torch.Tensor(input)
        if self.num_samples is not None and len(input.shape) < 3:
            input = input.reshape(self.num_samples, input.shape[0] // self.num_samples, input.shape[-1])
        input = input.cpu()
        res = []
        num_chunks = input.shape[0] // self.chunk_size if input.shape[0] % self.chunk_size == 0 else input.shape[0] // self.chunk_size + 1
        chunks = [torch.clone(input[i * self.chunk_size: (i + 1) * self.chunk_size]).cpu() for i in range(num_chunks)]
        person_indices_list = [self.person_indices[i * self.chunk_size: (i + 1) * self.chunk_size] for i in range(num_chunks)]
        for i, (chunk, person_indices) in enumerate(zip(chunks, person_indices_list)):
            if self.should_del_model:
                self.init_model()
            gc.collect()
            torch.cuda.empty_cache()
            curr_input = torch.clone(chunk).cuda()
            self.clf.set_person_range(person_indices)
            curr_res = (self.clf(curr_input)[0]).cpu()
            res.append(curr_res)
            if self.should_del_model:
                del self.clf.bert
                del curr_input, self.clf, chunk
                torch.cuda.synchronize()
        return torch.cat(res, dim = 0)

    def relprop(self, cam, **kwargs):
        return self.clf.relprop(cam, **kwargs)
        
