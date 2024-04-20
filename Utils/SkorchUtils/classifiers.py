
from ast import Not
from pyexpat import model
import torch
from skorch.classifier import NeuralNet
import numpy as np
from sklearn.base import BaseEstimator
from Models.rnn_model import VisitSimple
from Models.transformers_models import VTClassifer
from Models.visit_transformer import VisitTransformer
from Utils.SkorchUtils.criterions import my_criterion
from Utils.SkorchUtils.datasets import MyDatasetSingle
from skorch.dataset import unpack_data, get_len, ValidSplit
import scipy as sp
from skorch.utils import to_tensor, is_dataset, to_numpy
from skorch.utils import TeeGenerator
import hyper_params
import gc
from torch.utils.data.sampler import WeightedRandomSampler
import re

def reducer_sparse(func, x):
    #Creating a dense tensor that have NaN values, then uses nanmedian or  nanmean
    # dense_tensor = torch.full(x.shape, fill_value = float('nan')).cuda()
    # indices = x.indices()
    # dense_tensor[indices[0, :], indices[1, :], indices[2, :]] = x.values()
    return torch.nan_to_num(func(x))

class MyNeuralNetClassifier(NeuralNet, BaseEstimator):
    def __init__(
        self, 
        model_params = None,
        task_name = None,
        dataset_dict = None,
        callbacks = None,
        bert_weights = None,
        reducer = 'median',
        simple_model = None,
        visits_data_simple = None,
        *args, 
        train_split= None, 
        num_classes = 2,
        criterion = my_criterion,
        dataset = MyDatasetSingle, 
        batch_size = 1,
        device = 'cuda',
        use_sampler = False,
        max_grad_norm = hyper_params.MAX_GRAD_NORM,
        reserve_p = hyper_params.RESERVE_P,
        **kwargs
    ):
        self.reducer = reducer
        
        my_args = {}

        self.bert_weights = bert_weights
        self.curr_step = 0
        self.verbose_optimizer_step = 32
        self.use_sampler = use_sampler
        self.max_grad_norm = max_grad_norm
        self.reserve_p = reserve_p
        self.task_name = task_name
        self.dataset_dict = dataset_dict
        self.max_visits = hyper_params.MAX_VISITS
        super().__init__(criterion=criterion, train_split=train_split if not hyper_params.USE_TEST_GROUP else None, batch_size = batch_size, device = device, 
                         dataset = dataset, callbacks = callbacks, *args, **kwargs, **my_args)
        self.classes_ = np.array(range(num_classes))
        self.simple_model = simple_model
        self.visits_data_simple = visits_data_simple
        #self.criterion.simple_model = simple_model

    def initialize_module(self):
       if self.module is None:
        model_params = {k: v for k, v in self._kwargs.items() if k[:len(hyper_params.MODEL_PARAMS_PREFIX)] == hyper_params.MODEL_PARAMS_PREFIX }
        #model_params.update({hyper_params.MODEL_PARAMS_PREFIX + k: v for k, v in model_params.items()})
        clf, params = self.create_clf(model_params, self.task_name)
        self.__dict__['optimizer__params'] = params
        kwargs = {k: v for k, v in self._kwargs.items() if k[:len(hyper_params.MODEL_PARAMS_PREFIX)] != hyper_params.MODEL_PARAMS_PREFIX}
        if 'module' in kwargs:
          kwargs.pop('module')
        self.__dict__.update(model_params)
        # pylint: disable=attribute-defined-outside-init
        self.module_ = clf
        # pylint: disable=attribute-defined-outside-init
        self.module = clf
       #return super().initialize_module()

    def fit(self, X, y=None, **fit_params):
      return super().fit(X, y, **fit_params)
      
    def create_clf(self, model_params, task_name):
        model_params = {k[len(hyper_params.MODEL_PARAMS_PREFIX):]: v for k, v in model_params.items() }
        if hyper_params.USE_SIMPLE_MODEL:
            clf = VisitSimple(**model_params, task_name=task_name)
            clf.set_data(self.dataset_dict)
            params = [{'params': list(clf.parameters()), 'mode': None}]
            return clf, params
        base_model = VisitTransformer(
                **model_params, max_visits = self.max_visits, task_name = task_name
            )
        base_model.set_data(
            self.dataset_dict
            )
        clf = VTClassifer(
            base_model,  **model_params
        ).cuda()
        if self.bert_weights is not None or hyper_params.CONDUCT_TRANSFER_LEARNING:
          bert_pattern = re.compile('bert\.*')
          #bert_pattern = re.compile('bert\.tfs\.layers\.5* | bert\.tfs\.layers\.4* *')
          not_pretrain_layers = [v for k, v in list(clf.named_parameters()) if not bert_pattern.match(k)]
          if self.bert_weights != None:
            pretrain_state_dict = torch.load(self.bert_weights + '_original', map_location='cpu')
            pretrain_state_dict = {key: val for key, val in pretrain_state_dict.items() if bert_pattern.match(key)}
            clf.load_state_dict(pretrain_state_dict, strict=False)
          pretrain_layers = [v for k, v in list(clf.named_parameters()) if bert_pattern.match(k)]
          if hyper_params.BERT_LR > 0:
            params = [
              {'params': pretrain_layers, 'is_bert': True, 'mode': hyper_params.OPTIM_TYPE, 'weight_decay': hyper_params.WEIGHT_DECAY}, 
              {'params': not_pretrain_layers, 'is_bert': False, 'mode': None, 'weight_decay': hyper_params.WEIGHT_DECAY}
              ]
          else:
            params = [
              {'params': not_pretrain_layers, 'is_bert': False, 'weight_decay': self.weight_decay}
              ]
        else:
          params = [{'params': list(clf.parameters()), 'mode': None}]
        return clf, params

    def get_params(self, deep=True):
        params = BaseEstimator.get_params(self, deep=deep)
        params['reducer'] = self.reducer
        return params
    

    def set_params(self, **kwargs):
        reducer_funcs = {
            'mean':  lambda x: reducer_sparse(lambda y: torch.nanmean(y, dim = 1), x),
            'median': lambda x: reducer_sparse(lambda y: torch.nanmedian(y, dim = 1)[0], x)
        }
        for key, val in kwargs.items():
            if (key == 'reducer'):
                self.module.reducer = reducer_funcs[val]
                self.reducer = val

        return super().set_params(**kwargs)

    def predict_proba(self, X):
      probas = super().predict_proba(X)
      return probas

    def predict(self, X):
      nonlin = self._get_predict_nonlinearity()
      y_probas = []
      for yp in self.forward_iter(X, training=False):
            yp = yp[0] if isinstance(yp, tuple) and not hyper_params.ZERO_VISITS else (yp[1] if isinstance(yp, tuple) else yp)
            yp = nonlin(yp)
            y_probas.append(to_numpy(yp))
      probas = np.concatenate(y_probas, 0)
      if len(probas.shape) == 2:
        return np.argmax(probas, axis= 1)
      return probas

    def initialize_optimizer(self, triggered_directly=None):
        """Initialize the model optimizer. If ``self.optimizer__lr``
        is not set, use ``self.lr`` instead.

        Parameters
        ----------
        triggered_directly
          Deprecated, don't use it anymore.

        """
        named_parameters = self.get_all_learnable_params()
        args, kwargs = self.get_params_for_optimizer(
            'optimizer', named_parameters)

        # pylint: disable=attribute-defined-outside-init
        if hyper_params.OPTIMIZER == 'AdamW' and 'bert_lr' in kwargs:
           del kwargs['bert_lr']
        self.optimizer_ = self.optimizer(*args, **kwargs)
        return self

    def _get_params_for_optimizer(self, prefix, named_parameters):
        kwargs = self.get_params_for(prefix)

        args = ()

        # 'lr' is an optimizer param that can be set without the 'optimizer__'
        # prefix because it's so common
        if 'lr' not in kwargs:
            kwargs['lr'] = self.lr
        return args, kwargs

    def infer(self, x, **fit_params):
        """Perform a single inference step on a batch of data.

        Parameters
        ----------
        x : input data
          A batch of the input data.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        if len(x) == 3:
          self.module.set_visits_zeroed(x[2].detach().cpu())
        x = to_tensor(x, device=self.device)
        if hasattr(self.module, 'set_person_range'):
          self.module.set_person_range(x[0].detach().cpu())
        x = x[1]
        if isinstance(x, dict):
            x_dict = self._merge_x_and_fit_params(x, fit_params)
            return self.module_(**x_dict)
        if 'epoch_num' in fit_params:
          del fit_params['epoch_num']
          del fit_params['prev_batch']
        return self.module_(x, **fit_params)

    def train_step_single(self, batch, epoch_num = 0, prev_batch = None,**fit_params):
        """Compute y_pred, loss value, and update net's gradients.
        The module is set to be in train mode (e.g. dropout is
        applied).
        Parameters
        ----------
        batch
          A single batch returned by the data loader.
        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.
        Returns
        -------
        step : dict
          A dictionary ``{'loss': loss, 'y_pred': y_pred}``, where the
          float ``loss`` is the result of the loss function and
          ``y_pred`` the prediction generated by the PyTorch module.
        """
        self._set_training(True)
        Xi, yi = unpack_data(batch)

        Xi[1] = Xi[1].clone().detach().requires_grad_(True) 
        y_pred = self.infer(Xi, **fit_params)
        weight = None
        if self.simple_model is not None:
            simple_batch = [x.item() for x in Xi]
            Xi_simple = sp.sparse.vstack([self.visits_data_simple[p] for p in simple_batch], format='csr')
            weight = torch.Tensor(self.simple_model.predict_proba(Xi_simple)[:, yi][0])
            #simple_batch = [[x.item() for x in x_list] for x_list in batch]
        # for name, params in self.module.named_parameters():
        #     params.register_hook(lambda gradient: torch.nn.functional.normalize(gradient) * 10e-1 if len(gradient.shape) > 1 else gradient)
        loss = self.get_loss(y_pred, yi.to(torch.float32), X=Xi, training=True, weight = weight, 
                             epoch_num = epoch_num, prev_batch = prev_batch)
        loss.backward()

        #Taken from https://github.com/PKUnlp-icler/ChildTuning/blob/main/ChildTuningD.py
        if not hyper_params.IS_PRETRAINING and hyper_params.OPTIM_TYPE == 'ChildTuning-D':# and self.optimizer_.have_gradient_mask():
          gradient_mask = dict()
          for name, params in self.module.named_parameters():
              if 'bert' in name and params.grad is not None:
                  gradient_mask[params] = params.new_zeros(params.size()).detach().cpu()

          for name, params in self.module.named_parameters():
            if 'bert' in name and params.grad is not None:
              torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
              gradient_mask[params] += (params.grad.detach().cpu() ** 2) / len(self.dataset)

          #Calculate Fisher Info
          r = None
          for k, v in gradient_mask.items():
              v = v.view(-1).cpu().numpy()
              if r is None:
                  r = v
              else:
                  r = np.append(r, v)
          polar = np.percentile(r, (1-self.reserve_p)*100)
          for k in gradient_mask:
              gradient_mask[k] = (gradient_mask[k] >= polar).detach().cpu()

          if hyper_params.OPTIM_TYPE == 'ChildTuning-D':
            self.optimizer_.set_gradient_mask(gradient_mask)
          del gradient_mask

        else:
          for name, params in self.module.named_parameters():
            if ('bert' in name) and params.grad is not None:
              torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

        del batch, Xi[1]
        # gc.collect()
        # torch.cuda.empty_cache()
        #print(torch.cuda.memory_summary())
        return {
            'loss': loss,
            'y_pred': y_pred,
        }

    def get_loss(self, y_pred, y_true, X=None, training=False, weight = None, epoch_num = 0, prev_batch = None):
        """Return the loss for this batch.

        Parameters
        ----------
        y_pred : torch tensor
          Predicted target values

        y_true : torch tensor
          True target values.

        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        training : bool (default=False)
          Whether train mode should be used or not.

        """
        y_true = to_tensor(y_true, device="cpu")
        if hyper_params.USE_MSE_LOSS == False:
           if isinstance(y_pred, tuple):
              return self.criterion_(y_pred[0].cpu(), y_true)
           return self.criterion_(y_pred.cpu(), y_true)
        return self.criterion_(y_pred[0].cpu(), y_true) +  (0.25 * torch.nn.MSELoss()(y_pred[1].cpu(), X[1]) if not hyper_params.NOT_USE_MSE_LOSS == True else 0)

    def get_all_learnable_params(self):
        seen = set()
        for name in self._modules + self._criteria:
            module = getattr(self, name + '_')
            named_parameters = getattr(module, 'named_parameters', None)
            if not named_parameters:
                continue

            for param_name, param in named_parameters():
                if param.requires_grad == False:
                  continue 

                if param in seen:
                    continue

                seen.add(param)
                yield param_name, param

    def run_single_epoch(self, dataset, training, prefix, step_fn, **fit_params):
        """Compute a single epoch of train or validation.

        Parameters
        ----------
        dataset : torch Dataset or None
          The initialized dataset to loop over. If None, skip this step.

        training : bool
          Whether to set the module to train mode or not.

        prefix : str
          Prefix to use when saving to the history.

        step_fn : callable
          Function to call for each batch.

        **fit_params : dict
          Additional parameters passed to the ``step_fn``.
        """
        if dataset is None:
            return

        batch_count = 0
        if self.use_sampler and training:
          if 'dataset' in dir(dataset):
            y_vals = np.array(dataset.dataset.y)[dataset.indices]
          else:
            y_vals = np.array(dataset.y)
          class_weights = np.bincount(y_vals)
          labels_weights =  1./class_weights
          labels_weights[1] *= 2
          weights = labels_weights[y_vals]
          if 'weights' in dataset.dataset_dict:
            weights = [w * dataset.dataset_dict['weights'][p[1]] for w, p in zip(weights, dataset.X)]
          sampler = WeightedRandomSampler(weights, len(weights) * hyper_params.SAMPLE_FACTOR_TRAINING)
          self.__dict__['iterator_train__sampler'] = sampler
        else:
          self.__dict__['iterator_train__sampler'] = None
        #t = None
        prev_batch = None
        epoch_num = 0
        for batch in self.get_iterator(dataset, training=training):
          gc.collect()
          torch.cuda.empty_cache()
          #if t is not None:
            #print("Passed time: ", time.time() - t)
          if training:
            num_persons_removed = 0
            Xi, yi = unpack_data(batch)
            for i in range(Xi[1].shape[0]):
              curr_i = i - num_persons_removed
              if torch.all(Xi[1][curr_i, :, :] == 0).item():
                Xi[0] = torch.cat((Xi[0][0: curr_i], Xi[0][curr_i + 1:]), dim = 0).detach()
                Xi[1] = torch.cat((Xi[1][0: curr_i, :, :], Xi[1][curr_i + 1:, :, :]), dim = 0).detach()
                if len(Xi) > 2:
                  Xi[2] = torch.cat((Xi[2][0: curr_i, :], Xi[2][curr_i + 1:, :]), dim = 0).detach()
                yi = torch.cat((yi[0: curr_i], yi[curr_i + 1:]), dim = 0).detach()
                num_persons_removed += 1
            if len(Xi[0]) == 0:
              continue
            batch = (Xi, yi)

          self.notify("on_batch_begin", batch=batch, training=training)
          fit_params['epoch_num'] = epoch_num
          fit_params['prev_batch'] = prev_batch
          step = step_fn(batch, **fit_params)
          self.history.record_batch(prefix + "_loss", step["loss"].item())
          batch_size = (get_len(batch[0]) if isinstance(batch, (tuple, list))
                        else get_len(batch))
          self.history.record_batch(prefix + "_batch_size", batch_size)
          self.notify("on_batch_end", batch=batch, training=training, **step)
          batch_count += 1
          prev_batch = (unpack_data(batch)[1], step['y_pred'])
          epoch_num += 1
          # gc.collect()
          # torch.cuda.empty_cache()
          #t = time.time()

        self.history.record(prefix + "_batch_count", batch_count)


    def evaluation_step(self, batch, training=False):
        """Perform a forward step to produce the output used for
        prediction and scoring.

        Therefore, the module is set to evaluation mode by default
        beforehand which can be overridden to re-enable features
        like dropout by setting ``training=True``.

        Parameters
        ----------
        batch
          A single batch returned by the data loader.

        training : bool (default=False)
          Whether to set the module to train mode or not.

        Returns
        -------
        y_infer
          The prediction generated by the module.

        """
        self.check_is_fitted()
        Xi, _ = unpack_data(batch)
        with torch.set_grad_enabled(training):
            self._set_training(training)
            if len(Xi) == 3 or len(Xi[0]) != 2:
              return self.infer(Xi)[1] if not hyper_params.ZERO_VISITS else self.infer(Xi) #
            if type(Xi[0]) == torch.Tensor:
              return self.infer(Xi)
            return self.infer((Xi[0][1], Xi[1]))