import torch
import hyper_params
from Models.visit_transformer import NumLastVisits
import torch.nn as nn
class VTClassifer(torch.nn.Module):
    def __init__(
        self, bert_model,
        n_parallel_pools = 10,
        n_targets=2,
        hidden_layers = hyper_params.HIDDEN_LAYERS,
        is_pretraining = hyper_params.IS_PRETRAINING,
        should_freeze_cnn = False,
        return_raw_data = False,
        should_return_visit_embedding = False,
        kernel_size_visits = 5,
        add_cnn_layer = hyper_params.USE_CNN,
        activation_layer = None,
        activation_layer_params = None,
        xavier_gain = None,
        **kwargs
    ):
        super(VTClassifer, self).__init__()
        self.n_targets = n_targets
        self.emb_size = bert_model.hidden_size 
        self.bert = bert_model
        self.n_parallel_pools = n_parallel_pools
        #self.norm_layer_1 = torch.nn.BatchNorm1d(bert_model.hidden_size )
        self.is_pretraining = is_pretraining
        self.should_return_visit_embedding = should_return_visit_embedding
        self.hidden_size = 100
        self.kernel_size_visits = kernel_size_visits
        self.num_last_visits = NumLastVisits().get()
        self.reconstruction_layer = torch.nn.Linear(self.emb_size, self.emb_size)
        # if hyper_params.SHOULD_USE_GCT_COMPONENT:
        #     self.reconstruction_layer = NumericalEmbedderBackwards(self.bert.transform_size, self.bert.hidden_size)
        self.layers_num_neurons = [ self.bert.k * self.num_last_visits ] + hidden_layers + [n_targets]
        self.add_cnn_layer = add_cnn_layer
        self.activation_layer = activation_layer
        self.activation_layer_params = activation_layer_params
        self.xavier_gain = xavier_gain

        if self.add_cnn_layer:

            self.convs = torch.nn.Conv1d(
                self.emb_size, self.emb_size * self.n_parallel_pools, self.kernel_size_visits
            )
            self.maxpool = torch.nn.MaxPool1d(hyper_params.MAX_VISITS - self.kernel_size_visits)

            self.layers_num_neurons = [ self.emb_size * self.n_parallel_pools ] + hidden_layers + [n_targets]

        layers = [torch.nn.BatchNorm1d(self.bert.k)]
        converter_to_activation_layer = {"gelu":torch.nn.GELU, "tanh": torch.nn.Tanh, "leakyrelu": torch.nn.LeakyReLU, "elu": torch.nn.ELU}
        for i, (first, second) in enumerate(zip(self.layers_num_neurons[:-1],self.layers_num_neurons[1: ])):
            if i == len(self.layers_num_neurons) - 2:
                new_layers = [
                    torch.nn.Linear(first, second),
                ]
                torch.nn.init.constant_(new_layers[-1].bias, 0)
                torch.nn.init.xavier_normal_(new_layers[-1].weight, gain = self.xavier_gain) #, mode = 'fan_in')
            else:
                new_layers = [
                    torch.nn.Dropout(bert_model.dropout),
                    torch.nn.Linear(first, second),
                    torch.nn.BatchNorm1d(second),
                    converter_to_activation_layer[self.activation_layer.lower()](*self.activation_layer_params),#torch.nn.LeakyReLU(0.05),
                ]
                torch.nn.init.constant_(new_layers[-3].bias, 0)
                torch.nn.init.xavier_normal_(new_layers[-3].weight, gain = self.xavier_gain) #, mode = 'fan_in')
            
            layers += new_layers
        self.linear_layers = torch.nn.Sequential(*layers)
        self.dropout = torch.nn.Dropout(bert_model.dropout)
        self.soft_max = torch.nn.Softmax(dim = 1)
        self.return_raw_data = return_raw_data

    def set_max_values(self, max_values):
        self.max_values = max_values 

    def set_person_range(self, person_range):
        self.bert.set_person_range(person_range)

    def set_visits_zeroed(self, visits_zeroed):
        self.visits_zeroed = visits_zeroed

    def get_probas(self, x, interpret_debug=False):
        if not torch.is_tensor(x):
            self.set_person_range(x[0])
            x = x[1]
        x, _, output_rev = self.bert(x, train=False, return_mask = True)
        if x.shape[1] < hyper_params.MAX_VISITS:
            x = torch.nn.functional.pad(x, pad = (0, 0, 0, hyper_params.MAX_VISITS - x.shape[1], 0, 0))
        if output_rev is not None:
            with torch.no_grad():
                norm_tensor_std = torch.Tensor(self.bert.dataset_dict['norm_info']['std']).cuda().unsqueeze(0).unsqueeze(0).repeat(output_rev.shape[0], output_rev.shape[1], 1).cuda()
                norm_tensor_mean = torch.Tensor(self.bert.dataset_dict['norm_info']['mean']).cuda().unsqueeze(0).unsqueeze(0).repeat(output_rev.shape[0], output_rev.shape[1], 1).cuda()
                norm_tensor_std = torch.where(norm_tensor_std != 0.0, norm_tensor_std, 1)
                norm_tensor_mean = torch.where(norm_tensor_std != 0.0, norm_tensor_mean, 0)
                for i, p in enumerate(self.bert.person_range):
                    norm_tensor_mean[i, self.bert.n_visits[p]:, :] = 0
                    norm_tensor_std[i, self.bert.n_visits[p]:, :] = 1
                output_rev *= norm_tensor_std
                output_rev += norm_tensor_mean

        if self.is_pretraining:
            x = self.reconstruction_layer(x).reshape(-1, hyper_params.MAX_VISITS,self.bert.hidden_size)
            with torch.no_grad():
                norm_tensor_std = torch.Tensor(self.bert.dataset_dict['norm_info']['std']).cuda().unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1).cuda()
                norm_tensor_mean = torch.Tensor(self.bert.dataset_dict['norm_info']['mean']).cuda().unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1).cuda()
                norm_tensor_std = torch.where(norm_tensor_std != 0.0, norm_tensor_std, 1)
                norm_tensor_mean = torch.where(norm_tensor_std != 0.0, norm_tensor_mean, 0)
                for i, p in enumerate(self.bert.person_range):
                    norm_tensor_mean[i, self.bert.n_visits[p]:, :] = 0
                    norm_tensor_std[i, self.bert.n_visits[p]:, :] = 1
                x *= norm_tensor_std
                x += norm_tensor_mean
            return x, output_rev
        # if self.add_cnn_layer:
        #     cv = self.convs(x.transpose(1,2))
        #     cv_masked = cv.masked_fill(
        #         mask[:, :-self.kernel_size_visits + 1].unsqueeze(1).expand(
        #                 -1, self.emb_size * self.n_parallel_pools, -1
        #             ) == 0,
        #             -1e9
        #         )
        #     pooled = self.maxpool(cv_masked)
        #     y_pred = self.linear_layers(pooled.squeeze())
        #     if self.n_targets == 1:
        #         return y_pred.flatten(0, -1)
        #     if len(y_pred.shape) == 1:
        #         y_pred = y_pred.unsqueeze(dim = 0)
        #     y_pred = self.soft_max(y_pred)
        #     return y_pred
            
        
        x = x[:, 0, :]
        if self.return_raw_data:
            return x
        y_pred = self.linear_layers(x)
        if self.n_targets == 1:
            return y_pred.flatten(0, -1)
        y_pred = self.soft_max(y_pred)

        if output_rev is None:
            return y_pred

        return y_pred, output_rev

    def change_n_visits(self, n_visits):
        self.bert.n_visits = n_visits

    def forward(self, x, interpret_debug=False):
        y_pred = self.get_probas(x, interpret_debug=interpret_debug)
        return y_pred

    def predict_probas(self, x, interpret_debug = False):
        return self.get_probas(x, interpret_debug = interpret_debug)[:, 1]

    def predict(self, x, interpret_debug = False):
        return torch.argmax(self.predict_probas(x, interpret_debug=interpret_debug), dim = 1)

    def get_params_values(self):
        weights = []
        for curr_layer in self.linear_layers:
            if isinstance(curr_layer, torch.nn.Linear):
                weights.append(curr_layer.weight)
        return weights

