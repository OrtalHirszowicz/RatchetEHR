
import torch
from torch.nn import RNN, LSTM
import hyper_params

class VisitSimple(torch.nn.Module):
    def __init__(
        self, #featureSet,
        task_name, 
        dropout=0.4,
        num_features = 1,
        hidden_size = -1,
        hidden_layers = hyper_params.HIDDEN_LAYERS,
        **kwargs
    ):
        super(VisitSimple, self).__init__()
        
        self.data_set = False
        
        self.dropout = dropout
        
        self.max_visits = hyper_params.MAX_VISITS # max_visits
        

        self.num_features = num_features
        self.embedding_dim = self.num_features
        
        if hidden_size != -1:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = self.embedding_dim

        self.task_name = task_name
        if hyper_params.USE_RNN or hyper_params.USE_LSTM:
            self.layers_num_neurons = [ hyper_params.RNN_HIDDEN_SIZE] + hidden_layers + [2]
        else:
            self.layers_num_neurons = None
        self.hidden_layers = hidden_layers
        

    def init_linear_layers(self):
        if hyper_params.USE_RNN:
            MODEL = RNN
        elif hyper_params.USE_LSTM:
            MODEL = LSTM
        self.rnn_1 = MODEL(input_size = self.hidden_size, hidden_size = hyper_params.RNN_HIDDEN_SIZE, 
                num_layers= hyper_params.RNN_LAYERS, batch_first = True)
        self.norm_1 = torch.nn.BatchNorm1d(hyper_params.RNN_HIDDEN_SIZE)
        # self.dropout = torch.nn.Dropout(self.dropout)
        self.rnn_2 = MODEL(input_size = hyper_params.RNN_HIDDEN_SIZE, hidden_size = hyper_params.RNN_HIDDEN_SIZE, 
                num_layers= hyper_params.RNN_LAYERS, batch_first = True)
        self.norm_2 = torch.nn.BatchNorm1d(hyper_params.RNN_HIDDEN_SIZE)
        self.linear =  torch.nn.Linear(hyper_params.RNN_HIDDEN_SIZE, 2)
        
                
    def set_data(
        self, data_set_dict
    ):
        self.n_visits = data_set_dict['n_visits']
        if 'visits_data' in data_set_dict:
            self.visits_data = data_set_dict['visits_data']
        else:
            self.visits_data = None
        self.max_values = 0 #data_set_dict['max_values']
        self.num_non_variant_features = 0
        self.features_to_remove = None
        if 'features_to_remove' in data_set_dict:
            self.features_to_remove = data_set_dict['features_to_remove']
        if 'num_invariant_features' in data_set_dict:
            self.num_non_variant_features = data_set_dict['num_invariant_features']
        self.hidden_size += self.num_non_variant_features
        self.one_hot_size = self.visits_data.get_example()[2].shape[1]
        self.hidden_size += self.one_hot_size
        if 'not_good_features' in data_set_dict:
            self.hidden_size -= len(data_set_dict['not_good_features'])
        self.data_set = True
        if self.layers_num_neurons is None:
            self.layers_num_neurons = [self.hidden_size] + self.hidden_layers + [2]
        self.init_linear_layers()

    def change_n_visits(self, n_visits):
        self.n_visits = n_visits

    def set_person_range(self, person_range):
        self.person_range = person_range.cpu().detach().numpy()


    def forward(self, x, train=True, return_mask=False):

        assert(self.data_set)
        output = self.rnn_1(x)[0].transpose(1, 2)
        output = self.norm_1(output).transpose(1, 2)
        output = self.rnn_2(output)[0][:, -1, :].squeeze(dim = 0)
        output = self.norm_2(output)
        output = torch.nn.functional.softmax(self.linear(output), dim = -1)

        return output
