import torch
import math
from Models.MyTransformerEncoderLayer import MyTransformerEncoderLayer
import config 
import pickle
import gc
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import hyper_params
from torch import nn
import numpy as np
from einops import rearrange

class NumericalEmbedderBackwards(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        return torch.einsum('aij,ij->ai', x, self.weights)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)[:, :pe[:, 0, 1::2].shape[1]]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

#From https://github.com/lucidrains/tab-transformer-pytorch/blob/3a38072f9c384bee9fc4428763d4aded78139c44/tab_transformer_pytorch/ft_transformer.py#L91
class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x.reshape(-1, x.shape[-1]), 'b n -> b n 1')
        return x * self.weights + self.biases

class NumLastVisits():
    def get(self):
        return hyper_params.NUM_LAST_VISITS


class TransposeLayer(nn.Module):
    def __init__(self, dim1, dim2):
        self.dim1 = dim1
        self.dim2 = dim2
        super().__init__()
        #self._modules = {}

    def forward(self, x):
        return torch.transpose(x, self.dim1, self.dim2)
    

class VisitTransformer(torch.nn.Module):
    def __init__(
        self, #featureSet,
        task_name = 'eicu_' + hyper_params.CURR_TASK + '_' + str(hyper_params.NUM_MESUREMENTS) + '_' +str(hyper_params.NUM_HOURS_FOR_WINDOW) +'h', 
        n_heads=1, attn_depth=2,
        dropout=0.4,
        use_mask=True,
        backwards_attn=False,
        index_embedding=False,
        num_features = 1,
        max_visits = hyper_params.MAX_VISITS,
        hidden_size = -1,
        num_transformer_blocks_to_freeze = 0,
        tf_connection_hidden_size = 100,
        feature_attn_depth = None,
        feature_dropout = None,
        **kwargs
    ):
        super(VisitTransformer, self).__init__()
        
        self.data_set = False
        
        self.n_heads = n_heads
        self.dropout = dropout
        
        self.max_visits = hyper_params.MAX_VISITS # max_visits
        self.attn_depth = attn_depth
        print(self.attn_depth)
        
        self.use_mask = use_mask
        self.backwards_attn = backwards_attn
        self.index_embedding  = index_embedding

        self.num_features = num_features
        self.embedding_dim = self.num_features
        
        self.conv_layer = None
        if hidden_size != -1:
            self.conv_layer =  torch.nn.Conv1d(in_channels = self.embedding_dim, 
                                out_channels = hidden_size, kernel_size = 1)
            self.hidden_size = hidden_size
        else:
            self.hidden_size = self.embedding_dim

        self.task_name = task_name
        
        self.tf_connection_hidden_size = tf_connection_hidden_size
        self.num_transformer_blocks_to_freeze = num_transformer_blocks_to_freeze
        #self.norm_layer = torch.nn.LayerNorm(self.max_visits)
            
        if not self.use_mask:
            self.start_embedding = torch.nn.Parameter(torch.randn(self.embedding_dim))
            self.pad_embedding = torch.nn.Parameter(torch.zeros(self.embedding_dim))
            self.mask_embedding = torch.nn.Parameter(torch.randn(self.embedding_dim))
        else:
            self.pad_embedding = torch.zeros(self.embedding_dim).cuda()

        self.feature_attn_depth = feature_attn_depth
        self.feature_dropout = feature_dropout

        
                
    def set_data(
        self, data_set_dict
    ):
        self.person_indices = data_set_dict['person_indices']
        self.outcomes_filt = data_set_dict['outcomes_filt']
        self.idx_to_person = data_set_dict['idx_to_person']
        self.dataset_dict = data_set_dict
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
        self.hidden_size -= len(data_set_dict['not_good_features'])
        curr_example = self.visits_data.get_example()
        if len(curr_example) > 2:
            self.one_hot_size = self.visits_data.get_example()[2].shape[1]
            self.hidden_size += self.one_hot_size
        self.k = self.hidden_size
        self.tfs = TransformerEncoder(encoder_layer= 
        MyTransformerEncoderLayer(d_model = self.k, nhead=self.n_heads,  dim_feedforward=200,
        batch_first= True, dropout=self.dropout, norm_first = True) , 
        num_layers=self.attn_depth)
        for name, param in self.tfs.named_parameters():
            if 'weight' in name and 'norm' not in name:
                torch.nn.init.xavier_normal_(param)
            elif 'bias' in name and 'norm' not in name:
                torch.nn.init.constant_(param, 0)
        self.pos_encoder = PositionalEncoding(self.k)
        self.cls_embedding = torch.nn.Embedding(1, self.k)
        torch.nn.init.normal_(self.cls_embedding.weight, std = 0.02)
        
        self.norm_layer = torch.nn.LayerNorm(self.hidden_size)
        self.data_set = True

        if hyper_params.SHOULD_USE_GCT_COMPONENT:
            self.transform_size = hyper_params.FT_FEATURE_SIZE
            self.feature_transform = NumericalEmbedder(dim = self.transform_size, num_numerical_types=self.hidden_size)
            self.pos_encoder_feature = PositionalEncoding(self.transform_size)
            self.feature_batch_norm = torch.nn.BatchNorm1d(self.hidden_size)
            self.feature_transforms = torch.nn.Sequential(*[torch.nn.Linear(1, self.transform_size, bias = False).cuda() for i in range(self.hidden_size)])
            self.feature_detransforms = torch.nn.Sequential(*[torch.nn.Linear(self.transform_size, 1, bias = False).cuda() for i in range(self.hidden_size)])
            self.transformer_encoder_features = TransformerEncoder(encoder_layer=  MyTransformerEncoderLayer(d_model = self.transform_size, nhead = self.n_heads, dim_feedforward=self.transform_size // 4,
            batch_first = True, dropout = self.feature_dropout, norm_first = True) , 
            num_layers=self.feature_attn_depth)
            self.reconstruction_layer = NumericalEmbedderBackwards(self.transform_size, self.hidden_size)
            for name, param in self.transformer_encoder_features.named_parameters():
                if 'weight' in name and 'norm' not in name:
                    torch.nn.init.xavier_normal_(param)
                elif 'bias' in name and 'norm' not in name:
                    torch.nn.init.constant_(param, 0)

    def set_max_values(self, max_values, max_values_invariant):
        self.max_values = max_values
        self.max_value_invariant = max_values_invariant

    def get_dict_path(self, person_id):
        return config.DEFAULT_SAVE_LOC + '/Dictionary' + self.task_name +'/' + str(person_id)  + "_transformer"
        
    def set_person_range(self, person_range):
        self.person_range = person_range.cpu().detach().numpy()

    def get_person_data(self, p):
        dict_file = self.get_dict_path(p)
        with open(dict_file, 'rb') as pickle_file:
            t = pickle.load(pickle_file)
        return t

    def gct_component_forward(self, output_emb):
        if hyper_params.SHOULD_USE_GCT_COMPONENT:
            orig_output = output_emb
            #output_emb = self.feature_batch_norm(output_emb.transpose(1, 2)).transpose(1, 2)
            output_emb = self.feature_transform(output_emb)
            output_emb = self.transformer_encoder_features(output_emb)
            output_emb = self.reconstruction_layer(output_emb).reshape(-1, hyper_params.MAX_VISITS,self.hidden_size) + orig_output
            torch.cuda.empty_cache()
        return output_emb
        
    def forward(self, x, train=True, return_mask=False):
        
        #curr_time = time.time()
        use_mask = self.use_mask

        #x = x / x.max(0, keepdim = True)[0]

        assert(self.data_set)

        output_emb = x        
        #output_emb = torch.matmul(output_emb.reshape(-1, self.hidden_size), V[:, :self.k]).reshape(-1, hyper_params.MAX_VISITS,self.k)
        output_rev = None
        if hyper_params.SHOULD_USE_GCT_COMPONENT:
            output_emb = self.gct_component_forward(output_emb)
            output_rev = output_emb
            if hyper_params.TRAIN_FEATURE_TRANSFORMER:
                return output_emb, None, output_rev
        if use_mask:
            mask = torch.empty(size = (x.shape[0], x.shape[1]), dtype = torch.bool).fill_(False).cuda() #TODO: Ortal - change it to zeros!
            mask_features = torch.empty_like(x, dtype = torch.bool).fill_(False)
            if self.features_to_remove is not None:
                output_emb[:, :, list(self.features_to_remove)] = 0

        min_vals = [min(self.n_visits[p], self.max_visits) for p in self.person_range]
        if use_mask:
            for i, min_val in enumerate(min_vals):
                mask[i, min_val: ] = True

        if not hyper_params.IS_PRETRAINING:
            output_emb = torch.cat([self.cls_embedding(torch.tensor([0]).cuda()).repeat(output_emb.shape[0], 1).unsqueeze(1), output_emb], dim = 1)
        
        #output_emb = self.norm_layer(output_emb)
        output_emb = self.pos_encoder(output_emb.transpose(0, 1)).transpose(0, 1) #+ time_embedding


        persons_without_data = torch.all(mask, dim = 1)
        mask = (torch.Tensor(np.concatenate((np.expand_dims(np.full((mask.shape[0]),False), 1), mask.detach().cpu().numpy()), axis = 1)) == 1).cuda()
        mask[persons_without_data, :] = torch.empty(size = (torch.sum(persons_without_data == True).item(), x.shape[1] +  1), dtype = torch.bool).fill_(True).cuda()
        if use_mask:
            output_emb = self.tfs(src = output_emb) #, src_key_padding_mask = mask)
        else:
            output_emb = self.tfs(output_emb)
            
        if torch.sum(persons_without_data == True).item() > 0:
            output_emb[persons_without_data, :, :] = torch.empty(size = (torch.sum(persons_without_data == True).item(), output_emb.shape[1], output_emb.shape[2])).fill_(0).cuda()
                
        # gc.collect()
        # torch.cuda.empty_cache()

             
        if return_mask:
            return output_emb, mask, output_rev
        else:
            return output_emb, output_rev
