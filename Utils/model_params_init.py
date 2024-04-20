import hyper_params

def get_model_params(
        embedding_dim, 
        n_heads, 
        featureSetInfo,
        attn_depth = hyper_params.ATTN_DEPTH, 
        feature_attn_depth = hyper_params.FEATURE_ATTN_DEPTH, 
        activation_layer = hyper_params.ACTIVATION_LAYER,
        activation_layer_params = hyper_params.ACTIVATION_LAYER_PARAMS,
        dropout = hyper_params.DROPOUT,
        xavier_gain = hyper_params.XAVIER_GAIN,
        feature_dropout = hyper_params.FEATURE_DROPOUT,
        ):
    model_params = {
        'embedding_dim': int(embedding_dim / n_heads), # Dimension per head of visit embeddings
        'n_heads': n_heads, # Number of self-attention heads
        'attn_depth': attn_depth, #4, # Number of stacked self-attention layers
        'feature_attn_depth': feature_attn_depth,
        'activation_layer': activation_layer,
        'activation_layer_params': activation_layer_params,
        'dropout': dropout,  # Dropout rate for both self-attention and the final prediction layer
        'xavier_gain': xavier_gain,
        'feature_dropout': feature_dropout,
        'use_mask': True, # Only allow visits to attend to other actual visits, not to padding visits
        'num_features': len(featureSetInfo.numeric_feature_to_index),
        'n_targets' : 2, #Binary Classifier model
        'use_probas' : True,
        'n_parallel_pools' : 10,
        'tf_connection_hidden_size': 300,
        'optimizer__bert_lr': hyper_params.BERT_LR
        #'hidden_size' : 200
    }
    return {hyper_params.MODEL_PARAMS_PREFIX + k: v for k, v in model_params.items()}