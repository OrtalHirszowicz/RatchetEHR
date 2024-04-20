from toolz.dicttoolz import valmap
import torch

def postprocess_tensor(t, max_visits):
    curr_t = t[0].coalesce()
    dense_tensor = torch.full((max_visits, curr_t.shape[1]), fill_value = float('nan'))
    indices = curr_t.indices()
    dense_tensor[indices[0, :], indices[1, :]] = curr_t.values()
    dense_tensor = torch.nanmedian(dense_tensor, dim = 0)[0]
    dense_tensor = torch.nan_to_num(dense_tensor)
    return (dense_tensor, t[1])


def post_process(visits_data):
    max_visits = max([data[0].shape[0] for data in visits_data.values()] + [data[1].shape[0] for data in visits_data.values()]) + 10
    visits_data = valmap(lambda x: postprocess_tensor(x, max_visits), visits_data)
    return visits_data
    
