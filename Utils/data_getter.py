import config
import _pickle as pickle
import glob
import hyper_params

class DataGetter:
    def __init__(self, task_name):
        self.task_name = task_name
        self.visits_data = None
        self.example_id = -1
        self.postfix =   hyper_params.DATA_GETTER_POSTFIX

    def set_visits_data(self, visits_data):
        self.visits_data = visits_data

    def get_dict_path_empty(self, i = 0):
        return config.DEFAULT_SAVE_LOC + '/Dictionary' + self.task_name[i] +'/'

    def get_dict_path(self, person_id):
        return self.get_dict_path_empty(person_id[0]) + str(person_id[1])

    def get_data_transformer(self, person_id):
        dict_file = self.get_dict_path(person_id) + self.postfix #'_postprocessed_tensor'
        with open(dict_file, 'rb') as pickle_file:
            val = pickle.load(pickle_file)
        if len(val) > 2:
            return (val[0].to_dense(), val[1], val[2]) 
        return (val[0].to_dense(), val[1]) #val

    def get_example(self, dataset_idx = 0):
        if self.visits_data is not None:
            return self.visits_data[list(self.visits_data.keys())[0]]
        if self.example_id == -1:
            begin = self.get_dict_path_empty()
            ending = self.postfix
            self.example_id = [int(x[len(begin):-len(ending)]) for x in glob.glob(begin + '*' + ending)][0]
        return self.get_data_transformer((dataset_idx, self.example_id))

    def __getitem__(self, person_index):
        if self.visits_data is not None:
            if person_index[1] not in self.visits_data:
                print("Here")
            return self.visits_data[person_index[1]]
        return self.get_data_transformer(person_index)