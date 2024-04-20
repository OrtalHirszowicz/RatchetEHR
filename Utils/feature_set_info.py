from Utils.transformer_utils import post_process

class FeatureSetInfo:
    def __init__(self, featureSet, task_name, save_icd_codes = True):
        self.person_ids = featureSet.person_ids
        self.unique_id_col = featureSet.unique_id_col
        self.window_times_for_person = featureSet.window_times_for_person
        self.num_numeric_features = featureSet.num_numeric_features
        self.num_non_numeric_features = featureSet.num_non_numeric_features
        self.numeric_feature_to_index = featureSet.numeric_feature_to_index
        if save_icd_codes and hasattr(featureSet, "feature_codes_to_id"):
            self.feature_codes_to_id = featureSet.feature_codes_to_id
        post_process(featureSet.person_ids, task_name)