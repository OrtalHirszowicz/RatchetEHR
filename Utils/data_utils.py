import numpy as np
from sklearn.model_selection import train_test_split
import torch
import gc
import pandas as pd

def get_y (X, curr_cohort):
    return [curr_cohort[curr_cohort['example_id'] == int(idx[1])]['y'].values[0] for idx in X]

def update_summary_statistics(summary_statistics, step = 0, feature_set_info = None):
    #From https://github.com/MLforHealth/MIMIC_Extract/blob/master/resources/variable_ranges.csv
    ranges_df = pd.read_csv('Tables/variable_ranges.csv')
    ranges_df['VALID LOW'] = ranges_df['OUTLIER LOW'] / (1 - step)
    ranges_df['VALID HIGH'] = ranges_df['OUTLIER HIGH'] / (1 + step)
    
    converter_dict = feature_set_info.numeric_feature_to_index
    
    converter_names = {
        'Alanine aminotransferase': ['\tALT'],
        'Albumin': ['Albumin'],
        'Alkaline phosphate': ['ALP'],
        'Anion Gap': ['Anion Gap'],
        'Asparate aminotransferase': ['AST'],
        'Bicarbonate': ['HCO3'],
        'Bilirubin': ['Bilirubin'],
        'Blood urea nitrogen': ['BUN'],
        'Chloride': ['Chloride'],
        'Creatinine': ['Creatinine'], 
        'Diastolic blood pressure': ['NBP [Diastolic]'],
        'Glascow coma scale total': ['GCS Total'],
        'Glucose': ['Glucose'],
        'Hematocrit': ['Hematocrit'],
        'Hemoglobin': ['Hemoglobin'],
        'Lactate': ['Lactate'],
        'Lactate dehydrogenase': ['LD'],
        'Magnesium': ['Magnesium', 'Magnesium'],
        'Mean blood pressure': ['NBP Mean'],
        'Oxygen saturation': ['Oxygen Saturation'],
        'Partial thromboplastin time': ['PTT'],
        'Peak inspiratory pressure': ['PIP'],
        'pH': ['pH'],
        'Phosphate': ['Phosphate'],
        'Platelets': ['PLT'],
        'Potassium': ['Potassium'],
        'Prothrombin time': ['PT'],
        'Sodium': ['Sodium'],
        'Systolic blood pressure': ['NBP [Systolic]'], 
        'Weight': ['Daily Weight'],
        'White blood cell count': ['WBC']
    }

    for index, row in ranges_df.iterrows():
        for curr_name in converter_names[row['LEVEL2']]:
            if curr_name in converter_dict:
                summary_statistics[converter_dict[curr_name]] = {'25%_0': row['VALID LOW'], '25%_1': row['VALID LOW'], 
                                    '75%_0': row['VALID HIGH'], '75%_1': row['VALID HIGH'] }
    
    temp_summary_statistics = {}
    #From https://github.com/Anat12345/ICU-acquired-BSI-prediction-model/blob/master/create_signal_data.py
    temp_summary_statistics['Heart Rate'] = {'25%_0': (35 / (1 - step)), '25%_1': (35 / (1 - step)), '75%_0': (210 / (1+step)), '75%_1': (210 / (1+step))}
    temp_summary_statistics['Heart Rate'] = {'25%_0': (35 / (1 - step)), '25%_1': (35 / (1 - step)), '75%_0': (210 / (1+step)), '75%_1': (210 / (1+step))}
    temp_summary_statistics['Temperature C (calc)'] = {'25%_0': (34.5 / (1 - step)), '25%_1': (34.5 / (1 - step)), '75%_0': (45 / (1+step)), '75%_1': (45 / (1+step))}
    temp_summary_statistics['Temperature'] = {'25%_0': (34.5 / (1 - step)), '25%_1': (34.5 / (1 - step)), '75%_0': (45 / (1+step)), '75%_1': (45 / (1+step))}
    temp_summary_statistics['Temperature F'] = {'25%_0': (93.2 / (1 - step)), '25%_1': (93.2 / (1 - step)), '75%_0': (113 / (1+step)), '75%_1': (113 / (1+step))}
    temp_summary_statistics['Arterial BP [Diastolic]'] = {'25%_0': (20 / (1 - step)), '25%_1': (20 / (1 - step)), '75%_0': (310 / (1+step)), '75%_1': (310 / (1+step))}
    temp_summary_statistics['Arterial BP [Systolic]'] = {'25%_0': (30 / (1 - step)), '25%_1': (30 / (1 - step)), '75%_0': (350 / (1+step)), '75%_1': (350 / (1+step))}
    temp_summary_statistics['Respiratory Rate'] = {'25%_0': (4 / (1 - step)), '25%_1': (4 / (1 - step)), '75%_0': (80 / (1+step)), '75%_1': (80 / (1+step))}
    
    for feature_name, val in temp_summary_statistics.items():
        if feature_name in converter_dict:
            summary_statistics[converter_dict[feature_name]] = val

    return summary_statistics

def clean_data_person(curr_t, summary_statistics, n_visits, person_id, step = 0.5):
    k = person_id
    for j in range(curr_t.shape[1]):
        if j not in summary_statistics:
            continue
        curr_t[:n_visits[k], j][torch.logical_or(torch.logical_and((curr_t[:n_visits[k], j] < summary_statistics[j]['25%_0'] * (1 - step)), (curr_t[:n_visits[k], j] != 0.0)) , 
        torch.logical_and((curr_t[:n_visits[k], j] < summary_statistics[j]['25%_1'] * (1 - step)), (curr_t[:n_visits[k], j] != 0.0)))] = float('nan')
        curr_t[:n_visits[k], j][torch.logical_or(torch.logical_and((curr_t[:n_visits[k], j] > summary_statistics[j]['75%_0'] * (1 + step)), (curr_t[:n_visits[k], j] != 0.0)), 
        torch.logical_and((curr_t[:n_visits[k], j] > summary_statistics[j]['75%_1'] * (1 + step)), (curr_t[:n_visits[k], j] != 0.0)))] = float('nan')
    for feature_idx in range(curr_t.shape[1]):
        curr_signal = curr_t[:n_visits[k], feature_idx]
        if (curr_signal != 0).nonzero().shape[0] == 0 and torch.any(torch.isnan(curr_signal)).item() == False:
            continue
        first_non_zero_idx = (curr_signal != 0).nonzero()[0].item()
        curr_signal[:first_non_zero_idx] = torch.empty(size = (first_non_zero_idx, )).fill_(float('nan'))
        curr_signal = pd.DataFrame(curr_signal)
        curr_signal = curr_signal.set_index(pd.date_range(start='1/1/2000', periods=curr_signal.size, freq='2H'))
        curr_signal_interpolated = curr_signal.interpolate(method = 'time', limit_direction = 'both')
        curr_t[:n_visits[k], feature_idx] = torch.from_numpy(curr_signal_interpolated.values).squeeze(dim = 1)
    return curr_t

def clean_data(visits_data, X_train, y_train, feature_set_info, n_visits, step = 0.5):
    t_all = [visits_data[p_id][0] for p_id in X_train]
    y = torch.cat([torch.Tensor([val] * visits_data[p_id][0].shape[0]) for p_id, val in zip(X_train, y_train)])
    t_all = torch.cat(t_all)
    features = list(range(t_all.shape[1]))
    t_all = torch.cat((t_all, y.unsqueeze(dim = 1)), dim = 1)
    df = pd.DataFrame(t_all.numpy(), columns = features + ['y'])
    masked_df = df
    masked_df[features] = df[features].mask(df[features] == 0)
    df_y_1 = masked_df[masked_df['y'] == 1]
    df_y_0 = masked_df[masked_df['y'] == 0]
    summary_statistics = {}
    # for k in masked_df.columns:
    #     curr_0 = df_y_0[k].describe()
    #     curr_1 = df_y_1[k].describe()
    #     summary_statistics[k] = {'25%_0': curr_0['25%'], '75%_0': curr_0['75%'], '25%_1': curr_1['25%'], '75%_1': curr_1['75%']}
    summary_statistics = update_summary_statistics(summary_statistics, step, feature_set_info)
    for k, t in visits_data.items():
        curr_t = t[0]
        for j in range(curr_t.shape[1]):
            if j not in summary_statistics:
                continue
            curr_t[:n_visits[k], j][torch.logical_or(torch.logical_and((curr_t[:n_visits[k], j] < summary_statistics[j]['25%_0'] * (1 - step)), (curr_t[:n_visits[k], j] != 0.0)) , 
            torch.logical_and((curr_t[:n_visits[k], j] < summary_statistics[j]['25%_1'] * (1 - step)), (curr_t[:n_visits[k], j] != 0.0)))] = float('nan')
            curr_t[:n_visits[k], j][torch.logical_or(torch.logical_and((curr_t[:n_visits[k], j] > summary_statistics[j]['75%_0'] * (1 + step)), (curr_t[:n_visits[k], j] != 0.0)), 
            torch.logical_and((curr_t[:n_visits[k], j] > summary_statistics[j]['75%_1'] * (1 + step)), (curr_t[:n_visits[k], j] != 0.0)))] = float('nan')
        for feature_idx in range(curr_t.shape[1]):
            curr_signal = pd.DataFrame(curr_t[:n_visits[k], feature_idx])
            curr_signal_interpolated = curr_signal.interpolate()
            curr_t[:n_visits[k], feature_idx] = torch.from_numpy(curr_signal_interpolated.values).squeeze(dim = 1)
        visits_data[k] = (curr_t, t[1].to_dense(), t[2])

    return visits_data

def select_features(visits_data, X_train):
    t = [torch.cat((visits_data[p][0][:visits_data[p][2].shape[0]], visits_data[p][1].repeat(visits_data[p][2].shape[0], 1), visits_data[p][2]), dim = 1) for p in X_train]
    corr_df = pd.DataFrame(torch.cat(t, dim = 0).numpy())
    corr = corr_df.corr()
    correlated_indices = np.where(np.nan_to_num(corr.values) > 0.8)
    good_features = list(range(t[0].shape[1]))
    features_to_remove = set()
    for i, j in zip(list(correlated_indices[0]), list(correlated_indices[1])):
        if i != j and i in good_features:
            features_to_remove.add(i)
    return features_to_remove

def get_data(visits_data, person_indices, dataset_dict, 
                test_val_precentage, validation_precentage, 
                max_visits, n_visits, curr_cohort, featureSetInfo = None, fix_imbalance = False, need_to_clean_data = False, source_visits_data = None):
    orig_X = sorted(list(person_indices), key =  lambda x: x[1])
    X = orig_X
    if 'is_last_years' in dataset_dict:
        X = [x for x, is_last_year in zip(orig_X, dataset_dict['is_last_years']) if is_last_year == 0]
    y = get_y(X, curr_cohort)
    X_train, X_val_test = train_test_split(X,
        test_size = test_val_precentage,  stratify =y
    )
    y_test = get_y(X_val_test, curr_cohort)
    X_val, X_test = train_test_split(X_val_test, test_size = validation_precentage, stratify = y_test) #, random_state=42)
    y_train = get_y(X_train, curr_cohort)
    y_val = get_y(X_val, curr_cohort)
    y_test = get_y(X_test, curr_cohort)
    if 'is_last_years' in dataset_dict:
        X_val += X_test
        y_val += y_test
        X_test = [x for x, is_last_year in zip(orig_X, dataset_dict['is_last_years']) if is_last_year == 1]
        y_test = get_y(X_test, curr_cohort)
    if need_to_clean_data and (source_visits_data is not None):
        source_visits_data = clean_data(source_visits_data, [x[1] for x  in X_train], y_train, featureSetInfo, dataset_dict['n_visits'])
        features_to_remove = select_features(source_visits_data, [x[1] for x  in X_train])
        dataset_dict['features_to_remove'] = features_to_remove
    if source_visits_data is not None:
        visits_data.set_visits_data(source_visits_data)
    
    dataset_dict['n_visits'] = n_visits
    
    #print(len(X_train))
    gc.collect()
    torch.cuda.empty_cache()
    
    return X_train, y_train, X_val, y_val, X_test, y_test, dataset_dict
