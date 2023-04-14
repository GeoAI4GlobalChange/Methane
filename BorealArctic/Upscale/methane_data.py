import pandas as pd
import numpy as np
import random
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
import pickle
def fluxnetData4ML(dir_site_data,target_variable,input_cols,path_site_info,seed,site_types,depth,prediction_horizon,test_precent):
    dir=dir_site_data
    files = os.listdir(dir)
    files = files[1:]
    target = target_variable
    target_cols = input_cols
    train_datasets = {}  # Initialize the training datasets
    val_datasets = {}  # Initialize the validation datasets
    test_datasets = {}  # Initialize the testing datasets
    site_type_dict = {}  # Initialize the list of wetland types for each site
    df = pd.read_csv(
        path_site_info)  # load data of site information，and the 'site_location_v2.csv' is located in the same folder with this py file.
    df = df[['Name', 'Type']].values
    for idx in range(df.shape[0]):
        site_type_dict[df[idx, 0]] = df[idx, 1]  # obtain wetland type for each site
    site_start = {}  # to identify whether data has been loaded or not in a wetland type
    for item in site_types:
        site_start[item] = True
    for file in files:
        print(file)
        temp_type = site_type_dict[file.split('_')[0]]  # get wetland type for each site according to its name
        if temp_type in site_types:
            file_path = dir + file
            data = pd.read_csv(file_path)
            site_final_cols = target_cols
            # obtain datasets
            length = data.shape[0]
            data1 = data
            X_train1 = np.zeros((len(data1), depth, len(site_final_cols)))
            for i, name in enumerate(site_final_cols):
                for j in range(depth):
                    X_train1[:, j, i] = data1[name].shift(depth - j - 1)
            if prediction_horizon > 0:
                y_train1 = np.array(data1[target].shift(-prediction_horizon))
                X_train1 = X_train1[(depth - 1):-prediction_horizon]
                y_train1 = y_train1[(depth - 1):-prediction_horizon]
            else:
                y_train1 = np.array(data1[target])
                X_train1 = X_train1[(depth - 1):]
                y_train1 = y_train1[(depth - 1):]
            # remove samples with nan or extreme values
            y_nan_mask = np.isnan(y_train1) | np.isinf(y_train1)
            y_extreme_mask = np.abs(y_train1) > pow(10, 10)
            y_nan_mask = (y_nan_mask | y_extreme_mask)
            x_temp = X_train1.reshape((X_train1.shape[0], -1)).copy()
            x_nan_mask = np.any(np.isnan(x_temp), axis=1) | np.any(np.isinf(x_temp), axis=1)
            nan_mask = (y_nan_mask | x_nan_mask)
            if np.sum(nan_mask == False) > 0:
                X_train1 = X_train1[nan_mask == False]
                y_train1 = y_train1[nan_mask == False]
                all_idxs = [item_idx for item_idx in range(y_train1.shape[0])]
                random.seed(seed)
                test_idxs = random.sample(all_idxs, int(test_precent * len(all_idxs)))
                train_validate_idxs = list(set(all_idxs).difference(set(test_idxs)))
                random.seed(seed)
                validate_idxs = random.sample(train_validate_idxs, int(1 / 9 * len(train_validate_idxs)))
                train_idxs = np.array(list(set(train_validate_idxs).difference(set(validate_idxs))))
                test_idxs = np.array(test_idxs)
                validate_idxs = np.array(validate_idxs)
                if site_start[temp_type]:
                    train_datasets[temp_type] = {}
                    train_datasets[temp_type]['X'] = X_train1[train_idxs]
                    train_datasets[temp_type]['Y'] = y_train1[train_idxs]
                    test_datasets[temp_type] = {}
                    test_datasets[temp_type]['X'] = X_train1[test_idxs]
                    test_datasets[temp_type]['Y'] = y_train1[test_idxs]
                    val_datasets[temp_type] = {}
                    val_datasets[temp_type]['X'] = X_train1[validate_idxs]
                    val_datasets[temp_type]['Y'] = y_train1[validate_idxs]
                    site_start[temp_type] = False
                else:
                    train_datasets[temp_type]['X'] = np.concatenate(
                        [train_datasets[temp_type]['X'], X_train1[train_idxs]], axis=0)
                    train_datasets[temp_type]['Y'] = np.concatenate(
                        [train_datasets[temp_type]['Y'], y_train1[train_idxs]], axis=0)
                    test_datasets[temp_type]['X'] = np.concatenate([test_datasets[temp_type]['X'], X_train1[test_idxs]],
                                                                   axis=0)
                    test_datasets[temp_type]['Y'] = np.concatenate([test_datasets[temp_type]['Y'], y_train1[test_idxs]],
                                                                   axis=0)
                    val_datasets[temp_type]['X'] = np.concatenate(
                        [val_datasets[temp_type]['X'], X_train1[validate_idxs]], axis=0)
                    val_datasets[temp_type]['Y'] = np.concatenate(
                        [val_datasets[temp_type]['Y'], y_train1[validate_idxs]], axis=0)
    return train_datasets,val_datasets,test_datasets
def get_causal_strength():
    return
def Data4ML_tensor(wetland_type,train_datasets,val_datasets,test_datasets,chamber_input,chamber_output,batch_size,seed,test_precent):
    type=wetland_type
    x_chamber = chamber_input[type]
    y_chamber = chamber_output[type]
    x_chamber = np.vstack(x_chamber)
    y_chamber = np.vstack(y_chamber)[:, 0]
    x_train = train_datasets[type]['X']
    y_train = train_datasets[type]['Y']
    print('number of trained samples:', y_train.shape[0])
    x_test = test_datasets[type]['X']
    y_test = test_datasets[type]['Y']
    x_validate = val_datasets[type]['X']
    y_validate = val_datasets[type]['Y']
    x_all = np.concatenate((x_train, x_validate, x_test), axis=0)  # concatenate: combine together along axis 0
    y_all = np.concatenate((y_train, y_validate, y_test), axis=0)

    ###########combine eddy covariance datasets with chamber datasets
    x_all = np.concatenate((x_all, x_chamber), axis=0)
    y_all = np.concatenate((y_all, y_chamber), axis=0)

    x_means = np.nanmean(x_all.reshape((-1, x_all.shape[2])), axis=0)
    x_std = np.nanstd(x_all.reshape((-1, x_all.shape[2])), axis=0)
    y_means = np.nanmean(y_all, axis=0)
    y_std = np.nanstd(y_all, axis=0)

    x_train = (x_train - x_means) / (x_std + pow(10, -6))
    y_train = (y_train - y_means) / (y_std + pow(10, -6))
    x_test = (x_test - x_means) / (x_std + pow(10, -6))
    y_test = (y_test - y_means) / (y_std + pow(10, -6))
    x_validate = (x_validate - x_means) / (x_std + pow(10, -6))
    y_validate = (y_validate - y_means) / (y_std + pow(10, -6))

    x_train = torch.Tensor(x_train)
    x_test = torch.Tensor(x_test)
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)
    x_validate = torch.Tensor(x_validate)
    y_validate = torch.Tensor(y_validate)

    x_chamber = chamber_input[type]
    y_chamber = chamber_output[type]

    chamber_all_idxs = [chamber_idx for chamber_idx in range(len(x_chamber))]
    random.seed(seed)
    chamber_test_idxs = random.sample(chamber_all_idxs, int(test_precent * len(chamber_all_idxs)))
    chamber_train_validate_idxs = list(set(chamber_all_idxs).difference(set(chamber_test_idxs)))
    random.seed(seed)
    chamber_validate_idxs = random.sample(chamber_train_validate_idxs,
                                          int(1 / 9 * len(chamber_train_validate_idxs)))
    chamber_train_idxs = np.array(list(set(chamber_train_validate_idxs).difference(set(chamber_validate_idxs))))
    chamber_test_idxs = np.array(chamber_test_idxs)
    chamber_validate_idxs = np.array(chamber_validate_idxs)


    train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(x_validate, y_validate), shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(x_test, y_test), shuffle=False, batch_size=batch_size)
    return train_loader,val_loader,test_loader,\
           chamber_train_idxs,chamber_validate_idxs,chamber_test_idxs,\
           x_means,x_std,y_means,y_std


def get_causal_strength(data_source,depth,dir_causal_stren,site_types):
    causal_dict_1 = {}
    for temp_type in site_types:
        causal_dict_1[temp_type] = np.load(dir_causal_stren + f'{data_source}_{temp_type}_{depth}.npy',
                                           allow_pickle=True).astype(np.float32)
    return causal_dict_1

def chamberData4ML(chamber_dir,data_source,depth):
    with open(f"{chamber_dir}{data_source}_input_chamber_{depth}.pkl", "rb") as tf:
        chamber_input = pickle.load(tf)
    with open(f"{chamber_dir}{data_source}_output_chamber_{depth}.pkl", "rb") as tf:
        chamber_output = pickle.load(tf)
    with open(f"{chamber_dir}{data_source}_ID_chamber_{depth}.pkl", "rb") as tf:
        chamber_ID = pickle.load(tf)
    return chamber_input,chamber_output,chamber_ID