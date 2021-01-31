from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

def data_set_inverse_transform(data_frame, scaler_1, scaler_2, labels):
    data_set_rev = scaler_1.inverse_transform(scaler_2.inverse_transform(data_frame))
    data_set_rev_df = DataFrame(data_set_rev)
    data_set_rev_df.columns = data_frame.columns
    data_set_rev_df["pdg"] = labels

    return data_set_rev_df

def remove_momentum_from_data(data_array):
    data_array_p = data_array[:, 0]
    data_array_without_p = np.delete(data_array, 0, 1)

    return data_array_without_p, data_array_p

def binarize_labels(labels_array, pdg_code):
    labels_array_copy = labels_array.copy()
    labels_array[labels_array_copy == pdg_code] = 1
    labels_array[labels_array_copy != pdg_code] = 0

    return labels_array

def compute_weights(labels_array):
    len_labels_array = len(labels_array[labels_array == 0])
    len_0_label_frac = len_labels_array/len(labels_array[labels_array == 0])
    len_1_label_frac = len_labels_array/len(labels_array[labels_array == 1])

    return [len_0_label_frac, len_1_label_frac]

def data_loader(features_array, labels_array, batch_size=600, weights=None):
    pid_data_set = PIDDataSet(features_array, labels_array, weights)
    data_loader = DataLoader(pid_data_set, batch_size)

    return data_loader


class PIDDataSet(Dataset):
    
    def __init__(self, features, labels, weights):
        self.features = features
        self.labels = labels
        self.weights = weights
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.features[idx, :], self.labels[idx], self.weights[idx])

        return sample