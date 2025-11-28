import torch
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
from tqdm import tqdm


def create_sequences(data, divided_date, input_len=336, pred_len=24):
    total_samples = len(data) - input_len - pred_len
    train_size = sum(1 for i in range(total_samples) if data[i+input_len][0] < divided_date)
    valid_size = total_samples - train_size

    X_train, Y_train = np.empty((train_size, 336, 7)), np.empty((train_size, 24))
    X_valid, Y_valid = np.empty((valid_size, 336, 7)), np.empty((valid_size, 24))
    valid_idx = 0
    train_idx = 0

    for i in tqdm(range(len(data)-input_len-pred_len)):
        sample = data[i:i+input_len][:, 1:].reshape(1, 336, 7)
        target = data[i+input_len:i+input_len+pred_len][:, -1].reshape(1, -1)
        
        if data[i+input_len][0] >= divided_date:
            X_valid[valid_idx] = sample
            Y_valid[valid_idx] = target
            valid_idx += 1
        else:
            X_train[train_idx] = sample
            Y_train[train_idx] = target
            train_idx += 1

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    Y_valid = torch.tensor(Y_valid, dtype=torch.float32)

    return X_train, Y_train, X_valid, Y_valid


def get_dataloader(divided_date='2017.07.01 00:00:00'):
    data = pd.read_csv('data/zscore_data.csv')
    data['date'] = pd.to_datetime(data['date'])

    divided_date = pd.to_datetime(divided_date)
    end_date = divided_date + pd.DateOffset(months=1, days=1)

    data = data[data['date'] <= end_date].values
    print('Creating data...')
    train_features, train_target, valid_features, valid_target = create_sequences(data, divided_date)
    print('Finish!')
    train_dataset = TensorDataset(train_features, train_target)
    valid_dataset = TensorDataset(valid_features, valid_target)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, valid_loader
