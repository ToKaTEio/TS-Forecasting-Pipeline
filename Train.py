import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import yaml
import pickle
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from Dataloader import get_dataloader

from model.DLinear import DLinear
from model.GRU import GRU
from model.TCN import TCN
from model.Transformer import Transformer


def create_log():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join('log', now)
    os.makedirs(os.path.join(log_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'curve'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'loss'), exist_ok=True)
    return log_dir


def save_config(log_dir, config):
    config_path = os.path.join(log_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, leave=False, dynamic_ncols=True)
    for batch in pbar:
        optimizer.zero_grad()
        features, labels = batch[0].to(device), batch[1].to(device)
        assert not torch.isnan(features).any()
        outputs = model(features)
        assert not torch.isnan(outputs).any()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    pred = np.empty((len(dataloader.dataset),))
    pointer = 0
    pbar = tqdm(dataloader, leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            features, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(features)
            total_loss += criterion(outputs, labels).item()
            pred[pointer:pointer+labels.shape[0]] = outputs.cpu().numpy()[:, 0]
            pointer += labels.shape[0]
    return total_loss / len(dataloader.dataset), np.array(pred)


def plot_loss(train_loss_history, val_loss_history, model_name, log_dir, seed, date):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {model_name}_seed{seed}_window{date}')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(log_dir, 'curve/{model_name}_seed{seed}_window{date}_loss_curve.png')
    plt.savefig(plot_path)
    plt.close()


def train(config):
    log_dir = create_log()
    save_config(log_dir, config)
    config['log_dir'] = log_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for seed in config["seeds"]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        dates = pd.date_range(
            start="2017-07-01", 
            end="2018-06-01", 
            freq="MS"
        )

        timestamps = [date.strftime("%Y.%m.%d %H:%M:%S") for date in dates]

        for i, date in enumerate(timestamps):

            train_loader, valid_loader = get_dataloader(date)

            for model_name in config['model_list']:
                if model_name == 'GRU':
                    model = GRU(**config['GRU']).to(device)
                elif model_name == 'TCN':
                    model = TCN(**config['TCN']).to(device)

                elif model_name == 'DLinear':
                    model = DLinear(**config['DLinear']).to(device)
                elif model_name == 'Transformer':
                    model = Transformer(**config['Transformer']).to(device)
                else:
                    raise NotImplementedError

                optimizer = torch.optim.Adam(model.parameters(), lr=config["initial_lr"])
                criterion = nn.MSELoss()

                best_loss = float('inf')
                early_stop_counter = 0
                train_loss_history, val_loss_history = [], []
                best_pred = None
                print(f'Model{model_name}, seed{seed}, divided date {date}Train start!!!')
                print()

                for epoch in range(config["max_epochs"]):
                    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                    val_loss, pred = validate(model, valid_loader, criterion, device)

                    train_loss_history.append(train_loss)
                    val_loss_history.append(val_loss)
                    
                    print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                    # Early Stop
                    if val_loss < best_loss:
                        best_loss = val_loss
                        early_stop_counter = 0
                        model_path = os.path.join(config['log_dir'], f'model/best_{model_name}_seed{seed}_window{i}.pth')
                        pred_path = os.path.join(config['log_dir'], f'pred/best_{model_name}_seed{seed}_pred_window{i}.npz')
                        best_pred = pred
                        np.savez(pred_path, pred=best_pred)
                        torch.save(model.state_dict(), model_path)
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= config["early_stop_patience"]:
                            print(f"Early stopping at epoch {epoch+1}")
                            break

                loss_path = os.path.join(config['log_dir'], f'/loss/{model_name}_seed{seed}_window{i}_train_losses')
                with open(loss_path, 'wb') as f:
                    pickle.dump(train_loss_history, f)

                plot_loss(train_loss_history, val_loss_history, model_name, log_dir, seed, i)


if __name__ == "__main__":

    config = {
    "initial_lr": 1e-3,
    "max_epochs": 50,
    "early_stop_patience": 7,
    "seeds": [42],
    "model_list": ['DLinear', 'TCN'],
    "GRU": {
            'input_size': 7, 
            'hidden_size': 128, 
            'output_size': 24, 
            'num_layers': 2
        },
    "Transformer": {
            'd_model': 128, 
            'nhead': 4, 
            'num_layers': 2, 
            'output_size': 24,
            'input_dim': 7
        },
    "DLinear": {
            'input_len': 336,
            'pred_len': 24,
            'num_features': 7
        },
    "TCN": {
            'num_channels': [64, 128, 256, 128, 64],
            'kernel_size': 3,
            'input_size': 7,
            'output_size': 24
        }
    }

    train(config)