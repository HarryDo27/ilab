import torch
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import LungCancerDataset
from sklearn.model_selection import StratifiedKFold
import warnings
import os
import os.path as osp

warnings.filterwarnings("ignore")

def sopuid_to_recurrence(df):
    sopuid_to_rec = {}
    for _, row in df.iterrows():
        sopuid = row['SOPInstanceUID']
        
        if sopuid in sopuid_to_rec:
            continue
        
        sopuid_to_rec[sopuid] = row['Recurrence']
    
    return sopuid_to_rec

def mkdir_if_missing(path):
    if not osp.exists(path):
        print(f'Make dir at {path}')
        os.makedirs(path)


def split_data_client_wise(train_csv, val_csv, test_csv, img_dir, n_clients, save_dir):
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    train_sopuids = df_train['SOPInstanceUID'].unique()
    val_sopuids = df_val['SOPInstanceUID'].unique()

    train_labels_dict = sopuid_to_recurrence(df_train)
    val_labels_dict = sopuid_to_recurrence(df_val)

    train_labels = [train_labels_dict[sopuid] for sopuid in train_sopuids]
    val_labels = [val_labels_dict[sopuid] for sopuid in val_sopuids]

    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=42)

    client_train_sopuids = []
    client_val_sopuids = []

    for _, sample_indices in skf.split(train_sopuids, train_labels):
        client_train_sopuids.append(train_sopuids[sample_indices])

    for _, sample_indices in skf.split(val_sopuids, val_labels):
        client_val_sopuids.append(val_sopuids[sample_indices])
    
    # SOPUID to Client ID
    sopuids_to_cids = {}

    for idx, sopuids in enumerate(client_train_sopuids):
        for id in sopuids:
            sopuids_to_cids[id] = idx+1
    
    for idx, sopuids in enumerate(client_val_sopuids):
        for id in sopuids:
            sopuids_to_cids[id] = idx+1

    df_train['client_id'] = df_train['SOPInstanceUID'].map(sopuids_to_cids)
    df_val['client_id'] = df_val['SOPInstanceUID'].map(sopuids_to_cids)

    for id in range(1, n_clients+1):
        df_train_client = df_train[df_train.client_id == id].drop(columns=['client_id'])
        df_val_client = df_val[df_val.client_id == id].drop(columns=['client_id'])

        # save to csv
        df_train_client.to_csv(osp.join(save_dir, f'client_{id}_train.csv'), index=False)
        df_val_client.to_csv(osp.join(save_dir, f'client_{id}_val.csv'), index=False)

    return

if __name__ == '__main__':
    data_client_root = 'dataset/data_clients'
    train_csv = 'dataset/train.csv'
    test_csv = 'dataset/test.csv'
    val_csv = 'dataset/val.csv'
    img_dir = 'dataset/ROI'

    mkdir_if_missing(data_client_root)

    split_data_client_wise(train_csv, val_csv, test_csv, img_dir, n_clients=3, save_dir=data_client_root)