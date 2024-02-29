'''
Script that handles ECFP data input for the DNN
'''
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, partition, ml_table_filename, accession):
        self.partition = partition
        self.accession = accession

        self.filename_csv = ml_table_filename

        self.poses, self.labels, self.plec_indices = self.load_data()
        self.x = np.load(f'DNN_data/db_ecfps.npy', mmap_mode='r')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return_list = [self.x[self.plec_indices[idx]].astype(np.float32), np.array([self.labels[idx]]).astype(np.float32)]
            
        if self.partition == 'test':
            return_list.append(self.poses[idx])

        return return_list

    def load_data(self):
        data = pd.read_csv('DNN_data/' + self.filename_csv)
        data = data[(data['partition'] == self.partition) & (data['accession'] == self.accession)]
        data = self.norm_data(data)

        return data['activity_ID'].tolist(), data['pIC50'].tolist(), data['ECFP_index'].tolist()

    def norm_data(self, data):
        data['pIC50'] = data['pIC50'].apply(lambda x: (x - 3)/(12 - 3))

        return data

