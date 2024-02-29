'''
Script that handles PLEC data input for the DNN.

NOTE: It's very important that the ML table that you use lines up with the PLECs. So each line number in the ML table should correspond to the line number in the PLECs .npy file.
'''
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tqdm import tqdm


class CustomDatasetDB(Dataset):
    '''
    This class can be used to reproduce results from the paper
    '''
    def __init__(self, partition, ml_table_filename, docking_type):
        self.partition = partition

        self.filename_csv = ml_table_filename

        self.poses, self.labels, self.plec_indices = self.load_data()
        self.x = np.load(f'DNN_data/db_plecs_{docking_type}.npy', mmap_mode='r')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return_list = [self.x[self.plec_indices[idx]].astype(np.float32), np.array([self.labels[idx]]).astype(np.float32)]
            
        if self.partition == 'test':
            return_list.append(self.poses[idx])

        return return_list

    def load_data(self):
        data = pd.read_csv('DNN_data/' + self.filename_csv)
        data = data[data['partition'] == self.partition]
        data = self.norm_data(data)

        return data['pose_ID'].tolist(), data['pIC50'].tolist(), data['PLEC_index'].tolist()

    def norm_data(self, data):
        data['pIC50'] = data['pIC50'].apply(lambda x: (x - 3)/(12 - 3))

        return data

class CustomDatasetGenerate(Dataset):
    '''
    This class can be used to test separate datasets, here the PLEC fingerprints are generated inplace. (NOTE: This can be very slow!)
    '''
    def __init__(self, partition, depth_ligand=1, depth_protein=5, size=65536, count_bits=True, sparse=True):
        # Setup input generation
        plec_func = partial(PLEC,
                        depth_ligand=depth_ligand,
                        depth_protein=depth_protein,
                        size=size,
                        count_bits=count_bits,
                        sparse=sparse,
                        ignore_hoh=True)

        self.descriptors = universal_descriptor(plec_func, shape=size, sparse=sparse)
        
        self.partition = partition
        self.poses, self.x, self.labels = self.load_plecs()
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return_list = [np.array(self.x[idx]).astype(np.float32), np.array([self.labels[idx]]).astype(np.float32)]
            
        if self.partition == 'test':
            return_list.append(self.poses[idx])

        return return_list

    def load_plecs(self):
        data = pd.read_csv('DNN_data/ML_table.csv')
        data = self.norm_data(data)
        data = data.loc[data['partition'] == self.partition]
        data.sort_values('klifs_ID', inplace=True)

        all_poses = []
        all_x = []
        all_labels = []

        print('Generating PLECs...')

        for klifs_ID in tqdm(data['klifs_ID'].unique()):
            # Get ligands
            molblocks = data.loc[data['klifs_ID'] == klifs_ID]['molfile'].tolist()
            mols = [oddt.toolkit.readstring('mol', molblock) for molblock in molblocks]
        
            # Add hydrogens
            for mol in mols:
                mol.addh()

            # Get protein
            protein = next(oddt.toolkit.readfile('pdb', f'DNN_data/pdb/{klifs_ID}.pdb'))
            protein.protein = True

            # Get poses
            poses = data.loc[data['klifs_ID'] == klifs_ID]['pose_ID'].tolist()

            # Generate input
            x = self.descriptors.build(mols, protein)

            # Get labels
            labels = data.loc[data['klifs_ID'] == klifs_ID]['pIC50'].tolist()

            # Append to lists
            all_poses.extend(poses)
            all_x.extend(x)
            all_labels.extend(labels)

        return all_poses, all_x, list(map(float, all_labels))

    def norm_data(self, data):
        #data.loc[data['pIC50'] < 5] = 5.
        #data.loc[data['pIC50'] > 10] = 10.

        data['pIC50'] = data['pIC50'].apply(lambda x: (x - 3)/(12 - 3))

        return data

