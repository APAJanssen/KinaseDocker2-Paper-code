'''
Script to generate PLECs for the dataset.

NOTE: It's very important that the ML table that you use lines up with the PLECs. 
So each line number in the ML table should correspond to the line number in the PLECs .npy file. 
This is incorporated in the ML-table generation script. Therefore, generate the ML-table after the PLECs.
'''
import oddt
from oddt import fingerprints
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import numpy as np
from npy_append_array import NpyAppendArray
from oddt.fingerprints import PLEC
from itertools import repeat
import os


def gen_plec_mp(d, data_chunk, klifs_ID, depth_ligand=1, depth_protein=5, size=65536, count_bits=False, sparse=False):
    # Get ligands
    molblocks = data_chunk.loc[data_chunk['klifs_ID'] == klifs_ID]['molfile'].tolist()
    poses = data_chunk.loc[data_chunk['klifs_ID'] == klifs_ID]['pose_ID'].tolist()
    mols = [oddt.toolkit.readstring('mol', molblock) for molblock in molblocks]

    # Add hydrogens
    for mol in mols:
        mol.addh()

    # Get protein
    protein = next(oddt.toolkit.readfile('pdb', 'PATH_TO_KLIFS_PDB_FILE'))
    protein.protein = True

    for pose_id, mol in zip(poses, mols):
        plec = PLEC(mol, protein, depth_ligand=depth_ligand, depth_protein=depth_protein, size=size, count_bits=count_bits, sparse=sparse)

        d[pose_id] = plec

    pbar.update()


if __name__ == '__main__':
    FOLDER = 'vina'
    OUT_FILE = 'db_plecs_vina'
    chunks = pd.read_csv('db_vina_to_plec.csv', chunksize=100000) # table with columns: pose_ID, klifs_ID, molfile, pIC50

    for i, data_chunk in enumerate(chunks):
        # Generate PLECs
        print(f'Generating PLECs... (Chunk: {i})')

        data_chunk = data_chunk.sort_values('klifs_ID')
        
        pbar = tqdm(total=len(data_chunk['klifs_ID'].unique()))

        n_cores = 30

        with mp.Manager() as manager:
            plec_dict = manager.dict()

            with mp.Pool(n_cores) as pool:
                pool.starmap(gen_plec_mp, zip(repeat(plec_dict), repeat(data_chunk), data_chunk['klifs_ID'].unique()))

            print('Retrieving pIC50s...')

            data_chunk = data_chunk.set_index('pose_ID').to_dict(orient='index')
            targets = []

            for pose_ID in tqdm(list(plec_dict.keys())):
                targets.append(data_chunk[pose_ID]['pIC50'])

            df = pd.DataFrame({'pose_ID': list(plec_dict.keys()), 'pIC50': targets})

            if os.path.exists(os.path.join(FOLDER, OUT_FILE + '.csv')):
                df.to_csv(os.path.join(FOLDER, OUT_FILE + '.csv'), mode='a', index=False, header=False)
            else:
                df.to_csv(os.path.join(FOLDER, OUT_FILE + '.csv'), mode='w', index=False)

            print('Saving PLECs...')

            with NpyAppendArray(os.path.join(FOLDER, OUT_FILE + '.npy')) as naa:
                naa.append(np.vstack(plec_dict.values()))
