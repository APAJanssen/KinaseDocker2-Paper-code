'''
Script that generates the ECFPs.

NOTE: It's very important that the ML table that you use lines up with the ECFPs. 
So each line number in the ML table should correspond to the line number in the ECFPs .npy file. 
This is incorporated in the ML-table generation script. Therefore, generate the ML-table after the ECFPs.
'''
import oddt
from oddt import fingerprints
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import numpy as np
from npy_append_array import NpyAppendArray
from oddt.fingerprints import ECFP
from itertools import repeat
import os
import db


DATABASE = db.setup('PATH_TO_DATABASE')


def gen_ecfp_mp(d, mol_info, depth=2, size=2048, count_bits=False, sparse=False):
    inchikey, smiles = mol_info

    mol = oddt.toolkit.readstring('smi', smiles)
    mol.addh()

    d[inchikey] = ECFP(mol, depth=depth, size=size, count_bits=count_bits, sparse=sparse)

    pbar.update()


if __name__ == '__main__':
    OUT_FILE = 'db_ecfps'

    results = DATABASE.query('SELECT activities.InChIKey, SMILES, pchembl_value_Mean FROM activities \
                            INNER JOIN Compounds ON activities.InChIKey = Compounds.InChIKey')

    df = pd.DataFrame(results, columns=['InChIKey', 'SMILES', 'pIC50'])
    df.drop_duplicates('InChIKey', inplace=True)

    mols = df[['InChIKey', 'SMILES']].drop_duplicates().values.tolist()

    pbar = tqdm(total=len(mols))

    n_cores = 30

    with mp.Manager() as manager:
        plec_dict = manager.dict()

        with mp.Pool(n_cores) as pool:
            pool.starmap(gen_ecfp_mp, zip(repeat(plec_dict), mols))

            print('Retrieving pIC50s...')

            data_chunk = df.set_index('InChIKey').to_dict(orient='index')
            
            targets = []

            for inchikey in tqdm(list(plec_dict.keys())):
                targets.append(data_chunk[inchikey]['pIC50'])

            final_df = pd.DataFrame({'InChIKey': list(plec_dict.keys()), 'pIC50': targets})
            final_df.to_csv(os.path.join(OUT_FILE + '.csv'), mode='w', index=False)

            print('Saving PLECs...')

            np.save(os.path.join(OUT_FILE + '.npy'), np.vstack(plec_dict.values()))