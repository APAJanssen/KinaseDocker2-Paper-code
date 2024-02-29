'''
Script that generates the ML table for the ECFP data.

NOTE: It's very important that the ML table that you use lines up with the ECFPs. 
So each line number in the ML table should correspond to the line number in the ECFPs .npy file. 
This is incorporated in the ML-table generation script. Therefore, generate the ML-table after the ECFPs.
'''
import pandas as pd
import db
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing as mp
from sys import argv
from itertools import repeat
import random as rdm


DATABASE = db.setup('PATH_TO_DATABASE')


def split_activities(activities, test_size=0.2, random_state=69):
    train, test = train_test_split(activities, test_size=test_size, random_state=random_state)
    return train, test


def get_ecfp_indices(df):
    ecfp_table = pd.read_csv('PATH_TO_ECFP.csv') # Table with columns: InChIKey
    ecfp_table_dict = dict(zip(ecfp_table['InChIKey'], ecfp_table.index))

    print('Finished loading ECFP data')
    print(f'Retrieving indices...')

    df['ECFP_index'] = df['InChIKey'].apply(lambda x: ecfp_table_dict[x])

    return df


if __name__ == '__main__':
    OUT_FILE = f'ML_table_ECFP.csv'

    # filter on accession having at least 100 activities
    accessions = DATABASE.query('SELECT accession FROM activities GROUP BY accession HAVING COUNT(*) >= 100')

    results = DATABASE.query(f'SELECT activity_ID, accession, activities.InChIKey, pchembl_value_Mean FROM activities INNER JOIN compounds ON activities.InChIKey = compounds.InChIKey WHERE accession IN {tuple(accessions)}')

    df = pd.DataFrame(results, columns=['activity_ID', 'accession', 'InChIKey', 'pIC50'])
    df = df.sort_values('accession').reset_index(drop=True)
    df_grouped = df.groupby('accession')

    for name, group in tqdm(df_grouped):
        train, test = split_activities(group['activity_ID'].values.tolist())

        df.loc[df['activity_ID'].isin(train), 'partition'] = 'train'
        df.loc[df['activity_ID'].isin(test), 'partition'] = 'test' 

    final_data = get_ecfp_indices(df)
    final_data.to_csv(OUT_FILE, index=False)
