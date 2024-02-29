'''
Script that generates ML tables.

NOTE: It's very important that the ML table that you use lines up with the PLECs. 
So each line number in the ML table should correspond to the line number in the PLECs .npy file. 
This is incorporated in the ML-table generation script. Therefore, generate the ML-table after the PLECs.
'''
import pandas as pd
import db
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing as mp
from sys import argv
from itertools import repeat


DATABASE = db.setup('PATH_TO_DATABASE')


def split_activities(activities, test_size=0.2, random_state=42):
    train, test = train_test_split(activities, test_size=test_size, random_state=random_state)
    return train, test

def split_activities_compounds(df, test_size=0.2):
    '''
    df with columns: activity_ID, InChIKey
    '''
    num_test = int(len(df['activity_ID'].unique()) * test_size)

    grouped_df = df.groupby('InChIKey')

    # get groups until num_test is reached
    test_df = pd.DataFrame(columns=df.columns)

    for name, group in grouped_df:
        if len(test_df['activity_ID'].unique()) < num_test:
            test_df = pd.concat([test_df, group], ignore_index=True)
        else:
            break

    train_df = df[~df['activity_ID'].isin(test_df['activity_ID'])]

    return train_df['activity_ID'].unique(), test_df['activity_ID'].unique()

def split_activities_kinases(df, test_size=0.2):
    '''
    df with columns: activity_ID, accession
    '''
    num_test = int(len(df['activity_ID'].unique()) * test_size)
    grouped_df = df.groupby('accession')

    # get groups until num_test is reached
    test_df = pd.DataFrame(columns=df.columns)

    for name, group in grouped_df:
        if len(test_df['activity_ID'].unique()) < num_test:
            test_df = pd.concat([test_df, group], ignore_index=True)
        else:
            break

    train_df = df[~df['activity_ID'].isin(test_df['activity_ID'])]

    return train_df['activity_ID'].unique(), test_df['activity_ID'].unique()

def get_plec_indices(df):
    vina_table = pd.read_csv('PATH_TO_PLEC_TABLE.csv')
    vina_pose_dict = dict(zip(vina_table['pose_ID'], vina_table.index))

    print('Finished loading PLEC data')
    print(f'Retrieving indices...')

    df['PLEC_index'] = df['pose_ID'].apply(lambda x: vina_pose_dict[x])

    return df


if __name__ == '__main__':
    DOCKING_SOFTWARE = argv[1] # Change this between vina and diffdock
    VARIATION = argv[2] # Change this between random, compounds and kinases
    OUT_FILE = f'ML_table_{DOCKING_SOFTWARE}_{VARIATION}.csv'

    subset = pd.read_csv('PATH_TO_SUBSET_TABLE.csv') # Table with columns: activity_ID, accession, InChIKey
    subset_acts = subset['activity_ID'].unique()

    if VARIATION == 'random':
        train, test = split_activities(subset_acts) 
    elif VARIATION == 'compounds':
        train, test = split_activities_compounds(subset) 
    else:
        train, test = split_activities_kinases(subset) 

    print('Unique activities:', len(subset_acts))
    print('Train size (activities):', len(train))
    print('Test size (activities):', len(test))

    # Get train poses
    if DOCKING_SOFTWARE == 'vina':
        train_results = DATABASE.query(f'SELECT pose_ID, activities.accession, Poses_Vina.klifs_ID, activities.InChIKey, Poses_Vina.SMILES_docked, molfile, pchembl_value_Mean FROM Poses_Vina \
                                        INNER JOIN Overview ON Poses_Vina.klifs_ID = Overview.klifs_ID AND Poses_Vina.SMILES_docked = Overview.SMILES_docked \
                                        INNER JOIN activities ON Overview.activity_ID = activities.activity_ID \
                                        WHERE activities.activity_ID IN {tuple(train)} AND num_poses_vina >= 3')
    else:
        train_results = DATABASE.query(f'SELECT pose_ID, activities.accession, Poses_Diffdock.klifs_ID, activities.InChIKey, Poses_Diffdock.SMILES_docked, molfile, pchembl_value_Mean FROM Poses_Diffdock \
                                        INNER JOIN Overview ON Poses_Diffdock.klifs_ID = Overview.klifs_ID AND Poses_Diffdock.SMILES_docked = Overview.SMILES_docked \
                                        INNER JOIN activities ON Overview.activity_ID = activities.activity_ID \
                                        WHERE activities.activity_ID IN {tuple(train)} AND num_poses_diffdock >= 3')

    train_df = pd.DataFrame(train_results, columns=['pose_ID', 'accession', 'klifs_ID', 'InChIKey', 'SMILES_docked', 'molfile', 'pIC50'])
    train_grouped = train_df.groupby(['klifs_ID', 'SMILES_docked'])

    poses = []

    for name, group in tqdm(train_grouped):
        group = group.sort_values(by='molfile')
        poses.append(group[:3])

    train_poses = pd.concat(poses, ignore_index=True)
    train_poses.drop(columns=['molfile'], inplace=True)
    train_poses['partition'] = 'train'

    print('Train poses:', len(train_poses))

    # Get test poses
    if DOCKING_SOFTWARE == 'vina':
        test_results = DATABASE.query(f'SELECT pose_ID, activities.accession, Poses_Vina.klifs_ID, activities.InChIKey, Poses_Vina.SMILES_docked, molfile, pchembl_value_Mean FROM Poses_Vina \
                                        INNER JOIN Overview ON Poses_Vina.klifs_ID = Overview.klifs_ID AND Poses_Vina.SMILES_docked = Overview.SMILES_docked \
                                        INNER JOIN activities ON Overview.activity_ID = activities.activity_ID \
                                        WHERE activities.activity_ID IN {tuple(test)} AND num_poses_vina >= 3')
    else:
        test_results = DATABASE.query(f'SELECT pose_ID, activities.accession, Poses_Diffdock.klifs_ID, activities.InChIKey, Poses_Diffdock.SMILES_docked, molfile, pchembl_value_Mean FROM Poses_Diffdock \
                                        INNER JOIN Overview ON Poses_Diffdock.klifs_ID = Overview.klifs_ID AND Poses_Diffdock.SMILES_docked = Overview.SMILES_docked \
                                        INNER JOIN activities ON Overview.activity_ID = activities.activity_ID \
                                        WHERE activities.activity_ID IN {tuple(test)} AND num_poses_diffdock >= 3')

    test_df = pd.DataFrame(test_results, columns=['pose_ID', 'accession', 'klifs_ID', 'InChIKey', 'SMILES_docked', 'molfile', 'pIC50'])
    test_grouped = test_df.groupby(['klifs_ID', 'SMILES_docked'])

    poses = []

    for name, group in tqdm(test_grouped):
        group = group.sort_values(by='molfile')
        poses.append(group[:3])

    test_poses = pd.concat(poses, ignore_index=True)
    test_poses.drop(columns=['molfile'], inplace=True)
    test_poses['partition'] = 'test'

    print('Test poses:', len(test_poses))
    print('Total poses:', len(train_poses) + len(test_poses))

    final_data = pd.concat([train_poses, test_poses], ignore_index=True)
    final_data = get_plec_indices(final_data)
    final_data.to_csv(OUT_FILE, index=False)
