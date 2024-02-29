'''
Script to create a table of results
'''
import pandas as pd
import db
import os
from scipy import stats
from sklearn.metrics import mean_squared_error
from collections import defaultdict


DATABASE = db.setup('PATH_TO_DATABASE')

ROOT = 'PATH_TO_ROOT'
VARIATIONS = ['random', 'compounds', 'kinases']
for VARIATION in VARIATIONS:
    DIFF_FOLDER = ROOT + f'DNN_checkpoints/diffdock-{VARIATION}/'
    VINA_FOLDER = ROOT + f'DNN_checkpoints/vina-{VARIATION}/'
    ECFP_FOLDER = ROOT + 'DNN_checkpoints/ECFP1/'
    R2_OUT_FILE = ROOT + f'kinase_r2_{VARIATION}.csv'
    RMSE_OUT_FILE = ROOT + f'kinase_rmse_{VARIATION}.csv'
    ML_TABLE_DIFFDOCK = pd.read_csv(ROOT + f'DNN_data/db_ML_table_diffdock_{VARIATION}.csv')
    ML_TABLE_VINA = pd.read_csv(ROOT + f'DNN_data/db_ML_table_vina_{VARIATION}.csv')
    ML_TABLE_ECFP = pd.read_csv(ROOT + f'DNN_data/db_ML_table_ECFP.csv')


    def denormalize(value):
        max_value = 12
        min_value = 3

        return (value * (max_value - min_value)) + min_value

    def get_accession(poseID):
        klifs_ID = poseID.split('_')[0]
        return DATABASE.query(f'SELECT accession FROM klifs WHERE klifs_ID = {klifs_ID}')[0]

    def get_kinase_family(accession):
        return DATABASE.query(f'SELECT Kinasegroup FROM Kinases WHERE accession = "{accession}"')[0]

    def get_gene_name(accession):
        return DATABASE.query(f'SELECT Kinase FROM Kinases WHERE accession = "{accession}"')[0]

    def get_activities(accession):
        return DATABASE.query(f'SELECT COUNT(*) FROM activities WHERE accession = "{accession}"')[0]

    r2_results = defaultdict(dict)
    rmse_results = defaultdict(dict)

    for folder in [DIFF_FOLDER, VINA_FOLDER, ECFP_FOLDER]:
        if folder == ECFP_FOLDER:
            for accession in ML_TABLE_ECFP['accession'].unique():
                test_acts = len(ML_TABLE_ECFP.loc[(ML_TABLE_ECFP['accession'] == accession) & (ML_TABLE_ECFP['partition'] == 'test')]['activity_ID'].unique())
                train_acts = len(ML_TABLE_ECFP.loc[(ML_TABLE_ECFP['accession'] == accession) & (ML_TABLE_ECFP['partition'] == 'train')]['activity_ID'].unique())

                data = pd.read_csv(ECFP_FOLDER + f'results_{accession}.csv')
                data['real'] = data['real'].apply(denormalize)
                data['preds'] = data['preds'].apply(denormalize)

                slope, intercept, r_value, p_value, std_err = stats.linregress(data['real'], data['preds'])
                r2 = r_value**2
                rmse = mean_squared_error(data['real'], data['preds'], squared=False)

                r2_results[accession]['ECFP'] = r2
                r2_results[accession]['ECFP_test'] = test_acts
                r2_results[accession]['ECFP_train'] = train_acts
                
                rmse_results[accession]['ECFP'] = rmse
                rmse_results[accession]['ECFP_test'] = test_acts
                rmse_results[accession]['ECFP_train'] = train_acts
        else:
            data = pd.read_csv(folder + 'mean_results.csv')
            data['mean_real'] = data['mean_real'].apply(denormalize)
            data['mean_pred'] = data['mean_pred'].apply(denormalize)
            data['kinase'] = data['poseID'].apply(get_accession)

            for kinase in data['kinase'].unique():
                if folder == DIFF_FOLDER:
                    name = 'DIFFDOCK'
                    test_acts = len(ML_TABLE_DIFFDOCK.loc[(ML_TABLE_DIFFDOCK['accession'] == kinase) & (ML_TABLE_DIFFDOCK['partition'] == 'test')]['activity_ID'].unique())
                    train_acts = len(ML_TABLE_DIFFDOCK.loc[(ML_TABLE_DIFFDOCK['accession'] == kinase) & (ML_TABLE_DIFFDOCK['partition'] == 'train')]['activity_ID'].unique())
                else: # Folder is VINA_FOLDER
                    name = 'VINA'
                    test_acts = len(ML_TABLE_VINA.loc[(ML_TABLE_VINA['accession'] == kinase) & (ML_TABLE_VINA['partition'] == 'test')]['activity_ID'].unique())
                    train_acts = len(ML_TABLE_VINA.loc[(ML_TABLE_VINA['accession'] == kinase) & (ML_TABLE_VINA['partition'] == 'train')]['activity_ID'].unique())

                kinase_data = data[data['kinase'] == kinase]

                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(kinase_data['mean_real'], kinase_data['mean_pred'])
                    r2 = r_value**2
                except ValueError:
                    r2 = None

                rmse = mean_squared_error(kinase_data['mean_real'], kinase_data['mean_pred'], squared=False)

                r2_results[kinase][name] = r2
                r2_results[kinase][f'{name}_test'] = test_acts
                r2_results[kinase][f'{name}_train'] = train_acts
                rmse_results[kinase][name] = rmse
                rmse_results[kinase][f'{name}_test'] = test_acts
                rmse_results[kinase][f'{name}_train'] = train_acts

    r2_df = pd.DataFrame.from_dict(r2_results, orient='index').reset_index(names='accession')
    r2_df['kinase_group'] = r2_df['accession'].apply(get_kinase_family)
    r2_df['gene_name'] = r2_df['accession'].apply(get_gene_name)
    r2_df['num_acts'] = r2_df['accession'].apply(get_activities)
    r2_df.sort_values(by='kinase_group', inplace=True)
    r2_df.set_index(['kinase_group', 'accession'], inplace=True)

    rmse_df = pd.DataFrame.from_dict(rmse_results, orient='index').reset_index(names='accession')
    rmse_df['kinase_group'] = rmse_df['accession'].apply(get_kinase_family)
    rmse_df['gene_name'] = rmse_df['accession'].apply(get_gene_name)
    rmse_df['num_acts'] = rmse_df['accession'].apply(get_activities)
    rmse_df.sort_values(by='kinase_group', inplace=True)
    rmse_df.set_index(['kinase_group', 'accession'], inplace=True)

    r2_df.to_csv(R2_OUT_FILE)
    rmse_df.to_csv(RMSE_OUT_FILE)
