'''
Script to display the kinase results in a .html file
'''
from jinja2 import Environment, FileSystemLoader
from matplotlib import colors, cm
import pandas as pd
import numpy as np


file_loader = FileSystemLoader('.')
env = Environment(loader=file_loader)

template = env.get_template('kinase_table_template.html')

ROOT = 'PATH_TO_ROOT'
R2_FILE = ROOT + 'PATH_TO_R2_FILE'
RMSE_FILE = ROOT + 'PATH_TO_RMSE_FILE'
OUT_FILE = ROOT + 'PATH_TO_WRITE_RESULTS'

r2_data = pd.read_csv(R2_FILE)
r2_data.sort_values('accession', inplace=True)
r2_data['max'] = r2_data[['DIFFDOCK', 'VINA', 'ECFP']].max(axis=1)
r2_data = r2_data.groupby('kinase_group').agg(lambda x: x.tolist()).to_dict(orient='index')

rmse_data = pd.read_csv(RMSE_FILE)
rmse_data.sort_values('accession', inplace=True)
rmse_data['min'] = rmse_data[['DIFFDOCK', 'VINA', 'ECFP']].min(axis=1)
rmse_data = rmse_data.groupby('kinase_group').agg(lambda x: x.tolist()).to_dict(orient='index')

with open(OUT_FILE, 'w') as fi:
    fi.write(template.render(r2=r2_data, rmse=rmse_data))