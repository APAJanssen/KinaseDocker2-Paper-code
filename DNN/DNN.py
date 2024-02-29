'''
Script that runs the DNN on PLEC data
'''
import argparse
import math
import os
from collections import defaultdict
from statistics import mean

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import CustomDatasetDB, CustomDatasetMaster, CustomDatasetSubset
from scipy import stats
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class NN(nn.Module):
    '''
    Structure based on PLEC paper. (and Drugex)?
    '''
    def __init__(self, input_size, dropout):
        super(NN, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_size, 4000)
        self.fc2 = nn.Linear(4000, 1000)
        self.fc3 = nn.Linear(1000, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.training:
            x = self.dropout(x)
        
        x = F.relu(self.fc2(x))

        if self.training:
            x = self.dropout(x)

        x = self.fc3(x)

        return x

def train(args):
    writer = SummaryWriter(f'runs/{args.exp_name}')

    print('Retrieving data...')

    # Pick dataset
    if args.dataset == 'DB':
        dataset = CustomDatasetDB
    elif args.dataset == 'master':
        dataset = CustomDatasetMaster
    else:
        dataset = CustomDatasetSubset

    if args.dataset == 'DB':
        train_loader = DataLoader(dataset('train', args.input_file, args.docking_type), num_workers=0,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(dataset('test', args.input_file, args.docking_type), num_workers=0,
                                batch_size=args.batch_size, shuffle=False, drop_last=False)
    else:
        train_loader = DataLoader(dataset('train'), num_workers=0,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(dataset('test'), num_workers=0,
                                batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Init model
    device = args.device
    model = NN(input_size=args.plec_size, dropout=args.dropout).to(device)

    # Loss and optim
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_r2 = 0

    # Train network
    for epoch in range(args.epochs):
        print('Epoch:', epoch, '/', args.epochs)
        train_losses = []

        #########
        # TRAIN #
        ######### 
        model.train()
        for data, targets in tqdm(train_loader):
            targets = torch.as_tensor(targets[:, 0].numpy(), dtype=torch.float)

            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            scores = scores.flatten()

            loss = criterion(scores, targets)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        ########
        # TEST #
        ########
        model.eval()
        test_preds = []
        test_true = []
        with torch.no_grad():
            for data, targets, poses in tqdm(test_loader):
                data = data.to(device)

                scores = model(data)
                scores = scores.detach().cpu().numpy()

                preds = scores.flatten()

                test_preds.extend(preds)
                test_true.extend(targets.numpy().flatten())

        slope, intercept, r_value, p_value, std_err = stats.linregress(test_true, test_preds)
        r2 = r_value**2
        rmse = mean_squared_error(test_true, test_preds, squared=False)

        # tensorboard
        writer.add_scalar("Loss/train", np.mean(train_losses), epoch)
        writer.add_scalar("R2", r2, epoch)
        writer.add_scalar("RMSE", rmse, epoch)

        if r2 >= best_r2:
            best_r2 = r2

            model_state_dict = model.state_dict()
            savepath = 'DNN_checkpoints/%s/best_model.t7' % args.exp_name
            torch.save(model_state_dict, savepath)

    writer.flush()
    writer.close()

def test(args):
    # Pick dataset
    if args.dataset == 'DB':
        dataset = CustomDatasetDB
    elif args.dataset == 'master':
        dataset = CustomDatasetMaster
    else:
        dataset = CustomDatasetSubset

    if args.dataset == 'DB':
        test_loader = DataLoader(dataset('test', args.input_file, args.docking_type), num_workers=0,
                                batch_size=args.batch_size, shuffle=False, drop_last=False)
    else:
        test_loader = DataLoader(dataset('test'), num_workers=0,
                                batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = args.device
    model = NN(input_size=args.plec_size, dropout=args.dropout).to(device)

    checkpoint = torch.load('DNN_checkpoints/' + args.exp_name + '/best_model.t7')
    model.load_state_dict(checkpoint)
    model = model.eval()

    test_true = []
    test_preds = []
    test_stds = []
    all_poses = []
    pose_dict = defaultdict(lambda: defaultdict(list)) # In order to calculate the mean of the poses

    if args.dataset == 'DB':
        ml_table = pd.read_csv(f'DNN_data/{args.input_file}').set_index('pose_ID').to_dict(orient='index')
    elif args.dataset == 'subset':
        ml_table = pd.read_csv('DNN_data/ML_table_diffdock.csv').set_index('pose_ID').to_dict(orient='index')

    with torch.no_grad():
        for data, targets, poses in tqdm(test_loader):
            data = data.to(device)
            targets = targets.numpy().flatten()

            scores = model(data)
            scores = scores.detach().cpu().numpy()

            preds = scores.flatten()

            for i, pose in enumerate(poses):
                if args.dataset == 'master':
                    key = pose.split('_')[0]
                else:
                    key = str(ml_table[pose]['klifs_ID']) + '_' + ml_table[pose]['SMILES_docked']

                pose_dict[key]['preds'].append(preds[i])
                pose_dict[key]['true'].append(targets[i])

            test_preds.extend(preds)
            test_true.extend(targets)
            all_poses.extend(poses)

    data = pd.DataFrame({'poseID': all_poses, 'real': test_true, 'preds': test_preds})
    data.to_csv(f'DNN_checkpoints/{args.exp_name}/results.csv', index=False)

    slope, intercept, r_value, p_value, std_err = stats.linregress(test_true, test_preds)
    r2 = r_value**2
    rmse = mean_squared_error(test_true, test_preds, squared=False)

    print(f'R2: {r2}')
    print(f'RMSE: {rmse}')

    mean_true = []
    mean_pred = []
    max_true = []
    max_pred = []
    poses = []

    for pose, sub_dict in pose_dict.items():
        poses.append(pose)
        mean_true.append(mean(sub_dict['true']))
        mean_pred.append(mean(sub_dict['preds']))
        max_true.append(max(sub_dict['true']))
        max_pred.append(max(sub_dict['preds']))

    mean_data = pd.DataFrame({'poseID': poses, 'mean_real': mean_true, 'mean_pred': mean_pred})
    max_data = pd.DataFrame({'poseID': poses, 'max_real': max_true, 'max_pred': max_pred})

    mean_data.to_csv(f'DNN_checkpoints/{args.exp_name}/mean_results.csv', index=False)
    max_data.to_csv(f'DNN_checkpoints/{args.exp_name}/max_results.csv', index=False)

    slope, intercept, r_value, p_value, std_err = stats.linregress(mean_true, mean_pred)
    r2 = r_value**2
    rmse = mean_squared_error(mean_true, mean_pred, squared=False)

    print(f'R2 (mean of poses): {r2}')
    print(f'RMSE (mean of poses): {rmse}')

    slope, intercept, r_value, p_value, std_err = stats.linregress(max_true, max_pred)
    r2 = r_value**2
    rmse = mean_squared_error(max_true, max_pred, squared=False)

    print(f'R2 (max of poses): {r2}')
    print(f'RMSE (max of poses): {rmse}')

def mean_std(x, y):
    '''
    Calculate mean and SD from numpy arrays.
    '''
    mu = (1/sum(y))*(sum(x*y))
    stddev = math.sqrt(sum(((x-mu)**2)*y))

    return mu, stddev
    

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', required=True, type=str, default='failsave',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=128, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='dropout rate')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--plec_size', type=int, default=65536,
                        help='Size of PLEC')          
    parser.add_argument('--dataset', required=True, type=str, default='DB',
                        help='Dataset type (DB, subset, master)')
    parser.add_argument('--docking_type', required=True, type=str, default='vina',
                        help='Docking software (vina, diffdock)')                                            
    parser.add_argument('--input_file', required=True, type=str,
                        help='Input file name')
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    if not os.path.exists('DNN_checkpoints'):
        os.makedirs('DNN_checkpoints')
    if not os.path.exists('DNN_checkpoints/' + args.exp_name):
        os.makedirs('DNN_checkpoints/' + args.exp_name)
    elif not args.eval:
        check = input('This model already exists, do you wish to overwrite it? (y/n) ')

        if not check == 'y':
            print('Cancelling...')
            exit()

    if not args.eval:
        train(args)
        
    test(args)

    
