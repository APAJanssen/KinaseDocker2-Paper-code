'''
Script that runs the DNN on ECFP data.
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
from datasets_ECFP import CustomDataset
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
    # Retrieving accessions and looping, creating different trained models for each
    print('Retrieving data...')
    input_file = pd.read_csv(f'DNN_data/{args.input_file}')
    accessions = input_file['accession'].unique().tolist()

    # Loop over accessions
    for accession in accessions:
        print('-'*50)
        print(f'Training model for {accession}...')
        print('-'*50)

        writer = SummaryWriter(f'runs/{args.exp_name}/{accession}') # tensorboard logdir

        train_loader = DataLoader(CustomDataset('train', args.input_file, accession), num_workers=0,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(CustomDataset('test', args.input_file, accession), num_workers=0,
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
                savepath = f'DNN_checkpoints/{args.exp_name}/best_model_{accession}.t7'
                torch.save(model_state_dict, savepath)

        writer.flush()
        writer.close()

def test(args):
    print('Retrieving data...')
    input_file = pd.read_csv(f'DNN_data/{args.input_file}')
    accessions = input_file['accession'].unique().tolist()

    # Loop over accessions
    for accession in accessions:
        print('-'*50)
        print(f'Testing model for {accession}...')
        print('-'*50)

        test_loader = DataLoader(CustomDataset('test', args.input_file, accession), num_workers=0,
                                batch_size=args.batch_size, shuffle=False, drop_last=False)

        device = args.device
        model = NN(input_size=args.plec_size, dropout=args.dropout).to(device)

        checkpoint = torch.load(f'DNN_checkpoints/{args.exp_name}/best_model_{accession}.t7')
        model.load_state_dict(checkpoint)
        model = model.eval()

        test_true = []
        test_preds = []
        test_stds = []
        all_poses = []

        with torch.no_grad():
            for data, targets, poses in tqdm(test_loader):
                data = data.to(device)
                targets = targets.numpy().flatten()

                scores = model(data)
                scores = scores.detach().cpu().numpy()

                preds = scores.flatten()

                test_preds.extend(preds)
                test_true.extend(targets)
                all_poses.extend(poses)

        data = pd.DataFrame({'activity_ID': all_poses, 'real': test_true, 'preds': test_preds})
        data.to_csv(f'DNN_checkpoints/{args.exp_name}/results_{accession}.csv', index=False)

        slope, intercept, r_value, p_value, std_err = stats.linregress(test_true, test_preds)
        r2 = r_value**2
        rmse = mean_squared_error(test_true, test_preds, squared=False)

        print(f'R2: {r2}')
        print(f'RMSE: {rmse}')
    

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
    parser.add_argument('--plec_size', type=int, default=2048,
                        help='Size of PLEC')          
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

    
