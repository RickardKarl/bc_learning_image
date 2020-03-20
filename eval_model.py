import sys
import argparse
from os import walk
from os.path import isfile, join

import torch
from torch import cuda

from models.convnet import ConvNet
import dataset
from train import Trainer

def eval_parse():

    parser = argparse.ArgumentParser(description='Evaluate model from checkpoint')

    # General settings
    parser.add_argument('--checkpoint', required=True, help='Path to folder with models')
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100'])
    parser.add_argument('--data', required=True, help='Path to dataset')

    # Additional settings
    parser.add_argument('--BC', action='store_true', help='BC learning')
    parser.add_argument('--plus', action='store_true', help='Use BC+')
    parser.add_argument('--batchSize', type=int, default=128)

    opt = parser.parse_args()

    # Dataset details
    if opt.dataset == 'cifar10':
        opt.nClasses = 10
    else:  # cifar100
        opt.nClasses = 100

    return opt

def get_model_paths(folder):

    file_list = []

    # Walks through folder and searcher for files ending with .th
    for (dirpath, dirnames, filenames) in walk(folder):
        for f in filenames:

            f = join(dirpath, f)
            if f not in file_list and f.endswith(".th"):
                file_list.append(f)

    if len(file_list) == 0:
        raise ValueError("Empty list of checkpoints")

    return file_list

if __name__ == "__main__":

    opt = eval_parse()

    device = torch.device("cuda" if cuda.is_available() else "cpu")
    train_iter, val_iter = dataset.setup(opt)

    # Get list of model paths
    checkpoints = get_model_paths(opt.checkpoint) 

    for path in checkpoints:

        model = ConvNet(opt.nClasses)
        model.load_state_dict(torch.load(path, map_location=device))
    
        trainer = Trainer(model, None, train_iter, val_iter, opt)
        error_rate = trainer.val()

        print("Error rate: {} | Model: {}".format(error_rate, path))
