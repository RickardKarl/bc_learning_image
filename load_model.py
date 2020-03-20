import sys
import argparse

import torch
from torch import cuda

from models.convnet import ConvNet
import dataset
from train import Trainer

def eval_parse():

    parser = argparse.ArgumentParser(description='Evaluate model from checkpoint')

    # General settings
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
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
    

if __name__ == "__main__":

    opt = eval_parse()

    device = torch.device("cuda" if cuda.is_available() else "cpu")
    train_iter, val_iter = dataset.setup(opt)

    model = ConvNet(opt.nClasses)
    model.load_state_dict(torch.load(opt.checkpoint, map_location=device))
    
    trainer = Trainer(model, None, train_iter, val_iter, opt)
    error_rate = trainer.val()

    print("Error rate: {} | Filename: {}".format(error_rate, opt.checkpoint))
