import sys
import numpy as np
import torch
from torch import cuda
import torch.nn.functional as F
import time
from sklearn.metrics import accuracy_score

import utils


def accuracy(y, t):
    """ Computes the multiclass classification accuracy """
    print(y.shape)
    print(t.shape)
    pred = y.argmax(axis=1).reshape(t.shape)
    print("argmax", pred)
    #count = (pred == t).sum()
    #acc = np.asarray(float(count) / len(t.data))
    acc = accuracy_score(t, pred)
    print("acc", acc)
    return acc


class Trainer:
    def __init__(self, model, optimizer, train_iter, val_iter, opt):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.opt = opt
        # self.n_batches = len(train_iter)
        self.n_batches = (len(train_iter.dataset) - 1) // opt.batchSize + 1
        self.start_time = time.time()

    def train(self, epoch):
        """
            run one train epoch
        """
        self.optimizer.lr = self.lr_schedule(epoch)
        train_loss = 0
        train_acc = 0
        for i, (x_array, t_array) in enumerate(self.train_iter):
            device = torch.device("cuda" if cuda.is_available() else "cpu")
            self.optimizer.zero_grad()

            x = x_array.to(device)
            t = t_array.to(device)
            y = self.model(x)
            if self.opt.BC:
                t = t.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
                loss = utils.kl_divergence(y, t)
                t_indices = torch.argmax(t, dim=1)
                acc = accuracy(y.data, t_indices)
            else:
                """ F.cross_entropy already combines log_softmax and NLLLoss """
                t = t.to(device, dtype=torch.int32)
                loss = F.cross_entropy(y, t)
                acc = accuracy(y.data, t)

            
            loss.backward()
            self.optimizer.step()

            train_loss += float(loss.item()) * len(t.data)
            train_acc += float(acc.item()) * len(t.data)

            elapsed_time = time.time() - self.start_time
            progress = (self.n_batches * (epoch - 1) + i + 1) * 1.0 / (self.n_batches * self.opt.nEpochs)
            eta = elapsed_time / progress - elapsed_time

            line = '* Epoch: {}/{} ({}/{}) | Train: LR {} | Time: {} (ETA: {})'.format(
                epoch, self.opt.nEpochs, i + 1, self.n_batches,
                self.optimizer.lr, utils.to_hms(elapsed_time), utils.to_hms(eta))
            sys.stderr.write('\r\033[K' + line)
            sys.stderr.flush()

        train_loss /= len(self.train_iter.dataset)
        train_top1 = 100 * (1 - train_acc / len(self.train_iter.dataset))

        return train_loss, train_top1

    def val(self):
        self.model.train = False
        val_acc = 0
        for (x_array, t_array) in self.val_iter:
            device = torch.device("cuda" if cuda.is_available() else "cpu")

            # Disable gradient computation during validation
            with torch.no_grad(): 
                x = x_array.to(device)
                t = t_array.to(device, dtype=torch.int64)
                # TODO: figure out why to use softmax here since it also works fine without softmax
                y = F.softmax(self.model(x), dim=1)

            acc = accuracy(y.data, t)
            val_acc += float(acc.item()) * len(t.data)

        self.model.train = True
        val_top1 = 100 * (1 - val_acc / len(self.val_iter.dataset))

        return val_top1

    def lr_schedule(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule])
        decay = sum(epoch > divide_epoch)
        if epoch <= self.opt.warmup:
            decay = 1

        return self.opt.LR * np.power(0.1, decay)
