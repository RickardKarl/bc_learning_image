import sys
import numpy as np
import torch
from torch import cuda
import torch.nn.functional as F
import time

import utils


def accuracy(y, t, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = t.size(0)

    _, pred = y.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(t.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


class Trainer:
    def __init__(self, model, optimizer, train_iter, val_iter, opt):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.opt = opt
        self.n_batches = len(train_iter)
        # self.n_batches = (len(train_iter.dataset) - 1) // opt.batchSize + 1
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
            x = x_array.to(device)
            t = t_array.to(device, dtype=torch.int64)

            if self.opt.BC:
                y = F.log_softmax(self.model(x), dim=1)
                y = y.to(torch.float32)
                t = t.to(torch.float32)
                loss = utils.kl_divergence(y, t)
                t_values, t_indices = torch.max(t, dim=1)
                acc = accuracy(y, t_indices)[0]
            else:
                # TODO: find out softmax_cross_entropy in PyTorch
                y = F.softmax(self.model(x), dim=1)
                loss = F.cross_entropy(y, t)
                acc = accuracy(y, t)[0]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += float(loss.data) * len(t.data)
            train_acc += float(acc.data) * len(t.data)

            elapsed_time = time.time() - self.start_time
            progress = (self.n_batches * (epoch - 1) + i + 1) * 1.0 / (self.n_batches * self.opt.nEpochs)
            eta = elapsed_time / progress - elapsed_time

            line = '* Epoch: {}/{} ({}/{}) | Train: LR {} | Time: {} (ETA: {})'.format(
                epoch, self.opt.nEpochs, i + 1, self.n_batches,
                self.optimizer.lr, utils.to_hms(elapsed_time), utils.to_hms(eta))
            sys.stderr.write('\r\033[K' + line)
            sys.stderr.flush()

        # TODO: if reset() is necessary
        # self.train_iter.reset()
        train_loss /= len(self.train_iter.dataset)
        train_top1 = 100 * (1 - train_acc / len(self.train_iter.dataset))

        return train_loss, train_top1

    def val(self):
        self.model.train = False
        val_acc = 0
        for (x_array, t_array) in self.val_iter:
            device = torch.device("cuda" if cuda.is_available() else "cpu")
            x = x_array.to(device)
            t = t_array.to(device, dtype=torch.int64)
            y = F.softmax(self.model(x), dim=1)
            acc = accuracy(y, t)[0]
            val_acc += float(acc.data) * len(t.data)

        # TODO: if reset() is necessary
        # self.val_iter.reset()
        self.model.train = True
        val_top1 = 100 * (1 - val_acc / len(self.val_iter.dataset))

        return val_top1

    def lr_schedule(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule])
        decay = sum(epoch > divide_epoch)
        if epoch <= self.opt.warmup:
            decay = 1

        return self.opt.LR * np.power(0.1, decay)
