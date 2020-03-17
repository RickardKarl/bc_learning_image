"""
 Between-class Learning for Image Classification.
 Porting to PyTorch
"""

import sys
import os
import torch

import opts
import models
import dataset
from train import Trainer


def main():
    opt = opts.parse()
    torch.cuda.set_device(opt.gpu)
    for i in range(1, opt.nTrials + 1):
        print('+-- Trial {} --+'.format(i))
        train(opt, i)


def train(opt, trial):
    model = getattr(models, opt.netType)(opt.nClasses)
    model.cuda()
    # TODO: there is no direct method in PyTorch with NesterovAG
    optimizer = torch.optim.SGD(params=model.parameters(), lr=opt.LR, momentum=opt.momentum,
                                weight_decay=opt.weightDecay, nesterov=True)

    train_iter, val_iter = dataset.setup(opt)
    print(train_iter)
    trainer = Trainer(model, optimizer, train_iter, val_iter, opt)

    for epoch in range(1, opt.nEpochs + 1):
        train_loss, train_top1 = trainer.train(epoch)
        val_top1 = trainer.val()
        sys.stderr.write('\r\033[K')
        sys.stdout.write(
            '| Epoch: {}/{} | Train: LR {}  Loss {:.3f}  top1 {:.2f} | Val: top1 {:.2f}\n'.format(
                epoch, opt.nEpochs, trainer.optimizer.lr, train_loss, train_top1, val_top1))
        sys.stdout.flush()

    if opt.save != 'None':
        torch.save(model.state_dict(), os.path.join(opt.save, 'checkpoint.th'))


if __name__ == '__main__':
    main()
