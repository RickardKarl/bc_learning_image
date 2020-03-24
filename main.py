"""
 Between-class Learning for Image Classification.
 Porting to PyTorch
"""

import sys
import os
from datetime import datetime
import torch

import opts
import models
import dataset
from train import Trainer


def main():
    opt = opts.parse()
    if opt.noGPU == False:
        torch.cuda.set_device(opt.gpu)
    for i in range(1, opt.nTrials + 1):
        print('+-- Trial {} --+'.format(i))
        best_val_error = train(opt, i)
        print("Best validation rate: {}".format(best_val_error))


def train(opt, trial):

    # Get filename for saving model
    if opt.BC:
        if opt.plus:
            learning = 'BC+'
        else:
            learning = 'BC'
    else:
        learning = 'standard'
    # Get current time
    now = datetime.now() # current date and time
    string_date = now.strftime("%d%H%M")
    # Save filename
    filename = "{}_{}_trial{}_{}.th".format(learning, opt.dataset, trial, string_date)

    # Keep track of best validation error rate
    best_val_error = 100.0

    model = getattr(models, opt.netType)(opt.nClasses)
    if opt.noGPU == False:
        model.cuda()
    # TODO: there is no direct method in PyTorch with NesterovAG
    optimizer = torch.optim.SGD(params=model.parameters(), lr=opt.LR, momentum=opt.momentum,
                                weight_decay=opt.weightDecay, nesterov=True)

    train_iter, val_iter = dataset.setup(opt)
    trainer = Trainer(model, optimizer, train_iter, val_iter, opt)

    for epoch in range(1, opt.nEpochs + 1):
        train_loss, train_top1 = trainer.train(epoch)
        val_top1 = trainer.val()
        trainer.scheduler.step()
        sys.stderr.write('\r\033[K')
        sys.stdout.write(
            '| Epoch: {}/{} | Train: LR {}  Loss {:.3f}  top1 {:.2f} | Val: top1 {:.2f}\n'.format(
                epoch, opt.nEpochs, trainer.scheduler.get_last_lr()[0], train_loss, train_top1, val_top1))
        sys.stdout.flush()

        if val_top1 < best_val_error:
            best_val_error = val_top1
            if opt.save != 'None':
                print("New best validation error rate: {} (Saved checkpoint)".format(best_val_error))
                torch.save(model.state_dict(), os.path.join(opt.save, filename))
        
    return best_val_error



if __name__ == '__main__':
    main()
