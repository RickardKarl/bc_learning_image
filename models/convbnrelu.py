import torch
import torch.nn as nn
import torch.nn.functional as F


def _weights_init(m):
    """
        Initialization of CNN weights
    """
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, pad, bias=bias)
        # track_running_stats is used to initialize the
        # running estimates as well as to check if they should be updated in training
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.apply(_weights_init)

        # self.train = True

    def forward(self, x, train):
        h = self.conv(x)
        # self.train = train
        h = self.bn(h, track_running_stats=train)

        return F.relu(h)
