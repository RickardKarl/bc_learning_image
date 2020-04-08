# define the 11-layer convnet architecture

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.convbnrelu import ConvBNReLU


class ConvNet(nn.Module):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()
        self.conv11 = ConvBNReLU(3, 64, 3, pad=1)
        self.conv12 = ConvBNReLU(64, 64, 3, pad=1)
        self.conv21 = ConvBNReLU(64, 128, 3, pad=1)
        self.conv22 = ConvBNReLU(128, 128, 3, pad=1)
        self.conv31 = ConvBNReLU(128, 256, 3, pad=1)
        self.conv32 = ConvBNReLU(256, 256, 3, pad=1)
        self.conv33 = ConvBNReLU(256, 256, 3, pad=1)
        self.conv34 = ConvBNReLU(256, 256, 3, pad=1)
        self.conv41 = ConvBNReLU(256, 256, 5, pad=1)
        self.conv42 = ConvBNReLU(256, 512, 3, pad=1)
        self.conv43 = ConvBNReLU(512, 512, 3, pad=1)
        self.conv44 = ConvBNReLU(512, 512, 3, pad=1)
        self.conv51 = ConvBNReLU(512, 512, 5, pad=1)
        self.conv52 = ConvBNReLU(512, 512, 3, pad=1)
        self.conv53 = ConvBNReLU(512, 512, 3, pad=1)
        self.conv54 = ConvBNReLU(512, 128, 3, pad=1)
        self.fc4 = nn.Linear(3200, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, n_classes)

        self.train = True

    def forward(self, x):
        h = self.conv11(x)
        h = self.conv12(h)
        h = F.max_pool2d(h, 2)
        print(h.size())

        h = self.conv21(h)
        h = self.conv22(h)
        h = F.max_pool2d(h, 2)
        print(h.size())

        h = self.conv31(h)
        h = self.conv32(h)
        h = self.conv33(h)
        h = self.conv34(h)
        h = F.max_pool2d(h, 2)
        print(h.size())

        h = self.conv41(h)
        h = self.conv42(h)
        h = self.conv43(h)
        h = self.conv44(h)
        h = F.max_pool2d(h, 2)
        print(h.size())

        h = self.conv51(h)
        h = self.conv52(h)
        h = self.conv53(h)
        h = self.conv54(h)
        h = F.max_pool2d(h, 2)
        print(h.size())

        h = h.view(h.size(0), -1)
        print(h.size())

        h = F.dropout(F.relu(self.fc4(h)), training=self.train)
        h = F.dropout(F.relu(self.fc5(h)), training=self.train)

        return self.fc6(h)
