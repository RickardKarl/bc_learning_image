# define the 11-layer convnet architecture

import math
import random
import numpy
import torch
from torch import cuda
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
        self.fc4 = nn.Linear(256 * 4 * 4, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, n_classes)

        self.train = True

    def forward(self, x):

        Mix = False
        whereToMix = 0
        
        if type(x) == list:

            Mix = True

            # Split imput list
            labels = x[1]
            images = x[0]
            whereToMix = x[2]

            batchSize = images.size()[0]

            device = torch.device("cuda" if cuda.is_available() else "cpu")

            images1 = torch.zeros([batchSize, 3, 32, 32])
            images2 = torch.zeros([batchSize, 3, 32, 32])

            # Split image batches for parallel training
            for i in range(batchSize):
                images1[i] = images[i][0]
                images2[i] = images[i][1]
            
            images1 = images1.to(device)
            images2 = images2.to(device)

        if whereToMix > 0:
            # Parallel training
            h1 = self.conv11(images1)
            h2 = self.conv11(images2)

            h1 = self.conv12(h1)
            h2 = self.conv12(h2)

            h1 = F.max_pool2d(h1, 2)
            h2 = F.max_pool2d(h2, 2)
        else:
            h = self.conv11(x)
            h = self.conv12(h)
            h = F.max_pool2d(h, 2)

        if whereToMix > 1:
            h1 = self.conv21(h1)
            h2 = self.conv21(h2)

            h1 = self.conv22(h1)
            h2 = self.conv22(h2)

            h1 = F.max_pool2d(h1, 2)
            h2 = F.max_pool2d(h2, 2)
        else:
            if whereToMix == 1:
                # Mix images and labels
                h, mixedLabels = self.mix(h1, h2, labels)
            h = self.conv21(h)
            h = self.conv22(h)
            h = F.max_pool2d(h, 2)

        if whereToMix > 2:
            h1 = self.conv31(h1)
            h2 = self.conv31(h2)

            h1 = self.conv32(h1)
            h2 = self.conv32(h2)

            h1 = self.conv33(h1)
            h2 = self.conv33(h2)

            h1 = self.conv34(h1)
            h2 = self.conv34(h2)

            h1 = F.max_pool2d(h1, 2)
            h2 = F.max_pool2d(h2, 2)
        else:
            if whereToMix == 2:
                # Mix images and labels
                h, mixedLabels = self.mix(h1, h2, labels)
            h = self.conv31(h)
            h = self.conv32(h)
            h = self.conv33(h)
            h = self.conv34(h)
            h = F.max_pool2d(h, 2)

        if whereToMix > 3:
            h1 = h1.view(h1.size(0), -1)
            h2 = h2.view(h2.size(0), -1)
            h1 = F.dropout(F.relu(self.fc4(h1)), training=self.train)
            h2 = F.dropout(F.relu(self.fc4(h2)), training=self.train)
        else:
            if whereToMix == 3:
                # Mix images and labels
                h, mixedLabels = self.mix(h1, h2, labels)
            h = h.view(h.size(0), -1)
            h = F.dropout(F.relu(self.fc4(h)), training=self.train)

        if whereToMix > 4:
            h1 = F.dropout(F.relu(self.fc5(h1)), training=self.train)
            h2 = F.dropout(F.relu(self.fc5(h2)), training=self.train)
            h, mixedLabels = self.mix(h1, h2, labels)
        else:
            if whereToMix == 4:
                # Mix images and labels
                h, mixedLabels = self.mix(h1, h2, labels)
            h = F.dropout(F.relu(self.fc5(h)), training=self.train)

        if Mix:
            return self.fc6(h), mixedLabels
        else:
            return self.fc6(h)
    
    def mix(self, images1, images2, labels):

        device = torch.device("cuda" if cuda.is_available() else "cpu")
        dim = images1.size()
        batchSize = dim[0]

        mixedLabels = torch.zeros([batchSize, 10]).to(device) # Hard coded for 10 classes

        r = torch.tensor(numpy.random.uniform(size=batchSize)).to(device)

        if images1.dim() == 4:
            rdim = torch.zeros([batchSize, 1, 1, 1]).to(device)
            for i in range(batchSize):
                rdim[i][0][0][0] = r[i]

                # Mix two labels
                eye = torch.tensor(numpy.eye(10)) # Hard coded for 10 classes
                mixedLabels[i] = (eye[labels[i][0]] * r[i] + eye[labels[i][1]] * (1 - r[i]))

        else:
            rdim = torch.zeros([batchSize, 1]).to(device)
            for i in range(batchSize):
                rdim[i][0] = r[i]

                # Mix two labels
                eye = torch.tensor(numpy.eye(10)) # Hard coded for 10 classes
                mixedLabels[i] = (eye[labels[i][0]] * r[i] + eye[labels[i][1]] * (1 - r[i]))

        mixedImages = images1 * rdim + images2 * (1 - rdim)

        return mixedImages, mixedLabels