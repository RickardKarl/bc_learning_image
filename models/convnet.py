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
        
        if type(x) == list:

            Mix = True

            labels = x[1]
            x = x[0]

            device = torch.device("cuda" if cuda.is_available() else "cpu")

            images1 = torch.zeros([128, 3, 32, 32])
            images2 = torch.zeros([128, 3, 32, 32])

            batchSize = 128 # I'm not sure how to import opt, so this is currently hard coded

            for i in range(batchSize):
                images1[i] = x[i][0]
                images2[i] = x[i][1]
            
            images1 = images1.to(device)
            images2 = images2.to(device)

            h1 = self.conv11(images1)
            h2 = self.conv11(images2)

            h1 = self.conv12(h1)
            h2 = self.conv12(h2)

            h1 = F.max_pool2d(h1, 2)
            h2 = F.max_pool2d(h2, 2)
            print(h1.size())

            h, mixedLabels = self.mix(h1, h2, labels)

        else:     
            h = self.conv11(x)
            h = self.conv12(h)
            h = F.max_pool2d(h, 2)

        h = self.conv21(h)
        h = self.conv22(h)
        h = F.max_pool2d(h, 2)

        h = self.conv31(h)
        h = self.conv32(h)
        h = self.conv33(h)
        h = self.conv34(h)
        h = F.max_pool2d(h, 2)

        h = h.view(h.size(0), -1)

        h = F.dropout(F.relu(self.fc4(h)), training=self.train)
        h = F.dropout(F.relu(self.fc5(h)), training=self.train)

        if Mix:
            return self.fc6(h), mixedLabels
        else:
            return self.fc6(h)
    
    def mix(self, images1, images2, labels):

        batchSize = 128 # I'm not sure how to import opt, so this is currently hard coded

        #images1 = images1.numpy()
        #images2 = images2.numpy()
        #labels = labels.numpy()

        mixedImages = torch.zeros([128, 64, 16, 16])
        mixedLabels = torch.zeros([128, 10])

        for i in range(batchSize):
            r = torch.tensor(numpy.array(random.random()))
            mixedImages[i] = (images1[i] * r + images2[i] * (1 - r))

            # Mix two labels
            eye = np.eye(10) # Hard coded for 10 classes
            mixedLabels[i] = (eye[labels[i][0]] * r + eye[labels[i][1]] * (1 - r))

        return mixedImages, mixedLabels
