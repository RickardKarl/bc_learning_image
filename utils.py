import numpy as np
import random
import torch.nn.functional as F
import torch

def padding(pad):
    def f(image):
        return np.pad(image, ((0, 0), (pad, pad), (pad, pad)), 'constant')

    return f


def random_crop(size):
    def f(image):
        _, h, w = image.shape
        p = random.randint(0, h - size)
        q = random.randint(0, w - size)
        return image[:, p: p + size, q: q + size]

    return f


def horizontal_flip():
    def f(image):
        if random.randint(0, 1):
            image = image[:, :, ::-1]
        return image

    return f


def normalize(mean, std):
    def f(image):
        return (image - mean[:, None, None]) / std[:, None, None]

    return f


# For BC+
def zero_mean(mean, std):
    def f(image):
        image_mean = np.mean(image, keepdims=True)
        return (image - image_mean - mean[:, None, None]) / std[:, None, None]

    return f


def kl_divergence(y, t):
    entropy = F.softmax(t, dim=1) * F.log_softmax(t, dim=1)
    entropy = - entropy.sum()
    crossEntropy = t * F.log_softmax(y, dim=1)
    crossEntropy = - crossEntropy.sum()
    
    return torch.div((crossEntropy - entropy), y.shape[0])
    #return F.kl_div(y, t, reduction="batchmean")


def to_hms(time):
    h = int(time // 3600)
    m = int((time - h * 3600) // 60)
    s = int(time - h * 3600 - m * 60)
    if h > 0:
        line = '{}h{:02d}m'.format(h, m)
    else:
        line = '{}m{:02d}s'.format(m, s)

    return line
