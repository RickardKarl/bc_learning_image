import numpy as np
import random
import utils
import torch.nn.functional as F

"""
Different labels that are applied to the mixed image (in BC+ learning)
 (a) single label with softmax cross entropy loss: t = t1 if r > 0.5, otherwise t = t2
 (b) multi label with sigmoid cross entropy loss:  t = t1 + t2
 (c) ratio label with KL loss
"""


# Single label with softmax cross entropy loss
def ablation_single_label(label1, label2, r, nClasses):
    # Mix two labels
    eye = np.eye(nClasses)
    label = eye[label1].astype(np.float32) if r > 0.5 else eye[label2].astype(np.float32)

    return label


def ablation_single_loss(y, t):
    loss = F.cross_entropy(y, t)

    return loss


# Multi label with sigmoid cross entropy loss
def ablation_multi_label(label1, label2, nClasses):
    # Mix two labels
    eye = np.eye(nClasses)
    label = (eye[label1] + eye[label2]).astype(np.float32)

    return label


def ablation_multi_loss(y, t):
    loss = F.binary_cross_entropy(F.sigmoid(y), F.sigmoid(t))

    return loss


# Proposed ratio label with KL loss
def ablation_ratio_label(label1, label2, r, nClasses):
    # Mix two labels
    eye = np.eye(nClasses)
    label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)

    return label


def ablation_ratio_loss(y, t):
    loss = utils.kl_divergence(y, t)

    return loss
