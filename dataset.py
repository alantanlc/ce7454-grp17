def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from torch.utils.data import Dataset
import torch
import numpy as np
import sys
import time
import os
import cv2
import random
random.seed(42)
st = random.getstate()


class CheXpertDataset(Dataset):
    def __init__(self, args, training=True):
        self.training = training
        self.args = args

        #TODO load data


    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, index):
        #do whatever transforms u want here
        return torch.from_numpy(np.asarray(self.data[index])), torch.from_numpy(np.asarray(self.targets[index]))

    def normalize(self, data):
        min = data.min()
        max = data.max()
        return (data - min)/(max-min)