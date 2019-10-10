def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import torch
import numpy as np
import sys
import time
import os
import cv2
import random
import pandas as pd
random.seed(42)
st = random.getstate()


class CheXpertDataset(Dataset):
    def __init__(self, args, training=True, imputation=1.0, transforms = None):
        self.training = training
        self.args = args
        self.transforms = transforms
        self.imputation=imputation
        if self.training:
            self.csv = pd.read_csv('data/CheXpert-v1.0-small/train.csv')
        else:
            self.csv = pd.read_csv('data/CheXpert-v1.0-small/valid.csv')
        
        #change no_finding to 0 if any pathology present
        pathologies_present_idx = (self.csv.iloc[:,6:18]==1).any(1).index
        #change all pathology to 0 if no_finding is 0
        self.csv.iloc[pathologies_present_idx, 5] = 0.0 
        no_finding_idx = (self.csv.loc[:,'No Finding'] == 1).index
        self.csv.iloc[no_finding_idx,6:18] = 0.0
        self.csv= self.csv.fillna(self.imputation)
        self.csv= self.csv.replace(-1.0,self.imputation)
        self.labels_cols = self.csv.columns[-14:]
        self.img_tensorify = ToTensor()
        print('number of samples in dataset', self.csv.shape[0], 'Training:', self.training)


    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        pth = os.path.join('data',self.csv.loc[index, 'Path'])
        labels = self.csv.loc[index, self.labels_cols]

        img = Image.open(pth)
        
        if self.transforms != None:
            img = self.transforms(img)
        img = self.img_tensorify(img)
        return img, torch.Tensor(labels)

    def normalize(self, data):
        min = data.min()
        max = data.max()
        return (data - min)/(max-min)
