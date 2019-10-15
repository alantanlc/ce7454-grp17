def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from PIL import Image
from torchvision.transforms import ToTensor, Normalize
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
import torchvision.transforms as transforms

class_names = ['No Finding',
       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices']
# mean=127.898, std=74.69748171138374

class CheXpertDataset(Dataset):
    def __init__(self, training=True, imputation=1.0, transform = None, test=None):
        self.training = training
        self.transforms = transform
        self.imputation=imputation
        if self.training:
            self.csv = pd.read_csv('data/CheXpert-v1.0-small/train.csv')
        else:
            self.csv = pd.read_csv('data/CheXpert-v1.0-small/valid.csv')
        
        #if test present overrides train/val
        if test != None:
            self.csv = pd.read_csv(test)
            self.transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
        
        #change no_finding to 0 if any pathology present
        # pathologies_present_idx = (self.csv.iloc[:,6:18]==1).any(1).index
        #change all pathology to 0 if no_finding is 0
        # self.csv.iloc[pathologies_present_idx, 5] = 0.0 
        # no_finding_idx = (self.csv.loc[:,'No Finding'] == 1).index
        # self.csv.iloc[no_finding_idx,6:18] = 0.0
        self.csv= self.csv.fillna(0.0)
        self.csv= self.csv.replace(-1.0,self.imputation)
        self.labels_cols = self.csv.columns[-14:]
        self.img_tensorify = ToTensor()
        self.normalize = Normalize(mean=[127],std=[74.69])
        
        
        if test == None:
            eda = ''
            for name in class_names:
                eda += f'{name}: '
                eda += str(self.csv[name].value_counts().to_dict())
                eda += '\n'
            print(eda)
            print('number of samples in dataset', self.csv.shape[0], 'Training:', self.training)


    def __len__(self):
        # return self.csv.shape[0]
        return 100

    def __getitem__(self, index):
        pth = os.path.join('data',self.csv.loc[index, 'Path'])
        labels = self.csv.loc[index, self.labels_cols]

        img = Image.open(pth)
        
        if self.transforms != None:
            img = self.transforms(img)
        img = self.normalize(self.img_tensorify(img))
        return img, torch.Tensor(labels)

    def cal_mean_std(self):
        #will take quite a while
        pths = self.csv.loc[:,'Path']
        pixels = np.array([])
        for pth in pths:
            pixels = np.concatenate((pixels, np.array(Image.open(os.path.join('data',pth))).reshape(-1)))
            
        mean = pixels.mean()
        std = pixels.std()
        return mean, std
