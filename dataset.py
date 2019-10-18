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
    def __init__(self, training=True, imputation=1.0, num_classes=14, view='both', transform = None, test=None):
        assert((num_classes==14) or (num_classes==5))
        self.num_classes= num_classes
        self.training = training
        self.transforms = transform
        self.imputation=imputation
        if self.training:
            self.csv = pd.read_csv('temp/train_small.csv')
        else:
            self.csv = pd.read_csv('temp/valid_small.csv')
        if view=='frontal':
            self.csv = self.csv.loc[self.csv['Frontal/Lateral'] == 'Frontal']
            self.csv = self.csv.reset_index(drop=True)
        elif view=='lateral':
             self.csv = self.csv.loc[self.csv['Frontal/Lateral'] == 'Lateral']
             self.csv = self.csv.reset_index(drop=True)
        #if test present overrides train/val
        if test != None:
            self.csv = pd.read_csv(test)
            self.transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
            self.csv['Study'] = [row[3] for row in self.csv.Path.str.split('/')]
            self.csv['Patient_no'] = [row[2] for row in self.csv.Path.str.split('/')]

        
        #change no_finding to 0 if any pathology present
        # pathologies_present_idx = (self.csv.iloc[:,6:18]==1).any(1).index
        #change all pathology to 0 if no_finding is 0
        # self.csv.iloc[pathologies_present_idx, 5] = 0.0 
        # no_finding_idx = (self.csv.loc[:,'No Finding'] == 1).index
        # self.csv.iloc[no_finding_idx,6:18] = 0.0
        self.csv= self.csv.fillna(0.0)
        self.csv= self.csv.replace(-1.0,self.imputation)
        self.labels_cols = class_names
        self.img_tensorify = ToTensor()
        self.normalize = Normalize(mean=[127],std=[74.69])
        
        
        if test == None:
            print('number of samples in dataset', self.csv.shape[0], 'Training:', self.training)
            eda = ''
            for name in class_names:
                eda += f'{name}: '
                eda += str(self.csv[name].value_counts().to_dict())
                eda += '\n'
            print(eda)

    def __len__(self):
        return self.csv.shape[0]
        # return 100

    def __getitem__(self, index):
        pth = os.path.join('data',self.csv.loc[index, 'Path'])
        labels = self.csv.loc[index, self.labels_cols]
        if self.num_classes==5:
            labels = labels[[8,2,6,5,10]]

        img = Image.open(pth)
        
        if self.transforms != None:
            img = self.transforms(img)
        img = self.img_tensorify(img)
        # img = self.normalize(img) 
            
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
