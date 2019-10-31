def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from PIL import Image
from PIL.ImageOps import equalize
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import Dataset
import torch
import numpy as np
import sys
import time
import os
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
            self.csv = pd.read_csv('data/CheXpert-v1.0-small/train.csv')
        else:
            self.csv = pd.read_csv('data/CheXpert-v1.0-small/valid.csv')
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
            self.csv['study'] = [row[3] for row in self.csv.Path.str.split('/')]
            self.csv['patient_no'] = [row[2] for row in self.csv.Path.str.split('/')]

        
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
        img = equalize(img)
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


class CheXpertDataset_paired(Dataset):
    def __init__(self, training=True, imputation=1.0, num_classes=14, transform = None, test=None):
        assert((num_classes==14) or (num_classes==5))
        self.num_classes= num_classes
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

        self.csv['study'] = [row[3] for row in self.csv.Path.str.split('/')]
        self.csv['patient_no'] = [row[2] for row in self.csv.Path.str.split('/')]
        self.csv['patient_study'] = [f'{row[2]}_{row[3]}' for row in self.csv.Path.str.split('/')]
                
        
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
        
        frontal, frontal_img, lateral, lateral_img = 0,None,0,None
        
        labels = self.csv.loc[index, self.labels_cols]
        if self.num_classes==5:
            labels = labels[[8,2,6,5,10]]
        
        img = self.load_img(index)
        
        _,h,w = img.shape
        if self.csv.loc[index, 'Frontal/Lateral'] == 'Frontal':
            frontal = 1
            frontal_img = img
        else:
            lateral = 1
            lateral_img = img
            
        #randomly choose another view from the same study
        other_view_indexes = list(self.csv.loc[(self.csv.patient_study == self.csv.patient_study[index]),:].index)
        other_view_indexes.remove(index)
        if len(other_view_indexes) > 0:
            ran = random.choice(other_view_indexes)
            #if same view, just use the original index
            if self.csv.loc[index, 'Frontal/Lateral'] == self.csv.loc[ran, 'Frontal/Lateral']:
                #if original index is frontal, put the other as none
                if frontal ==1:
                    lateral_img = torch.zeros((1,h,w))
                elif lateral == 1:
                    frontal_img = torch.zeros((1,h,w))
            #if random is different view, load it
            else:
                if frontal == 1:
                    lateral_img = self.load_img(ran)
                elif lateral == 1:
                    frontal_img = self.load_img(ran)
        #if no other view available
        else:
            if frontal ==1:
                lateral_img = torch.zeros((1,h,w))
            elif lateral == 1:
                frontal_img = torch.zeros((1,h,w))
            

        
        return torch.stack([frontal_img, lateral_img]), torch.Tensor(labels)

    def cal_mean_std(self):
        #will take quite a while
        pths = self.csv.loc[:,'Path']
        pixels = np.array([])
        for pth in pths:
            pixels = np.concatenate((pixels, np.array(Image.open(os.path.join('data',pth))).reshape(-1)))
            
        mean = pixels.mean()
        std = pixels.std()
        return mean, std

    def load_img(self, index):
        pth = os.path.join('data',self.csv.loc[index, 'Path'])
        img = Image.open(pth)
        if self.transforms != None:
            img = self.transforms(img)
        img = equalize(img)
        img = self.img_tensorify(img)
        # img = self.normalize(img) 
        return img
        
        