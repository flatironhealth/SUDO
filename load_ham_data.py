#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:57:17 2023

@author: dani.kiyasseh

This script is used to:
1) load the HAM data to support feature extraction
"""

# [USER MUST MODIFY] path_to_ham_data
path_to_ham_data = '/Users/dani.kiyasseh/Desktop/Data/HAM'

import torch
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import os
import pandas as pd
import numpy as np
from PIL import Image
import random

means = [0.485, 0.456, 0.406]
stds  = [0.229, 0.224, 0.225]
test_transform = T.Compose([
    lambda x: x.convert('RGB'),
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize(mean=means, std=stds)
])

class HAM_dataset(torch.utils.data.Dataset):
    
    def __init__(self,purpose='feature extraction'):
        super(HAM_dataset,self).__init__()
        data_path = path_to_ham_data
        csv_path = 'HAM10000_metadata.csv'
        cutoff_frame = 29306 # for part2 HAM images
        if purpose in ['feature extraction','inference']:
            df = pd.read_csv(os.path.join(data_path,csv_path))
            self.df = df
        elif purpose == 'SUDO':
            df = pd.read_csv(os.path.join(data_path,csv_path))
            self.df = df
            df = self.get_split_df()
        self.purpose = purpose
        self.data_path = data_path
        self.cutoff_frame = cutoff_frame
    
    def get_split_df(self):
        random.seed(0)
        df = self.df
        df['dx_binary'] = df['dx'].apply(lambda dx:'malignant' if dx in ['mel','bcc'] else 'benign')
        df_benign = df[df['dx_binary']=='benign'].sample(1627,replace=False,random_state=0)
        df_malignant = df[df['dx_binary']=='malignant']
        df = pd.concat((df_benign,df_malignant),0)
        indices = list(range(len(df)))
        random.shuffle(indices)
        train_ratio = 0.8
        train_nsamples = int(train_ratio*len(df))
        train_indices, val_indices = indices[:train_nsamples], indices[train_nsamples:]
        train_df, val_df = df.iloc[train_indices,:], df.iloc[val_indices,:]
        data = {'train':train_df,'val':val_df}
        return data
    
    def __getitem__(self,idx):
        row = self.df.iloc[idx,:]
        image_file = row['image_id']
        image_num = int(image_file.split('_')[1])
        folder = 'HAM10000_images_part_1' if image_num < self.cutoff_frame else 'HAM10000_images_part_2'
        image_path = os.path.join(self.data_path,folder,image_file + '.jpg')
        im = Image.open(image_path)
        x = test_transform(im)
        dx = row['dx']
        dx = torch.tensor(1 if dx in ['mel','bcc'] else 0,dtype=torch.int)
        return image_file, x, dx
    
    def __len__(self):
        return len(self.df)

def get_HAM_dataloader(purpose):
    dataset = HAM_dataset(purpose)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=False)
    return dataloader