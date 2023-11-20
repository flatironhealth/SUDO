#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:57:17 2023

@author: dani.kiyasseh

This script is used to:
1) load the DDI data to support feature extraction
"""

# [USER MUST MODIFY] path_to_ddi_data
path_to_ddi_data = '/Users/dani.kiyasseh/Desktop/Data/DDI'

import torch
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import os
import pandas as pd
import numpy as np
from PIL import Image

means = [0.485, 0.456, 0.406]
stds  = [0.229, 0.224, 0.225]
test_transform = T.Compose([
    lambda x: x.convert('RGB'),
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize(mean=means, std=stds)
])

class DDI_dataset(torch.utils.data.Dataset):
    
    def __init__(self):
        super(DDI_dataset,self).__init__()
        data_path = path_to_ddi_data
        csv_path = 'ddi_metadata.csv'
        df = pd.read_csv(os.path.join(data_path,csv_path),index_col=0)
        self.df = df
        self.data_path = data_path
    
    def __getitem__(self,idx):
        row = self.df.iloc[idx,:]
        image_file = row['DDI_file']
        image_path = os.path.join(self.data_path,image_file)
        im = Image.open(image_path)
        x = test_transform(im)
        dx = torch.tensor(row['malignant'],dtype=torch.int)
        return image_file, x, dx
    
    def __len__(self):
        return len(self.df)

def get_DDI_dataloader():
    dataset = DDI_dataset()
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=False)
    return dataloader

#class DDI_Dataset(ImageFolder):
    _DDI_download_link = "https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965"
    """DDI Dataset.

    Args:
        root     (str): Root directory of dataset.
        csv_path (str): Path to the metadata CSV file. Defaults to `{root}/ddi_metadata.csv`
        transform     : Function to transform and collate image input. (can use test_transform from this file) 
    """
    def __init__(self, root, csv_path=None, download=True, transform=None, *args, **kwargs):
        if csv_path is None:
            csv_path = os.path.join(root, "ddi_metadata.csv")
        if not os.path.exists(csv_path) and download:
            raise Exception(f"Please visit <{DDI_Dataset._DDI_download_link}> to download the DDI dataset.")
        assert os.path.exists(csv_path), f"Path not found <{csv_path}>."
        super(DDI_Dataset, self).__init__(root, *args, transform=transform, **kwargs)
        self.annotations = pd.read_csv(csv_path)
        m_key = 'malignant'
        if m_key not in self.annotations:
            self.annotations[m_key] = self.annotations['malignancy(malig=1)'].apply(lambda x: x==1)

    def __getitem__(self, index):
        img, target = super(DDI_Dataset, self).__getitem__(index)
        path = self.imgs[index][0]        
        annotation = dict(self.annotations[self.annotations.DDI_file==path.split("/")[-1]])
        target = int(annotation['malignant'].item()) # 1 if malignant, 0 if benign
        skin_tone = annotation['skin_tone'].item() # Fitzpatrick- 12, 34, or 56
        return path, img, target, skin_tone

    """Return a subset of the DDI dataset based on skin tones and malignancy of lesion.

    Args:
        skin_tone    (list of int): Which skin tones to include in the subset. Options are {12, 34, 56}.
        diagnosis    (list of str): Include malignant and/or benign images. Options are {"benign", "malignant"}
    """
    def subset(self, skin_tone=None, diagnosis=None):
        skin_tone = [12, 34, 56] if skin_tone is None else skin_tone
        diagnosis = ["benign", "malignant"] if diagnosis is None else diagnosis
        for si in skin_tone: 
            assert si in [12,34,56], f"{si} is not a valid skin tone"
        for di in diagnosis: 
            assert di in ["benign", "malignant"], f"{di} is not a valid diagnosis"
        indices = np.where(self.annotations['skin_tone'].isin(skin_tone) & \
                           self.annotations['malignant'].isin([di=="malignant" for di in diagnosis]))[0]
        return Subset(self, indices)