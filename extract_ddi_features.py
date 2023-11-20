#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:09:36 2023

@author: dani.kiyasseh

This script is used to:
1) extract and store features from the Stanford DDI dataset and the HAM10000 dataset
2) perform inference on the test set and store corresponding probability values
"""

# [USER MUST MODIFY] root_path is where the DDI and HAM10000 data are located
root_path = '/Users/dani.kiyasseh/Desktop/Data'
path_to_ddi_metadata = os.path.join(root_path,'DDI/ddi_metadata.csv')

from load_ddi_model import load_model
from load_ddi_data import get_DDI_dataloader
from load_ham_data import get_HAM_dataloader
from tqdm import tqdm
import pickle
import os

# Define the task you are looking to achieve (see header)
purpose = 'inference' # options: feature extraction | inference

# Define the model you would like to use
model_type = 'DeepDerm' # options: 'HAM10000' | 'DeepDerm'
model = load_model(model_type,extract_features=True if purpose == 'feature extraction' else False)
model.eval()

# Load the desired dataset
dataset_name = 'DDI' # options: DDI | HAM
if dataset_name == 'DDI':
    dataloader = get_DDI_dataloader()
elif dataset_name == 'HAM':
    dataloader = get_HAM_dataloader(purpose)

# Perform a forward pass through the model and extract features
features_dict = dict()
batch = 1
for filenames,x,y in tqdm(dataloader):
    curr_features = model(x)
    curr_features_dict = dict(zip(filenames,curr_features.detach().numpy()))
    features_dict = {**features_dict,**curr_features_dict}
    batch += 1

# Save features for later use in SUDO experiments
save_folder = 'Features' if purpose == 'feature extraction' else 'Probs'
path_to_save_folder = os.path.join(root_path,'%s/%s' % (dataset_name,save_folder))
if not os.path.exists(path_to_save_folder):
    os.makedirs(path_to_save_folderh)
savename = os.path.join(path_to_save_folder,'features_%s' % model_type if purpose == 'feature extraction' else 'probs')
with open(savename,'wb') as f:
    pickle.dump(features_dict,f)
    
# Generate CSV of prediction probabilities for later use in SUDO experiments
if purpose == 'inference':
    import pandas as pd
    from scipy.special import expit
    
    probs = pd.DataFrame.from_dict(features_dict).T
    probs.reset_index(inplace=True)
    probs.columns = ['file','0','1']
    probs = probs.loc[:,['file','1']]
    probs.columns = ['file','logit']
    probs['Prob'] = probs['logit'].apply(lambda logit:expit(logit))

    meta_df = pd.read_csv(path_to_ddi_metadata,index_col=0)
    probs['Label'] = meta_df['malignant'].apply(lambda label:1 if label==True else 0)
    probs['skin_tone'] = meta_df['skin_tone']
    
    probs.to_csv('DDI_Probs_%s.csv' % model_type)





    


