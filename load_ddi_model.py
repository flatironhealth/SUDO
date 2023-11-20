#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:57:17 2023

@author: dani.kiyasseh

This script is used to:
1) load the model that was used by the original authors for inference on the DDI data
"""

import os
import torch
import torchvision
import torch.nn as nn

# google drive paths to our models
MODEL_WEB_PATHS = {
# base form of models trained on skin data
'HAM10000':'https://drive.google.com/uc?id=1ToT8ifJ5lcWh8Ix19ifWlMcMz9UZXcmo',
'DeepDerm':'https://drive.google.com/uc?id=1OLt11htu9bMPgsE33vZuDiU5Xe4UqKVJ',

# robust training algorithms
'GroupDRO':'https://drive.google.com/uc?id=193ippDUYpMaOaEyLjd1DNsOiW0aRXL75',
'CORAL':   'https://drive.google.com/uc?id=18rMU0nRd4LiHN9WkXoDROJ2o2sG1_GD8',
'CDANN':   'https://drive.google.com/uc?id=1PvvgQVqcrth840bFZ3ddLdVSL7NkxiRK',
}

# thresholds determined by maximizing F1-score on the test split of the train 
#   dataset for the given algorithm
MODEL_THRESHOLDS = {
    'HAM10000':0.733,
    'DeepDerm':0.687,
    # robust training algorithms
    'GroupDRO':0.980,
    'CORAL':0.990,
    'CDANN':0.980,
}

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def load_model(model_name, save_dir="DDI-models", download=True, extract_features=False):
    """Load the model and download if necessary. Saves model to provided save 
    directory."""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name.lower()}.pth")
    if not os.path.exists(model_path):
        if not download:
            raise Exception("Model not downloaded and download option not"\
                            " enabled.")
        else:
            # Requires installation of gdown (pip install gdown)
            import gdown
            gdown.download(MODEL_WEB_PATHS[model_name], model_path)
    model = torchvision.models.inception_v3(pretrained=False, transform_input=True)
    model.fc = torch.nn.Linear(2048, 2)
    model.AuxLogits.fc = torch.nn.Linear(768, 2)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model._ddi_name = model_name
    model._ddi_threshold = MODEL_THRESHOLDS[model_name]
    model._ddi_web_path = MODEL_WEB_PATHS[model_name]
    
    if extract_features == True:
        model.fc = Identity()
    
    return model