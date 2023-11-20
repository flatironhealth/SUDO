#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:08:19 2023

@author: dani.kiyasseh

This script is used to:
1) extract and store features from all splits (i.e., train, id_val, test) of the the WILDS-Camelyon dataset
2) perform inference on the test set and store corresponding probability values
"""

# [USER MUST MODIFY] path_to_wilds_folder 
path_to_wilds_folder = '/Users/dani.kiyasseh/Desktop/wilds'
# navigate to the examples/ directory

# Import the necessary packages (taken from the WILDS repo)
import os
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
from collections import defaultdict

try:
    import wandb
except Exception as e:
    pass

import wilds
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSPseudolabeledSubset

from utils import set_seed, Logger, BatchLogger, log_config, ParseKwargs, load, initialize_wandb, log_group_data, parse_bool, get_model_prefix, move_to
from train import train, evaluate, infer_predictions
from algorithms.initializer import initialize_algorithm, infer_d_out
from transforms import initialize_transform
from models.initializer import initialize_model
from configs.utils import populate_defaults
import configs.supported as supported

import torch.multiprocessing

from tqdm import tqdm
import numpy as np
import pickle

#%%

parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets)#, required=True)
parser.add_argument('--algorithm')#, required=True, choices=supported.algorithms)
parser.add_argument('--root_dir')#, required=True,
                    #help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')

# Dataset
parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                    help='If true, tries to download the dataset if it does not exist in root_dir.')
parser.add_argument('--frac', type=float, default=1.0,
                    help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
parser.add_argument('--version', default=None, type=str, help='WILDS labeled dataset version number.')

# Unlabeled Dataset
parser.add_argument('--unlabeled_split', default=None, type=str, choices=wilds.unlabeled_splits,  help='Unlabeled split to use. Some datasets only have some splits available.')
parser.add_argument('--unlabeled_version', default=None, type=str, help='WILDS unlabeled dataset version number.')
parser.add_argument('--use_unlabeled_y', default=False, type=parse_bool, const=True, nargs='?', 
                    help='If true, unlabeled loaders will also the true labels for the unlabeled data. This is only available for some datasets. Used for "fully-labeled ERM experiments" in the paper. Correct functionality relies on CrossEntropyLoss using ignore_index=-100.')

# Loaders
parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
parser.add_argument('--unlabeled_loader_kwargs', nargs='*', action=ParseKwargs, default={})
parser.add_argument('--train_loader', choices=['standard', 'group'])
parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?', help='If true, sample examples such that batches are uniform over groups.')
parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?', help='If true, enforce groups sampled per batch are distinct.')
parser.add_argument('--n_groups_per_batch', type=int)
parser.add_argument('--unlabeled_n_groups_per_batch', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--unlabeled_batch_size', type=int)
parser.add_argument('--eval_loader', choices=['standard'], default='standard')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of batches to process before stepping optimizer and schedulers. If > 1, we simulate having a larger effective batch size (though batchnorm behaves differently).')

# Model
parser.add_argument('--model', choices=supported.models)
parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for model initialization passed as key1=value1 key2=value2')
parser.add_argument('--noisystudent_add_dropout', type=parse_bool, const=True, nargs='?', help='If true, adds a dropout layer to the student model of NoisyStudent.')
parser.add_argument('--noisystudent_dropout_rate', type=float)
parser.add_argument('--pretrained_model_path', default=None, type=str, help='Specify a path to pretrained model weights')
parser.add_argument('--load_featurizer_only', default=False, type=parse_bool, const=True, nargs='?', help='If true, only loads the featurizer weights and not the classifier weights.')

# NoisyStudent-specific loading
parser.add_argument('--teacher_model_path', type=str, help='Path to NoisyStudent teacher model weights. If this is defined, pseudolabels will first be computed for unlabeled data before anything else runs.')

# Transforms
parser.add_argument('--transform', choices=supported.transforms)
parser.add_argument('--additional_train_transform', choices=supported.additional_transforms, help='Optional data augmentations to layer on top of the default transforms.')
parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
parser.add_argument('--resize_scale', type=float)
parser.add_argument('--max_token_length', type=int)
parser.add_argument('--randaugment_n', type=int, help='Number of RandAugment transformations to apply.')

# Objective
parser.add_argument('--loss_function', choices=supported.losses)
parser.add_argument('--loss_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for loss initialization passed as key1=value1 key2=value2')

# Algorithm
parser.add_argument('--groupby_fields', nargs='+')
parser.add_argument('--group_dro_step_size', type=float)
parser.add_argument('--coral_penalty_weight', type=float)
parser.add_argument('--dann_penalty_weight', type=float)
parser.add_argument('--dann_classifier_lr', type=float)
parser.add_argument('--dann_featurizer_lr', type=float)
parser.add_argument('--dann_discriminator_lr', type=float)
parser.add_argument('--afn_penalty_weight', type=float)
parser.add_argument('--safn_delta_r', type=float)
parser.add_argument('--hafn_r', type=float)
parser.add_argument('--use_hafn', default=False, type=parse_bool, const=True, nargs='?')
parser.add_argument('--irm_lambda', type=float)
parser.add_argument('--irm_penalty_anneal_iters', type=int)
parser.add_argument('--self_training_lambda', type=float)
parser.add_argument('--self_training_threshold', type=float)
parser.add_argument('--pseudolabel_T2', type=float, help='Percentage of total iterations at which to end linear scheduling and hold lambda at the max value')
parser.add_argument('--soft_pseudolabels', default=False, type=parse_bool, const=True, nargs='?')
parser.add_argument('--algo_log_metric')
parser.add_argument('--process_pseudolabels_function', choices=supported.process_pseudolabels_functions)

# Model selection
parser.add_argument('--val_metric')
parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

# Optimization
parser.add_argument('--n_epochs', type=int)
parser.add_argument('--optimizer', choices=supported.optimizers)
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--max_grad_norm', type=float)
parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for optimizer initialization passed as key1=value1 key2=value2')

# Scheduler
parser.add_argument('--scheduler', choices=supported.schedulers)
parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for scheduler initialization passed as key1=value1 key2=value2')
parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
parser.add_argument('--scheduler_metric_name')

# Evaluation
parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
parser.add_argument('--eval_splits', nargs='+', default=[])
parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

# Misc
parser.add_argument('--device', type=int, nargs='+', default=[0])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_dir', default='./logs')
parser.add_argument('--log_every', default=50, type=int)
parser.add_argument('--save_step', type=int)
parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False, help='Whether to resume from the most recent saved model in the current log_dir.')

# Weights & Biases
parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
parser.add_argument('--wandb_api_key_path', type=str,
                    help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

config = parser.parse_args()

# Provide paths to model parameters and the data
config.load_featurizer_only = True
config.pretrained_model_path = os.path.join(path_to_wilds_folder,'logs/camelyon17_seed_0_epoch_best_model.pth')
config.dataset = "camelyon17"
config.algorithm = "ERM"
config.root_dir = "../data"
config.target_resolution = [96, 96]
config.model = "densenet121"
config.transform = "image_base"
config.version = None
config.split_scheme = "official"
config.download = False
config.eval_loader = "standard"
config.device = None
config.seed = 0

# Define which of the two tasks you are looking to achieve (see header) 
purpose = 'feature extraction' # options: feature extraction | inference
data_split = 'train' # options: train | id_val | test 
batch_size = 128

set_seed(config.seed)
model = initialize_model(config,d_out=1)

if purpose == 'feature extraction':
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()

# Define the dataset and the data transform
full_dataset = wilds.get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        **config.dataset_kwargs)

eval_transform = initialize_transform(
        transform_name=config.transform,
        config=config,
        dataset=full_dataset,
        is_training=False)

# Get the dataloader 
set_seed(config.seed)
data = full_dataset.get_subset(data_split,transform=eval_transform)
loader = get_train_loader(config.eval_loader, data, batch_size=batch_size)

# Number of samples in the training, validation, and test splits
ntrain = 302436
nval = 33560
ntest = 85054

if data_split == 'train':
    start_idx = 0
elif data_split == 'id_val':
    start_idx = ntrain
elif data_split == 'test':
    start_idx = ntrain+nval
    
# Perform a forward pass through the model and store features (or probabilities) in a dictionary
features_dict = dict()
labels_dict = dict()
batch = 1
for idx,(x,y,meta) in tqdm(enumerate(loader)):
    curr_features = model(x)
    filenames = np.arange(start_idx+(batch_size*idx),start_idx+(batch_size*(idx+1)))
    curr_features_dict = dict(zip(filenames,curr_features.detach().numpy()))
    curr_labels_dict = dict(zip(filenames,y.numpy()))
    features_dict = {**features_dict,**curr_features_dict}
    labels_dict = {**labels_dict,**curr_labels_dict}
    batch += 1
    if batch == 200: # to limit the number of training samples we work with
        if data_split == 'train':
            break

# Save the features (or probabilities) for later use in SUDO experiments
data_path = os.path.join(path_to_wilds_folder,'camelyon17/%s' % (data_split))
if not os.path.exists(data_path):
    os.makedirs(data_path)
savename = os.path.join(data_path,'features' if purpose == 'feature extraction' else 'probs')
with open(savename,'wb') as f:
    pickle.dump(features_dict,f)
savename = os.path.join(data_path,'labels')
with open(savename,'wb') as f:
    pickle.dump(labels_dict,f)
    
# Generate CSV of prediction probabilities for later use in SUDO experiments
if purpose == 'inference':
    assert data_split == 'test'
    import pandas as pd
    from scipy.special import expit
    
    probs = pd.DataFrame.from_dict(features_dict).T
    probs.reset_index(inplace=True)
    probs.index = probs['index']
    probs.columns = ['index','logit']
    probs['Prob'] = probs['logit'].apply(lambda logit:expit(logit))
    
    meta_df = pd.DataFrame.from_dict(labels_dict,orient='index')
    probs['Label'] = meta_df
    probs = probs[['logit','Prob','Label']]
    
    probs.to_csv(os.path.join(data_path,'camelyon17_test_Probs.csv'))



