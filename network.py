#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 10:26:30 2021

@author: dani.kiyasseh

This script is used to:
1) define the neural network used for training and inference on any dataset
2) define the training and evaluation functions
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from collections import defaultdict
from tqdm import tqdm
from prepare_miscellaneous_structured import get_inputs_and_outputs
from tabulate import tabulate
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.special import softmax

class Network(nn.Module):
    
    def __init__(self,ninputs,task):
        super(Network,self).__init__()
        d1 = 100 #100
        d2 = 50 #50
        if task == '0 vs. 1 vs. 234':
            d3 = 3
        elif task in ['01 vs. 234','0 vs. 234','0']:
            d3 = 1
        self.linear1 = nn.Linear(ninputs,d1)
        self.linear2 = nn.Linear(d1,d2)
        self.linear4 = nn.Linear(d2,d3)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        h1 = self.relu(self.linear1(x))
        h2 = self.relu(self.linear2(h1))
        out = self.linear4(h2)
        return out, h2
    
class StructuredDataset(Dataset):
    
    def __init__(self,inputs_and_outputs,phase):
        phase_df = inputs_and_outputs[phase] #pd.DataFrame
        self.inputs = phase_df['inputs']
        self.outputs = phase_df['outputs']
        
    def __getitem__(self,idx):
        x = self.inputs.iloc[idx,:].to_numpy()
        y = self.outputs.iloc[idx]
        x, y = torch.tensor(x).float(), torch.tensor(y)
        return x, y
        
    def __len__(self):
        return self.inputs.shape[0]
        
def loadDataloader(inputs_and_outputs,phases,batch_size=32):
    shuffle_dict = {phase:True if phase == 'train' else False for phase in phases}
    #phases = ['train','val'] 
    dataset_dict = {phase: StructuredDataset(inputs_and_outputs,phase) for phase in phases}
    dataloader_dict = {phase: DataLoader(dataset_dict[phase],batch_size,shuffle=shuffle_dict[phase]) for phase in phases}
    return dataloader_dict
    
def train(inputs_and_outputs,dict_df,eval_type='filler',task='0 vs. 1 vs. 234',suffix='',goal='train and evaluate model',batch_size=32,lr=1e-4,phases=['train','val'],max_epochs=40,writer=''):
    """ Train the neural network for several epochs
    
    Args:
        inputs_and_outputs (dict): dictionary containing the inputs and outputs
        dict_df (dict): dictionary containing the original inputs and outputs
        task (string): task to achieve (e.g., binary classification)
    
    Returns:
        metrics_dict (dict): dictionary containing the results of the network
    """
    if goal in ['perform inference']:
        ninputs = inputs_and_outputs['inference']['inputs'].shape[1] #- 5
    elif goal in ['evaluate model']:
        ninputs = inputs_and_outputs['val']['inputs'].shape[1]
    else:
        ninputs = inputs_and_outputs['train']['inputs'].shape[1] #- 5
        
    dataloader = loadDataloader(inputs_and_outputs,phases,batch_size)
    model = Network(ninputs,task)
    
    if goal in ['evaluate model','perform inference']:
        model = torch.load('NN_Model%s' % suffix)
        print('Model Loaded...')
    
    if task == '0 vs. 1 vs. 234':
        criterion = nn.CrossEntropyLoss()
    elif task in ['01 vs. 234','0 vs. 234','0']:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    for epoch in range(max_epochs):
        print('Epoch %i/%i' % (epoch,max_epochs))
        loss_dict = defaultdict()
        acc_dict = defaultdict()
        auc_dict = defaultdict()
        probs_dict = defaultdict(list)
        output_dict = defaultdict(list)
        reps_dict = defaultdict(list)
        
        run_loss = 0
        for phase in phases:
            with torch.set_grad_enabled(phase=='train'):
                b = 0
                for inputs,outputs in tqdm(dataloader[phase]):
                    probs,reps = model(inputs)
                    if task in ['01 vs. 234','0 vs. 234','0']:
                        probs = probs.view_as(outputs)
                        outputs = outputs.type(torch.float)
                        
                    if goal not in ['perform inference']:
                        loss = criterion(probs,outputs)
                        run_loss += (1/(b+1))*(loss.detach().numpy() - run_loss)

                    probs_dict[phase].extend(probs.detach().numpy())
                    output_dict[phase].extend(outputs.detach().numpy())
                    reps_dict[phase].extend(reps.detach().numpy())
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            
                    b += 1 
            #break
        #break
            if goal not in ['perform inference']:
                loss_dict[phase] = run_loss
                metrics = evaluate_model(probs_dict,output_dict,dict_df[phase],phase,eval_type=eval_type,task=task)
                acc_dict[phase] = metrics[0] #accuracy_score(np.array(output_dict[phase]),np.argmax(softmax(np.array(probs_dict[phase]),1),1))
                auc_dict[phase] = metrics[1] #roc_auc_score(np.array(output_dict[phase]),softmax(np.array(probs_dict[phase]),1),multi_class='ovr')    
            
                metrics_dict = {'loss':loss_dict,'acc':acc_dict,'auc':auc_dict}
                names = []
                values = []
                for metric,values_dict in metrics_dict.items():
                    for phase,value in values_dict.items():
                        name = phase + '-' + metric
                        names.append(name)
                        values.append('%.3f' % value)
                
                writer.add_scalar('Loss/%s' % phase, loss_dict[phase], epoch)
                writer.add_scalar('AUC/%s' % phase, auc_dict[phase], epoch)
        
        if goal not in ['perform inference']:
            table = tabulate([names,values])
            print(table)
        elif goal in ['perform inference']:
            metrics_dict = None
        
        if goal in ['train and evaluate model']:
            torch.save(model,'NN_Model')
            print('Model Saved')
    
    return metrics_dict, reps_dict, probs_dict

def train_for_OOD(inputs_and_outputs,dict_df,eval_type='filler',task='0 vs. 1 vs. 234',goal='perform inference',batch_size=32,lr=1e-4,phases=['train','val'],max_epochs=40,writer='',seed=1):
    """ Train the neural network for out-of-distribution (OOD) detection
    
    Args:
        inputs_and_outputs (dict): dictionary containing the inputs and outputs
        dict_df (dict): dictionary containing the original inputs and outputs
        task (string): task to achieve (e.g., binary classification)
    
    Returns:
        probs_dict (dict): dictionary containing the probabilities of the network outputs
    """
    if task in ['0']:
        if goal in ['perform inference']:
            ninputs = inputs_and_outputs['inference']['inputs'].shape[1] #- 5
        elif goal in ['train and evaluate model']:
            ninputs = inputs_and_outputs['train']['inputs'].shape[1]
            
    dataloader = loadDataloader(inputs_and_outputs,phases,batch_size)
    model = Network(ninputs,task)
    
    if task in ['0']:
        if goal in ['perform inference']:
            if 'inference' in phases:
                model = torch.load('OC_OOD_Model%s' % seed)
                print('Model Loaded')
        elif goal in ['train and evaluate model']:
            if 'val' in phases:
                if 'OC_OOD_Model%s' % seed in os.listdir():
                    model = torch.load('OC_OOD_Model%s' % seed)
                    print('Model Loaded')                    
    
    if task == '0 vs. 1 vs. 234':
        criterion = nn.CrossEntropyLoss()
    elif task in ['01 vs. 234','0 vs. 234','0']:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    for epoch in range(max_epochs):
        print('Epoch %i/%i' % (epoch,max_epochs))
        probs_dict = defaultdict(list)
        output_dict = defaultdict(list)
        
        run_loss = 0
        for phase in phases:
            with torch.set_grad_enabled(phase=='train'):
                b = 0
                for inputs,outputs in tqdm(dataloader[phase]):
                    probs, _ = model(inputs)
                    if task in ['01 vs. 234','0 vs. 234','0']:
                        probs = probs.view_as(outputs)
                        outputs = outputs.type(torch.float)
                    loss = criterion(probs,outputs)

                    probs_dict[phase].extend(probs.detach().numpy())
                    output_dict[phase].extend(outputs.detach().numpy())
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            
                    run_loss += (1/(b+1))*(loss.detach().numpy() - run_loss)
                    b += 1 
    
    if task in ['0']:
        if goal in ['train and evaluate model']:
            torch.save(model,'OC_OOD_Model%s' % seed)
            print('Model Saved')
    
    return probs_dict


#%%
#train(dict_df)

def evaluate_model(probs_dict,output_dict,df,phase,eval_type='instance',task='0 vs. 1 vs. 234'):
    """ Evaluate the performance of the model. 
        Note this is only used for neural network models. 
    
    Args:
        model (model object): model which has already been trained
        inputs (pd.DataFrame): inputs to evaluate on
        outputs (pd.DataFrame): ground-truth labels
        eval_type (string): determines how to perform evaluation
        task (string): task of interest e.g., binary classification, etc. 
    
    Returns:
        acc, auc (floats): performance metrics
    """

    if task == '0 vs. 1 vs. 234':
        probs = softmax(np.array(probs_dict[phase]),1)
        preds = np.argmax(probs,1)
    elif task in ['01 vs. 234','0 vs. 234','0']:
        probs = np.expand_dims(np.array(probs_dict[phase]),-1)
        preds = (probs > 0.5).astype(int)
    outputs = np.array(output_dict[phase])
    cols_to_groupby = eval_type.split('-')
    
    if task in ['01 vs. 234','0 vs. 234','0']:
        probs = probs[:,-1] #take single column reflecting prob of positive class
        class_bins = [0,1,2]
        nclasses = 1
        col_num = 0
    elif task == '0 vs. 1 vs. 234':
        class_bins = [0,1,2,3]
        nclasses = 3    
        col_num = [0,1,2]
    
    if eval_type in ['instance']:
        preds_df = pd.DataFrame(preds,columns=['preds'],index=df.index)
        preds_df[cols_to_groupby] = df[cols_to_groupby]
        preds_df = preds_df.groupby(by=cols_to_groupby).progress_apply(lambda x:np.argmax(np.histogram(x,class_bins)[0])).reset_index() #specific to 3 class problem
        preds = preds_df.iloc[:,-1].astype(int).values
    
        probs_df = pd.DataFrame(probs,index=df.index)
        probs_df[cols_to_groupby] = df[cols_to_groupby]
        probs_df = probs_df.groupby(by=cols_to_groupby).progress_apply(lambda x:x.mean()).iloc[:,:nclasses].reset_index()
        probs = probs_df.loc[:,col_num] #specific to 3 class problem

        outputs_df = pd.DataFrame(outputs,index=df.index)
        outputs_df[cols_to_groupby] = df[cols_to_groupby] #order matters for this line 
        outputs_df = outputs_df.groupby(by=cols_to_groupby).mean().reset_index()
        outputs = outputs_df.iloc[:,-1].astype(int).values
    
    acc = accuracy_score(outputs,preds)     
    auc = roc_auc_score(outputs,probs,multi_class='ovr')
    #prec = precision_score(outputs,preds,average='macro')
    #rec = recall_score(outputs,preds,average='macro')
    
    return acc, auc







        
