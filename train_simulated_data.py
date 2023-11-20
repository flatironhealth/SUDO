#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:27:26 2023

@author: dani.kiyasseh

This script is used to:
1) conduct the SUDO experiments on the simulated dataset
"""

# [USER MUST MODIFY] 
path_to_save_folder = '/Users/dani.kiyasseh/Desktop/Data/SIM/Results'
path_to_sim_probs = '/Users/dani.kiyasseh/Desktop/Data/SIM/Probs'
path_to_current_scripts = '/Users/dani.kiyasseh/Desktop/Scripts'

import os
new_path = path_to_current_scripts
import sys
import pickle
import random
from operator import itemgetter
if new_path not in sys.path:
    sys.path.append(new_path)
import pandas as pd
from scipy.special import expit
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.linear_model import LogisticRegression, LinearRegression
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns


from prepare_miscellaneous import load_model, train_model                                                        
from network import train, train_for_OOD
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
path = path_to_sim_probs
os.chdir(path)

#%%
def sample_gaussian_points(mean=[0,0],cov=[[0.1,0],[0,0.1]],nsamples=100):
    np.random.seed(0)
    random.seed(0)
    return np.random.multivariate_normal(mean,cov,nsamples)

def get_wild_data(setting,nsamples=1000):
    if setting == 'out-of-domain':
        nsamples = nsamples # validation set
        mean1, mean2 = [2,-1], [3,0]
        cov1, cov2 = [[1,0],[0,1]], [[1,0],[0,1]]
        data_class1 = pd.DataFrame(sample_gaussian_points(mean=mean1,cov=cov1,nsamples=nsamples))
        data_class1['Label'] = 0
        data_class2 = pd.DataFrame(sample_gaussian_points(mean=mean2,cov=cov2,nsamples=nsamples))
        data_class2['Label'] = 1
        wild_data = pd.concat((data_class1,data_class2),0)
    elif setting == 'out-of-domain-third-class':
        nsamples = nsamples # validation set
        mean1, mean2, mean3 = [2,-1], [3,0], [3,-1]
        cov1, cov2, cov3 = [[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]] 
        data_class1 = pd.DataFrame(sample_gaussian_points(mean=mean1,cov=cov1,nsamples=nsamples))
        data_class1['Label'] = 0
        data_class2 = pd.DataFrame(sample_gaussian_points(mean=mean2,cov=cov2,nsamples=nsamples))
        data_class2['Label'] = 1
        data_class3 = pd.DataFrame(sample_gaussian_points(mean=mean3,cov=cov3,nsamples=nsamples))
        data_class3['Label'] = 2
        wild_data = pd.concat((data_class1,data_class2,data_class3),0)
    elif setting == 'out-of-domain-one-class':
        nsamples = nsamples # validation set
        mean1 = [2,-1]
        cov1 = [[1,0],[0,1]]
        data_class1 = pd.DataFrame(sample_gaussian_points(mean=mean1,cov=cov1,nsamples=nsamples))
        data_class1['Label'] = 0
        wild_data = data_class1.copy() 
    elif setting == 'out-of-domain-imbalance':
        mean1, mean2 = [2,-1], [3,0]
        cov1, cov2 = [[1,0],[0,1]], [[1,0],[0,1]]
        nsamples1, nsamples2 = nsamples*4, nsamples//2
        data_class1 = pd.DataFrame(sample_gaussian_points(mean=mean1,cov=cov1,nsamples=nsamples1))
        data_class1['Label'] = 0
        data_class2 = pd.DataFrame(sample_gaussian_points(mean=mean2,cov=cov2,nsamples=nsamples2))
        data_class2['Label'] = 1
        wild_data = pd.concat((data_class1,data_class2),0)
    elif setting == 'out-of-domain-shift-pos':
        mean1, mean2 = [2,-1], [2,-1]
        cov1, cov2 = [[1,0],[0,1]], [[1,0],[0,1]]
        nsamples1, nsamples2 = nsamples, nsamples
        data_class1 = pd.DataFrame(sample_gaussian_points(mean=mean1,cov=cov1,nsamples=nsamples1))
        data_class1['Label'] = 0
        data_class2 = pd.DataFrame(sample_gaussian_points(mean=mean2,cov=cov2,nsamples=nsamples2))
        data_class2['Label'] = 1
        wild_data = pd.concat((data_class1,data_class2),0)
    elif setting == 'out-of-domain-shift-pos+':
        mean1, mean2 = [2,-1], [1,-2]
        cov1, cov2 = [[1,0],[0,1]], [[1,0],[0,1]]
        nsamples1, nsamples2 = nsamples, nsamples
        data_class1 = pd.DataFrame(sample_gaussian_points(mean=mean1,cov=cov1,nsamples=nsamples1))
        data_class1['Label'] = 0
        data_class2 = pd.DataFrame(sample_gaussian_points(mean=mean2,cov=cov2,nsamples=nsamples2))
        data_class2['Label'] = 1
        wild_data = pd.concat((data_class1,data_class2),0)        
    elif setting == 'out-of-domain-shift-pos++':
        mean1, mean2 = [2,-1], [0,-3]
        cov1, cov2 = [[1,0],[0,1]], [[1,0],[0,1]]
        nsamples1, nsamples2 = nsamples, nsamples
        data_class1 = pd.DataFrame(sample_gaussian_points(mean=mean1,cov=cov1,nsamples=nsamples1))
        data_class1['Label'] = 0
        data_class2 = pd.DataFrame(sample_gaussian_points(mean=mean2,cov=cov2,nsamples=nsamples2))
        data_class2['Label'] = 1
        wild_data = pd.concat((data_class1,data_class2),0) 
        
    return wild_data

#%%
def evaluate_model(model,inputs,outputs,df):
    """ Evaluate the performance of the model
    
    Args:
        model (model object): model which has already been trained
        inputs (pd.DataFrame): inputs to evaluate on [ N x D ] N = Number of Samples, D = Number of Features
        outputs (pd.DataFrame): ground-truth labels [ N x 1 ] 
        eval_type (string): determines how to perform evaluation
        task (string): task of interest e.g., binary classification, etc. 
    
    Returns:
        acc, auc, prec, rec (floats): performance metrics
        probs_df (pd.DataFrame): probability values for the output prediction
                                 Rows: contain the probability values for each instance
                                 Columns: probability-instance
    """
    preds = model.predict(inputs)
    probs = model.predict_proba(inputs)

    acc = accuracy_score(outputs,preds)     
    auc = roc_auc_score(outputs,probs[:,1])
    prec = precision_score(outputs,preds)
    rec = recall_score(outputs,preds)
    
    return acc, auc, prec, rec

#%%

def run_variants(settings,setting='in-domain',label_name='Combined',suffix='',lab_collapse_strategy='max',doc_collapse_strategy='concat',labelled_and_unlabelled='labelled_only',goal='train and evaluate model',phases=['train','val'],dataset_type='All',max_inference_seed=1,batch_size=32,lr=1e-4,verbose=0,save_model_flag=False,switch=False):
    """ Conduct classification experiments on the pan-tumour dataset. 
        This iterates over all configuration settings (which can be turned on and off)
        
    Args:
        settings (dict): dictionary containing the settings to experiment with
        label_name (string): column name associated with labels
        lab_collapse_strategy (string): strategy on how to aggregate lab information across dates
        labelled_and_unlabelled (string): determines whether to deal with labelled or unlabelled data
        goal (string): task of interest (e.g., train and evaluate, perform inference, etc.)
        phases (list): outline the experimental phases (e.g., train, val, test)
    
    Returns:
        results_df (pd.DataFrame): matrix of results for the different settings
                                   Rows: results for each experimental setting
                                   Columns: AUC, ACC, Precision, Recall
        probs_df (pd.DataFrame): matrix of output probabilities (note: the exact nature of this depends on goal and other factors)
                                 Rows: probability output by network for each instance
                                 Columns: probability-instance
    """
    if goal in ['train and evaluate model','evaluate model','validate tgt data']:
        if goal == 'train and evaluate model':
            cols = ['Model','Train Acc','Train AUC','Train Precision','Train Recall','Val ACC','Val AUC','Val Precision','Val Recall']
        elif goal == 'evaluate model':
            cols = ['Model','Val ACC','Val AUC','Val Precision','Val Recall']
        elif goal == 'validate tgt data':
            cols = ['Wild Data Setting','Validation Label Noise','Model','Threshold Side','Threshold','Seed','Val ACC','Val AUC','Val Precision','Val Recall']#,'Val BCE']        
        results_df = pd.DataFrame(columns=cols)

    ncombos = np.prod(list(map(lambda ls:len(ls),settings.values())))
    count = 1
    task = task_list[0]
    
    """ Source data to sample from for SUDO experiments """
    ntraining_samples = 500 # training set
    mean1, mean2 = [1,1], [2,2]
    cov1, cov2 = [[0.8,0],[0,0.8]], [[0.8,0],[0,0.8]]
    data_class1 = pd.DataFrame(sample_gaussian_points(mean=mean1,cov=cov1,nsamples=ntraining_samples))
    data_class1['Label'] = 0
    data_class2 = pd.DataFrame(sample_gaussian_points(mean=mean2,cov=cov2,nsamples=ntraining_samples))
    data_class2['Label'] = 1
    data = {'train':pd.concat((data_class1,data_class2),0)}
    
    nval_samples = 200 # validation set
    mean1, mean2 = [1,1], [2,2]
    data_class1 = pd.DataFrame(sample_gaussian_points(mean=mean1,nsamples=nval_samples))
    data_class1['Label'] = 0
    data_class2 = pd.DataFrame(sample_gaussian_points(mean=mean2,nsamples=nval_samples))
    data_class2['Label'] = 1
    data['val'] = pd.concat((data_class1,data_class2),0)
    
    model_type = 'LR'
    print('*** Source Model ***')
    src_model = load_model(model_type,goal=goal,verbose=verbose)
    train_model(src_model, data['train'][[0,1]], data['train']['Label'], model_type)
    
    for validation_label_noise in settings['validation label noise list']:
        
        if validation_label_noise != 0:
            ncorrupt = int(validation_label_noise * nval_samples)
            random.seed(0)
            indices_to_corrupt = random.sample(list(range(nval_samples)),ncorrupt)
            data['val']['Label'] = data['val'].reset_index().apply(lambda row: 1 - row['Label'] if row['index'] in indices_to_corrupt else row['Label'],axis=1) 
        
        for wild_data_setting in settings['wild data setting list']:
            wild_data = get_wild_data(wild_data_setting) # getting same wild data --> good sign
                          
            """ Retrieve Desired Subset of Unlablled Data """
            probs = src_model.predict_proba(wild_data[[0,1]])[:,1] # first column is probability of positive class
            probs_df = pd.DataFrame(probs,columns=['Prob'])
            probs_df['Label'] = wild_data['Label'].reset_index(drop=True)
            if not 'SIM_Probs_%s.csv' % wild_data_setting in os.listdir(): 
                probs_df.to_csv('SIM_Probs_%s.csv' % wild_data_setting)
            prob_col = 'Prob' #'Prob' = default | 'Random Prob' to investigate random stuff
        
            for nfeatures in settings['nfeatures list']:
                
                """ Loops for Validating Unlabelled Data Pseudo Labels """
                for inference_seed in range(max_inference_seed):
                    
                    for inference_threshold_side in settings['inference threshold side list']: #looking at low pseudo or high pseudo labels 
                        if inference_threshold_side == 'low':
                            prev_threshold = 0
                        elif inference_threshold_side == 'high':
                            prev_threshold = 0.9 # 0.9 for the shift-pos experiments
        
                        for inference_threshold in settings['inference threshold list']: #threshold for performing pseudo vs. real classification 
                            #print(inference_threshold_side_list,switch,inference_seed,prev_threshold,inference_threshold)
                            dict_df = dict()
                            dict_df['train'] = pd.DataFrame(data['train']).sample(data['train'].shape[0],replace=False,random_state=0)
                            dict_df['val'] = data['val'][[0,1]]
                            labels_df = dict()
                            labels_df['train'] = label_encoder.fit_transform(pd.DataFrame(data['train']['Label']).sample(len(data['train']),replace=False,random_state=0))
                            labels_df['val'] = label_encoder.transform(pd.DataFrame(data['val']['Label']))
                        
                        # for wild_data_setting in settings['wild data setting list']:
                        #     wild_data = get_wild_data(wild_data_setting) # getting same wild data --> good sign
                            
                            if goal in ['validate tgt data']:
                                
                                if inference_threshold_side == 'low':
                                    if switch == False:
                                        pseudo_label = 0
                                        real_label = 1
                                    elif switch == True:
                                        pseudo_label = 1
                                        real_label = 0
                                    
                                    nsamples = 10 # Number of data points sampled from each probability interval
                                    bool1 = probs_df[prob_col] > prev_threshold
                                    bool2 = probs_df[prob_col] <= inference_threshold
                                    boolComb = bool1 & bool2 
                                    prev_threshold = inference_threshold
                                    """ Retrieve Unlabelled Data """
                                    subset_probs_df = probs_df[boolComb]
                                    indices = subset_probs_df.index.tolist()
                                    tgt_data = wild_data.iloc[indices,:]
                                    tgt_data_subset = tgt_data.sample(n=nsamples,random_state=inference_seed)
                                    tgt_data_subset = tgt_data_subset[[0,1]]
                                    tgt_labels_subset = pd.Series([pseudo_label]*len(tgt_data_subset)) #tgt_labels[boolComb]                            
                                    """ Retrieve Matched Labelled Data from Different CLass """
                                    src_data = dict_df['train'][labels_df['train']==real_label]
                                    src_data_subset = src_data.sample(n=nsamples,random_state=inference_seed) 
                                    src_data_subset = src_data_subset[[0,1]]
                                    src_labels_subset = pd.Series([real_label]*len(src_data_subset))
                                elif inference_threshold_side == 'high':
                                    if switch == False:
                                        pseudo_label = 1
                                        real_label = 0
                                    elif switch == True:
                                        pseudo_label = 0
                                        real_label = 1

                                    nsamples = 10 # Number of data points sampled from each probability interval
                                    bool1 = probs_df[prob_col] < prev_threshold
                                    bool2 = probs_df[prob_col] >= inference_threshold
                                    boolComb = bool1 & bool2 
                                    prev_threshold = inference_threshold
                                    """ Retrieve Unlabelled Data """
                                    subset_probs_df = probs_df[boolComb]
                                    indices = subset_probs_df.index.tolist()
                                    tgt_data = wild_data.iloc[indices,:]
                                    tgt_data_subset = tgt_data.sample(n=nsamples,random_state=inference_seed)
                                    tgt_data_subset = tgt_data_subset[[0,1]]
                                    tgt_labels_subset = pd.Series([pseudo_label]*len(tgt_data_subset)) #tgt_labels[boolComb]                            
                                    """ Retrieve Matched Labelled Data from Different CLass """
                                    src_data = dict_df['train'][labels_df['train']==real_label]
                                    src_data_subset = src_data.sample(n=nsamples,random_state=inference_seed) 
                                    src_data_subset = src_data_subset[[0,1]]
                                    src_labels_subset = pd.Series([real_label]*len(src_data_subset))
                                """ Combine Pseudo Unlabelled Data and Real Labelled Data """
                                data_subset = pd.concat((src_data_subset,tgt_data_subset),0).sample(n=nsamples*2,replace=False,random_state=0)
                                labels_subset = pd.concat((src_labels_subset,tgt_labels_subset),0).sample(n=nsamples*2,replace=False,random_state=0)
                                dict_df['train'] = data_subset
                                labels_df['train'] = labels_subset
                            
                            train_inputs2, val_inputs2 = pd.DataFrame(dict_df['train']), pd.DataFrame(dict_df['val'])
                            train_outputs2, val_outputs2 = labels_df['train'], labels_df['val']
                            
                            # only used for test set evaluation experiment 
                            #val_inputs2, val_outputs2 = pd.DataFrame(dict_df['test']), labels_df['test']
                            
                            """ Scale Inputs """
                            for scale in settings['scale list']:
                                train_inputs3, train_outputs3 = train_inputs2.copy(), train_outputs2.copy()
                                val_inputs3, val_outputs3 = val_inputs2.copy(), val_outputs2.copy()

                                """ Load Model, Train, and Evaluate """
                                for model_type in settings['model list']:
                                    print('Starting Setting %i/%i' % (count,ncombos))
                                            
                                    if model_type in ['LR','RF','XGB','SVM']:
                                        if goal in ['train and evaluate model','validate tgt data']:
                                            model = load_model(model_type,goal=goal,verbose=verbose)
                                            train_model(model, train_inputs3, train_outputs3, model_type)
                                            
                                            """ Needs to be Changed """
                                            train_metrics = evaluate_model(model, train_inputs3, train_outputs3, dict_df['train'])
                                            val_metrics = evaluate_model(model, val_inputs3, val_outputs3, dict_df['val'])
                                            
                                            train_acc,train_auc,train_prec,train_rec = train_metrics
                                            val_acc,val_auc,val_prec,val_rec = val_metrics

                                    elif model_type in ['NN']:
                                        inputs_and_outputs = defaultdict()
                                        if goal in ['train and evaluate model','validate tgt data']:
                                            inputs_and_outputs['train'] = defaultdict()
                                            inputs_and_outputs['train']['inputs'] = train_inputs3
                                            inputs_and_outputs['train']['outputs'] = pd.Series(train_outputs3)
                                            inputs_and_outputs['val'] = defaultdict()
                                            inputs_and_outputs['val']['inputs'] = val_inputs3
                                            inputs_and_outputs['val']['outputs'] = pd.Series(val_outputs3)
                                            max_epochs = 20
                                        elif goal in ['evaluate model']:
                                            inputs_and_outputs['val'] = defaultdict()
                                            inputs_and_outputs['val']['inputs'] = val_inputs3
                                            inputs_and_outputs['val']['outputs'] = pd.Series(val_outputs3)
                                            phases = ['val']
                                            suffix = '_OR' #options are : '_OR' | '_OT' 
                                            max_epochs = 1
                                        elif goal in ['perform inference']:
                                            inputs_and_outputs['inference'] = defaultdict()
                                            inputs_and_outputs['inference']['inputs'] = val_inputs3
                                            inputs_and_outputs['inference']['outputs'] = pd.Series(val_outputs3)  
                                            phases = ['inference']
                                            max_epochs = 1
                                            
                                        savepath = os.path.join(os.getcwd(),'results')
                                        writer = SummaryWriter(savepath)
                                        metrics_dict, reps_dict, probs_dict = train(inputs_and_outputs,dict_df,phases=phases,batch_size=batch_size,suffix=suffix,lr=lr,task=task,goal=goal,writer=writer,max_epochs=max_epochs)
                                        
                                        train_acc, train_auc = 0,0 #metrics_dict['acc']['train'], metrics_dict['auc']['train']
                                        val_acc, val_auc = 0,0 #metrics_dict['acc']['val'], metrics_dict['auc']['val']
                                        train_prec, train_rec = 0,0
                                        val_prec, val_rec = 0,0
                                    
                                    """ Store Results """
                                    if goal in ['train and evaluate model','evaluate model','validate tgt data']:
                                        if goal == 'train and evaluate model':
                                            curr_df = pd.DataFrame([model_type,train_acc,train_auc,train_prec,train_rec,val_acc,val_auc,val_prec,val_rec]).T
                                        elif goal == 'evaluate model':
                                            curr_df = pd.DataFrame([model_type,val_acc,val_auc,val_prec,val_rec]).T
                                        elif goal == 'validate tgt data':
                                            curr_df = pd.DataFrame([wild_data_setting,validation_label_noise,model_type,inference_threshold_side,inference_threshold,inference_seed,val_acc,val_auc,val_prec,val_rec]).T  
                                        curr_df.columns = cols
                                        results_df = pd.concat((results_df,curr_df),0)

                                    count += 1
    
    if goal in ['train and evaluate model','evaluate model','validate tgt data']:
        return results_df, None #val_probs_df
        
    elif goal in ['perform inference']:
        return None, probs_df

#%%
def get_settings(inference_threshold_side_list,inference_threshold_list,wild_data_setting_list):
    settings = {'weak supervision list':weak_supervision_list,
                'task list':task_list,
                'labs list':labs_list,
                'documents list':documents_list,
                'demo list':demo_list,
                'diagnosis list':diagnosis_list,
                'vitals list':vitals_list,
                'medications list':medications_list,
                'feature list':features_list,
                'ngram list':ngram_list,
                'nfeatures list':nfeatures_list,            
                'imputation type list':imputation_type_list,
                'sampling strategy list':sampling_strategy_list,
                'scale list':scale_list,
                'model list':model_list,
                'inference threshold side list':inference_threshold_side_list,
                'inference threshold list':inference_threshold_list,   
                'wild data setting list':wild_data_setting_list,    
                'validation label noise list':validation_label_noise_list,
                }
    return settings    

#%%
if __name__ == '__main__':
    
    setting = 'in-domain' #options: 'in-domain' | 'out-of-domain'
    weak_supervision_list = [False] 
    task_list = ['01 vs. 234'] #['0'] #
    labs_list = [False]
    documents_list = [True]
    demo_list = [False]
    diagnosis_list = [False]
    vitals_list = [False]
    medications_list = [False] 
    features_list = ['BoW']
    ngram_list = [2]
    nfeatures_list = [5000]
    imputation_type_list = ['median']
    sampling_strategy_list = [False]
    scale_list = [False]
    model_list = ['LR'] #NN #RF
    batch_size = 32 #applicable only with NN
    lr = 1e-4 #applicable only with NN
    lab_collapse_strategy = 'max' #default is 'max' options: max | min | max/min
    doc_collapse_strategy = 'concat' #options are: concat | latest
    verbose = 0 #options: | 0 | 1 | 2
    labelled_and_unlabelled = 'test_labelled_only' #'labelled_and_unlabelled' #options: labelled_only | unlabelled_only | labelled_and_unlabelled | test_labelled_only
    goal = 'validate tgt data' #'validate tgt data' #options: 'train and evaluate model' | evaluate model | calibrate model | perform inference | validate_unlabelled_data
    dataset_type = 'All'
    inference_threshold_side_list = ['low'] #options: low | high
    inference_threshold_list = [1] #np.arange(0.1,0.6,0.1) #[0] #[0.9, 0.8, 0.7, 0.6, 0.5] #np.arange(0.1,0.6,0.1) # np.arange(0.80,0.40,-0.2) ## #options: values in [0,1]
    wild_data_setting_list = ['out-of-domain-shift-pos','out-of-domain-shift-pos+','out-of-domain-shift-pos++'] #['out-of-domain','out-of-domain-third-class','out-of-domain-imbalance']#,'out-of-domain-one-class']
    validation_label_noise_list = [0] #[0,0.1,0.2,0.3,0.4,0.5]
    max_inference_seed = 5 #5 #number of times to perform inference (primarily used for validating unlabelled data)
    suffix = '' #options are: '' | '_Masked' (underscore is important) - only used when loading models for inference
    save_model_flag = False #save ml model, scaler, etc. 
    
    settings = {'weak supervision list':weak_supervision_list,
                'task list':task_list,
                'labs list':labs_list,
                'documents list':documents_list,
                'demo list':demo_list,
                'diagnosis list':diagnosis_list,
                'vitals list':vitals_list,
                'medications list':medications_list,
                'feature list':features_list,
                'ngram list':ngram_list,
                'nfeatures list':nfeatures_list,            
                'imputation type list':imputation_type_list,
                'sampling strategy list':sampling_strategy_list,
                'scale list':scale_list,
                'model list':model_list,
                'inference threshold side list':inference_threshold_side_list,
                'inference threshold list':inference_threshold_list,   
                'wild data setting list':wild_data_setting_list,   
                'validation label noise list':validation_label_noise_list,
                }
    
    if goal == 'validate tgt data':
        df = pd.DataFrame()
        for inference_threshold_side_list,inference_threshold_list in zip([['low'],['high']],[np.arange(0.1,0.6,0.1),[0.8,0.7,0.6,0.5]]): # ['high'] [0.9, 0.8, 0.7, 0.6, 0.5] | for EXTRA experiments -> [0.8, 0.7, 0.6, 0.5] (change prev_threshold in func)
            for switch in [False,True]:
                settings = get_settings(inference_threshold_side_list,inference_threshold_list,wild_data_setting_list)
                results_df, probs_df = run_variants(settings,setting=setting,label_name='Combined',suffix=suffix,lab_collapse_strategy=lab_collapse_strategy,
                                             doc_collapse_strategy=doc_collapse_strategy,labelled_and_unlabelled=labelled_and_unlabelled,goal=goal,batch_size=batch_size,
                                             dataset_type=dataset_type,max_inference_seed=max_inference_seed,lr=lr,verbose=verbose,save_model_flag=save_model_flag,switch=switch
                                             )
                
                if inference_threshold_side_list == ['low']:
                    if switch == False:
                        env = 'Pseudo Low Label'
                    else:
                        env = 'Pseudo High Label'
                elif inference_threshold_side_list == ['high']:
                    results_df['Threshold'] = results_df['Threshold'] + 0.1
                    if switch == False:
                        env = 'Pseudo High Label'
                    else:
                        env = 'Pseudo Low Label'
                results_df['Setup'] = env
                df = pd.concat((df,results_df),0)
        
    else:
        results_df, probs_df = run_variants(settings,setting=setting,label_name='Combined',suffix=suffix,lab_collapse_strategy=lab_collapse_strategy,
                                     doc_collapse_strategy=doc_collapse_strategy,labelled_and_unlabelled=labelled_and_unlabelled,goal=goal,batch_size=batch_size,
                                     dataset_type=dataset_type,max_inference_seed=max_inference_seed,lr=lr,verbose=verbose,save_model_flag=save_model_flag)
    
    
    
#%%
visualize_stuff = False
#if visualize_stuff == True:
sns.set(font_scale=2)
sns.set_style('ticks')

def get_source_data():
    ntraining_samples = 500 # training set
    mean1, mean2 = [1,1], [2,2]
    cov1, cov2 = [[0.8,0],[0,0.8]], [[0.8,0],[0,0.8]]
    data_class1 = pd.DataFrame(sample_gaussian_points(mean=mean1,cov=cov1,nsamples=ntraining_samples))
    data_class1['Label'] = 0
    data_class2 = pd.DataFrame(sample_gaussian_points(mean=mean2,cov=cov2,nsamples=ntraining_samples))
    data_class2['Label'] = 1
    data = {'train':pd.concat((data_class1,data_class2),0)}
    
    nval_samples = 200 # validation set
    mean1, mean2 = [1,1], [2,2]
    data_class1 = pd.DataFrame(sample_gaussian_points(mean=mean1,nsamples=nval_samples))
    data_class1['Label'] = 0
    data_class2 = pd.DataFrame(sample_gaussian_points(mean=mean2,nsamples=nval_samples))
    data_class2['Label'] = 1
    data['val'] = pd.concat((data_class1,data_class2),0)
    return data

def generate_data_plot(wild_data_setting='out-of-domain'):
    src_data = get_source_data()
    wild_data = get_wild_data(wild_data_setting) # getting same wild data --> good sign
    fig,axes = plt.subplots(2,1,figsize=(8,8),gridspec_kw={'height_ratios':(0.08,0.90)})
    axes[0].remove()
    ax = axes[1]
    cols = ['x','y','Label']
    src_data['train'].columns = cols
    src_data['val'].columns = cols
    wild_data.columns = cols
    palette = {0: 'royalblue', 1: 'forestgreen'}
    wild_palette = {0: 'navy', 1: 'darkgreen', 2: 'darkorange'}
    sns.scatterplot(x='x',y='y',hue='Label',data=src_data['train'],alpha=1,ci=None,s=50,ec='k',ax=ax,palette=palette,legend='brief')
    sns.scatterplot(x='x',y='y',hue='Label',data=src_data['val'],alpha=0.3,ci=None,s=50,ec='k',ax=ax,palette=palette,legend=False)
    sns.scatterplot(x='x',y='y',hue='Label',data=wild_data,alpha=1,ci=None,s=50,ec='k',ax=ax,palette=wild_palette,legend='brief',zorder=0)
    cols = [r'$x_{1}$',r'$x_{2}$','Label']
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    handles,labels = ax.get_legend_handles_labels()
    labels = ['0 (src)', '1 (src)', '0 (wild)', '1 (wild)']
    ncols = 2
    if wild_data_setting == 'out-of-domain-third-class':
        labels += ['2 (wild)']
        ncols = 3
    ax.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,1.35),frameon=True,ncol=ncols,facecolor='white',title='class and data source')
    return fig

def get_sudo_info(metric='Val AUC',wild_data_setting='out-of-domain',validation_label_noise=0):
    df = pd.read_csv('Validating_Unlabelled_Data_Thresholds.csv',index_col=0)
    bool1 = df['Wild Data Setting'] == wild_data_setting
    bool2 = df['Validation Label Noise'] == validation_label_noise
    boolComb = bool1 & bool2
    df = df[boolComb]
    low_label, high_label = 'Pseudo Negative', 'Pseudo Positive'
    df['Setup'] = df['Setup'].replace({'Pseudo Low Label':low_label,'Pseudo High Label':high_label})    
    print(metric)
    g = df.groupby(by=['Threshold']).apply(lambda row:row[row['Setup'] == low_label].mean() - row[row['Setup'] == high_label].mean())
    g['Threshold'] = g.index
    #g['Val AUC Abs'] = g[metric].abs()
    g['Color'] = g[metric].apply(lambda auc: 'royalblue' if auc > 0 else 'forestgreen')
    return df, g
    
def generate_violin_plot(metric='Val AUC',wild_data_setting='out-of-domain',validation_label_noise=0):
    df, g = get_sudo_info(metric,wild_data_setting,validation_label_noise)
    """ Violin Plot """
    low_label, high_label = 'Pseudo Negative', 'Pseudo Positive'
    palette = {low_label: 'royalblue', high_label: 'forestgreen'}
    fig,axes = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':(0.05,0.95)})
    axes[0].remove()
    ax = axes[1]
    sns.violinplot(x='Threshold',y=metric,hue='Setup',data=df,ax=ax,palette=palette)
    ax.set_xticklabels(list(map(lambda el:'%.2f' % el,sorted(df['Threshold'].unique().tolist()))))
    ax.set_xlabel('Threshold')
    legend,handles = ax.get_legend_handles_labels()
    ax.legend(legend,handles,loc='upper center',bbox_to_anchor=(0.5,1.2),frameon=True,ncol=2,facecolor='white')
    return fig
    
def generate_sudo_plot(metric='Val AUC',wild_data_setting='out-of-domain',validation_label_noise=0):
    df, g = get_sudo_info(metric,wild_data_setting,validation_label_noise)
    fig,axes = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':(0.05,0.95)})
    axes[0].remove()
    ax = axes[1]
    sns.barplot(x=np.arange(len(g)),y=metric,data=g,ec='black',ax=ax,palette=g['Color'])
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Pseudo Label Discrepancy')
    ax.set_xticklabels(list(map(lambda el:'%.2f' % el,sorted(g['Threshold'].unique().tolist()))))
    ax.set_ylim([-1.1,1.1])
    return fig
    
def get_correlation(metric='Val AUC',wild_data_setting='out-of-domain',validation_label_noise=0):
    """ Correlation Plot """
    df, g = get_sudo_info(metric,wild_data_setting,validation_label_noise)
    probs_df = pd.read_csv('SIM_Probs_%s.csv' % wild_data_setting,index_col=0)
    
    if 'shift-pos' in wild_data_setting:# in ['out-of-domain-on-boundary','out-of-domain-flipped']:
        bins, labels = np.arange(0,1,0.1), np.arange(0.1,1,0.1)
    else:
        bins, labels = np.arange(0,1.1,0.1), np.arange(0.1,1.1,0.1)
    
    prob_col = 'Prob' #option: Calib Prob
    probs_df['Range'] = pd.cut(probs_df[prob_col],bins=bins,labels=labels)
    counts = probs_df['Range'].value_counts().tolist()
    if wild_data_setting == 'out-of-domain-third-class':
        acc = probs_df.groupby(by=['Range'])['Label'].apply(lambda labels:np.histogram(labels,[0,1,2,3])[0][1]/sum(np.histogram(labels,[0,1,2,3])[0]))
    else:        
        acc = probs_df.groupby(by=['Range'])['Label'].mean().reset_index(drop=True)
    acc.index = g.index
    g['Proportion High'] = acc #1-acc
    
    g['Count Weight'] = np.array(counts)/sum(counts)
    g['Bin Acc'] = acc
    bin_probs = probs_df.groupby(by=['Range'])[prob_col].mean()
    bin_probs.index = g.index
    g['Bin Prob'] = bin_probs
    
    corr = g[[metric,'Proportion High']].corr()
    corr = corr[metric]['Proportion High']
    return df, g, corr

def generate_correlation_plot(metric='Val AUC',wild_data_setting='out-of-domain',validation_label_noise=0):
    df, g, corr = get_correlation(metric,wild_data_setting,validation_label_noise)
    m = LinearRegression()
    m.fit(pd.DataFrame(g['Val AUC']),pd.DataFrame(g['Proportion High']))
    
    line = m.predict(pd.DataFrame(np.arange(g['Val AUC'].min()-0.05,g['Val AUC'].max()+0.05,0.05)))
    fig,axes = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':(0.05,0.95)})
    axes[0].remove()
    ax = axes[1]
    sns.scatterplot(g['Val AUC'],g['Proportion High'],ci=None,s=200,ec='k',color='goldenrod',ax=ax,label=r'$\rho$ = %.2f' % corr)
    sns.lineplot(np.arange(g['Val AUC'].min()-0.05,g['Val AUC'].max()+0.05,0.05),line.squeeze(),lw=2,color='royalblue',ax=ax)#,label=r'$\rho$=0.95')
    plt.xlabel(r'$\mathrm{SUDO}_{\mathrm{%s}}$' % metric.split(' ')[1])
    plt.ylabel('Proportion of positive instances')
    #plt.ylim([0,1.1])
    return fig
    
def generate_histogram_plot(wild_data_setting='out-of-domain'):
    """ Histogram Plot """
    probs_df = pd.read_csv('SIM_Probs_%s.csv' % wild_data_setting,index_col=0)
    fig,axes = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':(0.05,0.95)})
    axes[0].remove()
    ax = axes[1]
    probs_df['Label'] = probs_df['Label'].replace({0:'Negative',1:'Positive'})
    palette = {'Negative': 'royalblue', 'Positive': 'forestgreen', 2: 'darkorange'}
    sns.histplot(x='Prob',hue='Label',data=probs_df,palette=palette,bins=50,ax=ax,legend='brief')
    legend = ax.get_legend()
    handles = legend.legendHandles
    _,labels = ax.get_legend_handles_labels()
    if wild_data_setting == 'out-of-domain-third-class':
        labels = [0,1,2]
        ncols = 3
    else:
        labels = [0,1]
        ncols = 2
    
    ax.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,1.2),frameon=True,ncol=ncols,facecolor='white')
    ax.set_xlabel('AI-based probability')
    ax.set_ylabel('Number of instances')
    ax.set_xlim([-0.05,1.05])
    return fig

def generate_correlation_vs_rotation_plot(metric,wild_data_settings):
    corr_df = pd.DataFrame()
    mapping = {'out-of-domain':0,
               'out-of-domain-shift-pos':np.sqrt(2),
               'out-of-domain-shift-pos+':2*np.sqrt(2),
               'out-of-domain-shift-pos++':3*np.sqrt(2)
               }
    for wild_data_setting in wild_data_settings:
        df, g, corr = get_correlation(metric,wild_data_setting,validation_label_noise)
        corr = 0 if np.isnan(corr) else corr
        curr_corr_df = pd.DataFrame([mapping[wild_data_setting],corr]).T
        corr_df = pd.concat((corr_df,curr_corr_df),0)
    corr_df.columns = ['Change in distribution mean','Correlation']
    
    fig,axes = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':(0.05,0.95)})
    axes[0].remove()
    ax = axes[1]
    sns.scatterplot(x='Change in distribution mean',y='Correlation',data=corr_df,marker='v',ax=ax,legend='brief',ci=None,s=200,ec='k',color='purple')
    ax.axhline(0,lw=1.5,ls='--',zorder=0,color='k')
    #ax.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,1.2),frameon=True,ncol=ncols,facecolor='white')
    ax.set_xlabel(r'$\Delta$ location of positive distribution in the wild')
    ax.set_ylabel(r'$\rho$ between $\mathrm{SUDO}_{\mathrm{%s}}$ and accuracy' % metric.split(' ')[1])
    
    ax.set_xticks(list(mapping.values()))
    ax.set_xticklabels(['0',r'$\sqrt{2}$',r'2$\sqrt{2}$',r'3$\sqrt{2}$'])
    
    return fig


#%%
metric = 'Val AUC' # options: Val AUC | Val Precision 
corr_df = pd.DataFrame()
for wild_data_setting in ['out-of-domain','out-of-domain-third-class','out-of-domain-imbalance']:
    for validation_label_noise in [0,0.1,0.2,0.3,0.4,0.5]:
        _, _, corr = get_correlation(metric,wild_data_setting,validation_label_noise)
        cur_corr_df = pd.DataFrame([metric,wild_data_setting,validation_label_noise,corr]).T
        cur_corr_df.columns = ['Metric','Wild Data Setting','Validation Label Noise','Corr']
        corr_df = pd.concat((corr_df,cur_corr_df),0)

#%%
sns.set(font_scale=2)
sns.set_style('ticks')
plot_type = 'data'
metric = 'Val AUC'
save_images = True
save_path = path_to_save_folder

if plot_type == 'correlation_vs_rotation':
    wild_data_settings = ['out-of-domain','out-of-domain-shift-pos','out-of-domain-shift-pos+','out-of-domain-shift-pos++'] 
    fig = generate_correlation_vs_rotation_plot(metric,wild_data_settings)
    
    if save_images == True:
            save_name = 'sim_%s.png' % (plot_type)
            fig.savefig(os.path.join(save_path,save_name))
    
else:
    for wild_data_setting in ['out-of-domain-shift-pos','out-of-domain-shift-pos++']: #['out-of-domain','out-of-domain-third-class','out-of-domain-imbalance']:
        if plot_type == 'data':
            fig = generate_data_plot(wild_data_setting)
        elif plot_type == 'histogram':
            fig = generate_histogram_plot(wild_data_setting)
        elif plot_type == 'correlation':
            fig = generate_correlation_plot(wild_data_setting=wild_data_setting)
       
        if save_images == True:
            save_name = 'sim_%s_%s.png' % (plot_type,wild_data_setting)
            fig.savefig(os.path.join(save_path,save_name))

