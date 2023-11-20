#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 18:17:47 2021

@author: dani.kiyasseh

This script is used to:
1) conduct the SUDO experiments on the Multi-Domain Sentiment dataset
"""

# [USER MUST MODIFY] 
path_to_amazon_folder = '/Users/dani.kiyasseh/Desktop/Data/processed_acl'
path_to_current_scripts = '/Users/dani.kiyasseh/Desktop/Scripts'

import os
new_path = path_to_current_scripts
import sys
import pickle
import random
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

from prepare_miscellaneous import load_model, train_model, load_scaler, scale_inputs
from network import train, train_for_OOD
from torch.utils.tensorboard import SummaryWriter

from load_amazon_data import load_amazon_reviews, \
                            split_reviews_into_phases, \
                            retrieve_vocabs, \
                            get_vectorizers, \
                            get_data

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
path = path_to_amazon_folder
os.chdir(path)
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
            cols = ['Model','Threshold Side','Threshold','Seed','Val ACC','Val AUC','Val Precision','Val Recall']#,'Val BCE']        
        results_df = pd.DataFrame(columns=cols)

    ncombos = np.prod(list(map(lambda ls:len(ls),settings.values())))
    count = 1
    task = task_list[0]
    
    reviews = load_amazon_reviews()
    phase_reviews, labels = split_reviews_into_phases(reviews)
    
    for nfeatures in settings['nfeatures list']:
        domain_vocabs = retrieve_vocabs(phase_reviews,nfeatures)
        
        """ Loops for Validating Unlabelled Data Pseudo Labels """
        for inference_seed in range(max_inference_seed):
    
            for inference_threshold_side in settings['inference threshold side list']: #looking at low pseudo or high pseudo labels 
                if inference_threshold_side == 'low':
                    prev_threshold = 0
                elif inference_threshold_side == 'high':
                    prev_threshold = 1
            
                for inference_threshold in settings['inference threshold list']: #threshold for performing pseudo vs. real classification 
                    src = 'books'
                    tgt = 'electronics'
                    savepath = os.path.join(path,'results',src)
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    os.chdir(savepath)
                                    
                    vecs_dict = get_vectorizers(domain_vocabs)
                    data = get_data(phase_reviews,vecs_dict[src],src)
                    
                    dict_df = dict()
                    dict_df['train'] = pd.DataFrame(data[src]['train']).sample(data[src]['train'].shape[0],replace=False,random_state=0)
                    dict_df['val'] = pd.DataFrame(data[src]['val'])
                    dict_df['test'] = pd.DataFrame(data[src]['test']) 
                    labels_df = dict()
                    labels_df['train'] = label_encoder.fit_transform(pd.DataFrame(labels[src]['train']).sample(len(labels[src]['train']),replace=False,random_state=0))
                    labels_df['val'] = label_encoder.transform(pd.DataFrame(labels[src]['val']))
                    labels_df['test'] = label_encoder.transform(pd.DataFrame(labels[src]['test']))

                    if setting == 'out-of-domain':
                        vecs_dict = get_vectorizers(domain_vocabs)
                        data = get_data(phase_reviews,vecs_dict[src],tgt)
                        
                        if goal in ['perform inference']:
                            dict_df = dict()
                            dict_df['val'] = pd.DataFrame(data[tgt]['train'])
                            dict_df['train'] = pd.DataFrame(data[tgt]['val']) # do not use
                            
                            labels_df = dict()
                            labels_df['val'] = label_encoder.fit_transform(pd.DataFrame(labels[tgt]['train']))
                            labels_df['train'] = label_encoder.transform(pd.DataFrame(labels[tgt]['val'])) # do not use
                        else:
                            """ Retrieve Desired Subset of Unlablled Data """
                            probs_df = pd.read_csv('%s_to_%s_Probs_NN5000_OT.csv' % (src,tgt),index_col=0)
                            #probs_df = pd.read_csv('%s_to_%s_Probs_NN5000_OR.csv' % (src,src),index_col=0)
                            probs_df['Random Prob'] = [random.uniform(0,1) for _ in range(probs_df.shape[0])]
                            tgt_data = pd.DataFrame(data[tgt]['train']) #np.concatenate((data[tgt]['train'],data[tgt]['val'],data[tgt]['test']),0))

                            prob_col = 'Prob' #'Prob' = default | 'Random Prob' to investigate random stuff
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
                                tgt_data_subset = tgt_data[boolComb]
                                tgt_data_subset = tgt_data_subset.sample(n=nsamples,random_state=inference_seed)
                                tgt_labels_subset = pd.Series([pseudo_label]*len(tgt_data_subset)) #tgt_labels[boolComb]
                                """ Retrieve Matched Labelled Data from Different CLass """
                                src_data_subset = dict_df['train'][labels_df['train']==real_label]
                                src_data_subset = src_data_subset.sample(n=nsamples,random_state=0)
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
                                tgt_data_subset = tgt_data[boolComb]
                                tgt_data_subset = tgt_data_subset.sample(n=nsamples,random_state=inference_seed)
                                tgt_labels_subset = pd.DataFrame([pseudo_label]*len(tgt_data_subset)) #tgt_labels[boolComb]
                                """ Retrieve Matched Labelled Data from Different CLass """
                                src_data_subset = dict_df['train'][labels_df['train']==real_label]
                                src_data_subset = src_data_subset.sample(n=nsamples,random_state=0)
                                src_labels_subset = pd.DataFrame([real_label]*len(src_data_subset))
                            """ Combine Pseudo Unlabelled Data and Real Labelled Data """
                            data_subset = pd.concat((src_data_subset,tgt_data_subset),0).sample(n=nsamples*2,replace=False,random_state=0)
                            labels_subset = pd.concat((src_labels_subset,tgt_labels_subset),0).sample(n=nsamples*2,replace=False,random_state=0)
                            dict_df['train'] = data_subset
                            labels_df['train'] = labels_subset
                    elif setting == 'in-domain':
                        vecs_dict = get_vectorizers(domain_vocabs)
                        data = get_data(phase_reviews,vecs_dict[src],src)

                        if goal in ['validate tgt data']:
                            """ Retrieve Desired Subset of Unlablled Data """
                            tgt = src
                            probs_df = pd.read_csv('%s_to_%s_Test_Probs_NN5000_OT.csv' % (src,tgt),index_col=0)
                            probs_df['Random Prob'] = [random.uniform(0,1) for _ in range(probs_df.shape[0])]
                            tgt_data = pd.DataFrame(data[tgt]['test']) #np.concatenate((data[tgt]['train'],data[tgt]['val'],data[tgt]['test']),0))

                            prob_col = 'Prob' #'Prob' = default | 'Random Prob' to investigate random stuff
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
                                tgt_data_subset = tgt_data[boolComb]
                                tgt_data_subset = tgt_data_subset.sample(n=nsamples,random_state=inference_seed)
                                tgt_labels_subset = pd.Series([pseudo_label]*len(tgt_data_subset)) #tgt_labels[boolComb]
                                """ Retrieve Matched Labelled Data from Different CLass """
                                src_data_subset = dict_df['train'][labels_df['train']==real_label]
                                src_data_subset = src_data_subset.sample(n=nsamples,random_state=0)
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
                                tgt_data_subset = tgt_data[boolComb]
                                tgt_data_subset = tgt_data_subset.sample(n=nsamples,random_state=inference_seed)
                                tgt_labels_subset = pd.DataFrame([pseudo_label]*len(tgt_data_subset)) #tgt_labels[boolComb]
                                """ Retrieve Matched Labelled Data from Different CLass """
                                src_data_subset = dict_df['train'][labels_df['train']==real_label]
                                src_data_subset = src_data_subset.sample(n=nsamples,random_state=0)
                                src_labels_subset = pd.DataFrame([real_label]*len(src_data_subset))
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
                        if scale == True:
                            scaler_type = 'Standard' #options: 'Standard' | 'MinMax'
                            if goal in ['train and evaluate model','validate tgt data']:
                                scaler = load_scaler(scaler_type)
                                train_inputs3, train_outputs3 = scale_inputs(scaler,train_inputs2,'train'), train_outputs2.copy()
                                val_inputs3, val_outputs3 = scale_inputs(scaler,val_inputs2,'val'), val_outputs2.copy()
                            elif goal in ['evaluate model','calibrate model','calibrate pdm','perform inference']:
                                scaler = load('StandardScaler%s.joblib' % (str(nfeatures)))
                                val_inputs3, val_outputs3 = pd.DataFrame(scaler.transform(val_inputs2)), val_outputs2.copy()
                        else:
                             train_inputs3, train_outputs3 = train_inputs2.copy(), train_outputs2.copy()
                             val_inputs3, val_outputs3 = val_inputs2.copy(), val_outputs2.copy()
                        
                        if save_model_flag == True:
                            dump(scaler,'StandardScaler%s.joblib' % (str(nfeatures)))
                        
                        """ Load Model, Train, and Evaluate """
                        for model_type in settings['model list']:
                            print('Starting Setting %i/%i' % (count,ncombos))
                                    
                            if model_type in ['LR','RF','XGB','SVM']:
                                if goal in ['train and evaluate model','validate tgt data']:
                                    model = load_model(model_type,goal=goal,verbose=verbose)
                                    train_model(model, train_inputs3, train_outputs3, model_type)
                                    
                                    if save_model_flag == True:
                                        dump(model,'%s%s.joblib' % (model_type,str(nfeatures)))
                                        #dump(scaler,'StandardScaler%s.joblib' % (str(nfeatures)))
                                    
                                    """ Needs to be Changed """
                                    train_metrics = evaluate_model(model, train_inputs3, train_outputs3, dict_df['train'])
                                    val_metrics = evaluate_model(model, val_inputs3, val_outputs3, dict_df['val'])
                                    
                                    train_acc,train_auc,train_prec,train_rec = train_metrics
                                    val_acc,val_auc,val_prec,val_rec = val_metrics
                                elif 'calibrate' in goal:
                                    if goal == 'calibrate model':
                                        model = LogisticRegression() #needed for Platt Scaling
                                        val_probs_df = pd.read_csv('%s_to_%s_Val_Probs_NN5000.csv' % (src,src),index_col=0)
                                        val_probs = pd.DataFrame(val_probs_df['Prob'])
                                        val_outputs = pd.DataFrame(val_probs_df['Label'])
                                        model.fit(val_probs,val_outputs) #inputs are output probs, labels are class label 0 vs. 1
                                        calib_probs = pd.DataFrame(model.predict_proba(val_probs)[:,1])
                                        calib_probs_df = pd.concat((calib_probs,val_probs,val_outputs),1)
                                        calib_probs_df.columns = ['Calib Prob','Prob','Label']
                                    elif goal == 'calibrate pdm':
                                        df = pd.read_csv('electronics_validation_OT.csv',index_col=0)
                                        g = df.groupby(by=['Threshold','Seed']).apply(lambda row:row[row['Setup'] == 'Pseudo Low Label'].mean() - row[row['Setup'] == 'Pseudo High Label'].mean())
                                        g = g.iloc[:,2:].reset_index()
                                        pdm = pd.DataFrame(g['Val AUC'])
                                        thresh = pd.DataFrame(g['Threshold'])
                                        model = LinearRegression()
                                        model.fit(pdm,thresh)
                                        calib_pdm = pd.DataFrame(model.predict(pdm))
                                        calib_probs_df = pd.concat((calib_pdm,pdm,thresh),1)
                                        calib_probs_df.columns = ['Calib Prob','Prob','Label']
                                    
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
                                    curr_df = pd.DataFrame([model_type,inference_threshold_side,inference_threshold,inference_seed,val_acc,val_auc,val_prec,val_rec]).T  
                                curr_df.columns = cols
                                results_df = pd.concat((results_df,curr_df),0)
                            elif goal in ['perform inference']:
                                probs_df = pd.DataFrame(expit(probs_dict['inference']))
                                probs_df.columns = ['Prob']
                                probs_df['Label'] = inputs_and_outputs['inference']['outputs']
                                probs_df.to_csv('%s_to_%s_Probs_NN5000.csv' % (src,tgt))
                            
                            if goal in ['evaluate model']:
                                probs_df = pd.DataFrame(expit(probs_dict['val']))
                                probs_df.columns = ['Prob']
                                probs_df['Label'] = inputs_and_outputs['val']['outputs']
                                probs_df.to_csv('%s_to_%s_Val_Probs_NN5000.csv' % (src,src))
                            
                            count += 1
    
    if goal in ['train and evaluate model','evaluate model','validate tgt data']:
        return results_df, None #val_probs_df
        
    elif goal in ['perform inference']:
        return None, probs_df
        
    elif 'calibrate' in goal:
        if goal in['calibrate model']:
            calib_probs_df.to_csv('%s_to_%s_Calib_Probs_NN5000.csv' % (src,src))
            dump(model,'Platt_Model_Probs.joblib')
        elif goal in ['calibrate pdm']:
            calib_probs_df.to_csv('%s_to_%s_Calib_PDM_NN5000.csv' % (src,src))
            dump(model,'Platt_Model_PDM.joblib')
        return None, calib_probs_df

#%%
def get_settings(inference_threshold_side_list,inference_threshold_list):
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
    scale_list = [True]
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
                }
    
    if goal == 'validate tgt data':
        df = pd.DataFrame()
        for inference_threshold_side_list,inference_threshold_list in zip([['low'],['high']],[np.arange(0.1,0.6,0.1),[0.9, 0.8, 0.7, 0.6, 0.5]]):
            for switch in [False,True]:
                settings = get_settings(inference_threshold_side_list,inference_threshold_list)
                results_df, probs_df = run_variants(settings,setting=setting,label_name='Combined',suffix=suffix,lab_collapse_strategy=lab_collapse_strategy,
                                             doc_collapse_strategy=doc_collapse_strategy,labelled_and_unlabelled=labelled_and_unlabelled,goal=goal,batch_size=batch_size,
                                             dataset_type=dataset_type,max_inference_seed=max_inference_seed,lr=lr,verbose=verbose,save_model_flag=save_model_flag,switch=switch)
                
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
if visualize_stuff == True:
    sns.set(font_scale=2)
    sns.set_style('darkgrid')
    
    df = pd.read_csv('electronics_validation_OR.csv',index_col=0)
    low_label, high_label = 'Pseudo Negative', 'Pseudo Positive'
    df['Setup'] = df['Setup'].replace({'Pseudo Low Label':low_label,'Pseudo High Label':high_label})    
    #palette = {'Pseudo Low Label': 'royalblue', 'Pseudo High Label': 'forestgreen'}
    palette = {low_label: 'royalblue', high_label: 'forestgreen'}
    
    g = df.groupby(by=['Threshold']).apply(lambda row:row[row['Setup'] == low_label].mean() - row[row['Setup'] == high_label].mean())
    g['Threshold'] = g.index
    g['Val AUC Abs'] = g['Val AUC'].abs()
    g['Color'] = g['Val AUC'].apply(lambda auc: 'royalblue' if auc > 0 else 'forestgreen')
    
    """ Violin Plot """
    fig,axes = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':(0.05,0.95)})
    axes[0].remove()
    ax = axes[1]
    sns.violinplot(x='Threshold',y='Val AUC',hue='Setup',data=df,ax=ax,palette=palette)
    ax.set_xticklabels(list(map(lambda el:'%.2f' % el,sorted(df['Threshold'].unique().tolist()))))
    ax.set_xlabel('Threshold')
    legend,handles = ax.get_legend_handles_labels()
    ax.legend(legend,handles,loc='upper center',bbox_to_anchor=(0.5,1.2),frameon=True,ncol=2,facecolor='white')
    ax.set_ylim([0.40,0.75])
    
    #%%
    fig,axes = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':(0.05,0.95)})
    axes[0].remove()
    ax = axes[1]    
    sns.barplot(x=np.arange(len(g)),y='Val AUC',data=g,ec='black',ax=ax,palette=g['Color'])
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Pseudo Label Discrepancy')
    ax.set_xticklabels(list(map(lambda el:'%.2f' % el,sorted(g['Threshold'].unique().tolist()))))
    ax.set_ylim([-0.10,0.20])
    
    #%%
    """ Correlation Plot """
    probs_df = pd.read_csv('books_to_electronics_Probs_NN5000_OR.csv',index_col=0)
    
    prob_col = 'Prob' #option: Calib Prob
    probs_df['Range'] = pd.cut(probs_df[prob_col],bins=np.arange(0,1.1,0.1),labels=np.arange(0.1,1.1,0.1))
    counts = probs_df['Range'].value_counts().tolist()
    acc = probs_df.groupby(by=['Range'])['Label'].mean().reset_index(drop=True)
    acc.index = g.index
    g['Proportion High'] = acc #1-acc
    
    g['Count Weight'] = np.array(counts)/sum(counts)
    g['Bin Acc'] = acc
    bin_probs = probs_df.groupby(by=['Range'])[prob_col].mean()
    bin_probs.index = g.index
    g['Bin Prob'] = bin_probs
    ece = np.sum(np.abs(g['Bin Prob'] - g['Bin Acc']) * g['Count Weight'])
    
    """ ece for PDM """    
    probs_df = pd.read_csv('books_to_books_Val_Probs_NN5000.csv',index_col=0)
    prob_col = 'Prob' #option: Calib Prob
    probs_df['Range'] = pd.cut(probs_df[prob_col],bins=np.arange(0,1.1,0.1),labels=np.arange(0.1,1.1,0.1))
    counts = probs_df['Range'].value_counts().tolist()
    acc = probs_df.groupby(by=['Range'])['Label'].mean().reset_index(drop=True)
    acc.index = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    acc = pd.DataFrame(acc)
    
    probs_df = pd.read_csv('books_to_books_Calib_PDM_NN5000.csv',index_col=0)
    probs_df['Label'] = probs_df['Label'].round(1)
    probs_df.index = probs_df['Label']
    g = probs_df.merge(acc,how='left',left_index=True,right_index=True)    
    g.columns = ['Calib PDM','PDM','Threshold','Bin Acc']
    ece = np.mean(np.abs(g['Calib PDM'] - g['Bin Acc']))# * g['Count Weight'])
    
    
    
    
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(pd.DataFrame(g['Val AUC']),pd.DataFrame(g['Proportion High']))
    corr = g[['Val AUC','Proportion High']].corr()
    corr = corr['Val AUC']['Proportion High']
    line = m.predict(pd.DataFrame(np.arange(g['Val AUC'].min()-0.05,g['Val AUC'].max()+0.05,0.05)))
    fig,axes = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':(0.05,0.95)})
    axes[0].remove()
    ax = axes[1]
    sns.scatterplot(g['Val AUC'],g['Proportion High'],ci=None,s=200,ec='k',color='goldenrod',ax=ax,label=r'$\rho$ = %.2f' % corr)
    sns.lineplot(np.arange(g['Val AUC'].min()-0.05,g['Val AUC'].max()+0.05,0.05),line.squeeze(),lw=2,color='royalblue',ax=ax)#,label=r'$\rho$=0.95')
    plt.xlabel('Pseudo Label Discrepancy')
    plt.ylabel('Accuracy')
    plt.ylim([0,1.1])
    
    #%%
    """ Histogram Plot """
    probs_df = pd.read_csv('books_to_electronics_Probs_NN5000_OT.csv',index_col=0)
    fig,axes = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':(0.05,0.95)})
    axes[0].remove()
    ax = axes[1]
    probs_df['Label'] = probs_df['Label'].replace({0:'Negative',1:'Positive'})
    palette = {'Negative': 'royalblue', 'Positive': 'forestgreen'}
    sns.histplot(x='Prob',hue='Label',data=probs_df,palette=palette,bins=50,ax=ax)
    legend = ax.get_legend()
    handles = legend.legendHandles
    ax.legend(handles,['Negative','Positive'],loc='upper center',bbox_to_anchor=(0.5,1.2),frameon=True,ncol=2,facecolor='white')
    ax.set_xlabel('Probability')
    
    
    
    #%%
    """ Learn Function to Map PDM or Confidence to Accuracy """
    suffix = 'OT'
    df = pd.read_csv('books_validation_%s.csv' % suffix,index_col=0)
    low_label, high_label = 'Pseudo Negative', 'Pseudo Positive'
    df['Setup'] = df['Setup'].replace({'Pseudo Low Label':low_label,'Pseudo High Label':high_label})    
    g = df.groupby(by=['Threshold','Seed'])[['Val AUC','Setup']].apply(lambda row:row[row['Setup'] == low_label]['Val AUC'] - row[row['Setup'] == high_label]['Val AUC']).reset_index()
    g['Threshold'] = g['Threshold'].round(1)
    
    probs_df = pd.read_csv('books_to_books_Test_Probs_NN5000_%s.csv' % suffix,index_col=0)
    prob_col = 'Prob'
    probs_df['Threshold'] = pd.cut(probs_df[prob_col],bins=np.arange(0,1.1,0.1),labels=np.arange(0.1,1.1,0.1))
    #counts = probs_df['Threshold'].value_counts().tolist()
    acc = probs_df.groupby(by=['Threshold'])['Label'].mean().reset_index()
    acc['Threshold'] = acc['Threshold'].astype(float)
    acc['Threshold'] = acc['Threshold'].round(1)
    df = g.merge(acc,how='left',on='Threshold')
    df.columns = ['Threshold','Seed','PDM','Acc']
    
    """ PDM to Accuracy """
    inputs = pd.DataFrame(df['PDM'])
    outputs = pd.DataFrame(df['Acc'])
    model = LinearRegression()
    model.fit(inputs,outputs)
    dump(model,'Platt_Model_PDM_%s.joblib' % suffix)
    
    """ Platt Scaling """
    inputs = pd.DataFrame(probs_df['Prob'])
    outputs = pd.DataFrame(probs_df['Label'])
    model = LogisticRegression()
    model.fit(inputs,outputs)    
    dump(model,'Platt_Model_Probs_%s.joblib' % suffix)
    
    #%%
    suffix = 'OT'
    
    pdm_df = pd.read_csv('electronics_validation_%s.csv' % suffix,index_col=0)
    low_label, high_label = 'Pseudo Negative', 'Pseudo Positive'
    pdm_df['Setup'] = pdm_df['Setup'].replace({'Pseudo Low Label':low_label,'Pseudo High Label':high_label})    
    g = pdm_df.groupby(by=['Threshold','Seed'])[['Val AUC','Setup']].apply(lambda row:row[row['Setup'] == low_label]['Val AUC'] - row[row['Setup'] == high_label]['Val AUC']).reset_index()
    g['Threshold'] = g['Threshold'].round(1)
    g.columns = ['Threshold','Seed','PDM']

    probs_df = pd.read_csv('books_to_electronics_Probs_NN5000_%s.csv' % suffix,index_col=0)
    prob_col = 'Prob'
    probs_df['Threshold'] = pd.cut(probs_df[prob_col],bins=np.arange(0,1.1,0.1),labels=np.arange(0.1,1.1,0.1))
    #counts = probs_df['Threshold'].value_counts().tolist()
    acc = probs_df.groupby(by=['Threshold'])['Label'].mean().reset_index()
    acc['Threshold'] = acc['Threshold'].astype(float)
    acc['Threshold'] = acc['Threshold'].round(1)
    df = g.merge(acc,how='left',on='Threshold')
    df.columns = ['Threshold','Seed','PDM','Acc']

    pdm_model = load('Platt_Model_PDM_%s.joblib' % suffix)
    df['Pred Acc'] = pdm_model.predict(pd.DataFrame(df['PDM']))
    df['Error'] = df[['Acc','Pred Acc']].apply(lambda row:np.power(row['Pred Acc'] - row['Acc'],2),axis=1)
    pdm_error = df['Error'].mean()
    
    prob_model = load('Platt_Model_Probs_%s.joblib' % suffix)
    probs_df['Calib Prob'] = prob_model.predict_proba(pd.DataFrame(probs_df['Prob']))[:,1]
    prob_col = 'Calib Prob'
    probs_df['Threshold'] = pd.cut(probs_df[prob_col],bins=np.arange(0,1.1,0.1),labels=np.arange(0.1,1.1,0.1))
    acc = probs_df.groupby(by=['Threshold'])['Label'].mean().reset_index()
    acc['Threshold'] = acc['Threshold'].astype(float)
    acc['Threshold'] = acc['Threshold'].round(1)
    acc.columns = ['Threshold','Acc']
    prob_error = np.mean(np.power(acc['Threshold'] - acc['Acc'],2))


