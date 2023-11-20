#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:57:17 2023

@author: dani.kiyasseh

This script is used to:
1) conduct the SUDO experiments on the Camelyon17-WILDS dataset
"""

# [USER MUST MODIFY]
path_to_wilds_folder = '/Users/dani.kiyasseh/Desktop/wilds'
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
from scipy import interpolate

from prepare_miscellaneous import load_model, train_model                            
from network import train, train_for_OOD
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

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

    # Number of samples in the trainin, validation, and test splits 
    ntrain = 302436
    ntrain_downsampled = 128*199 # limits number of samples in the training set (must match value chosen in extract_camelyon_features.py)
    nval = 33560
    ntest = 85054
    
    train_filenames = np.arange(0,ntrain_downsampled)
    val_filenames = np.arange(ntrain,ntrain+nval)
    test_filenames = np.arange(ntrain+nval,ntrain+nval+ntest)
    
    # Load the saved features from each split
    with open(os.path.join(path_to_wilds_folder,'camelyon17/train/features'),'rb') as f:
        src_features_dict = pickle.load(f)
    with open(os.path.join(path_to_wilds_folder,'camelyon17/id_val/features'),'rb') as f:
        val_features_dict = pickle.load(f)
    with open(os.path.join(path_to_wilds_folder,'camelyon17/test/features'),'rb') as f:
        tgt_features_dict = pickle.load(f)
    # Load the saved labels from each split
    with open(os.path.join(path_to_wilds_folder,'camelyon17/train/labels'),'rb') as f:
        src_labels_dict = pickle.load(f)
    with open(os.path.join(path_to_wilds_folder,'camelyon17/id_val/labels'),'rb') as f:
        val_labels_dict = pickle.load(f)
    with open(os.path.join(path_to_wilds_folder,'camelyon17/test/labels'),'rb') as f:
        tgt_labels_dict = pickle.load(f)
    
    for nfeatures in settings['nfeatures list']:
        
        """ Loops for Validating Unlabelled Data Pseudo Labels """
        for inference_seed in range(max_inference_seed):
            
            for inference_threshold_side in settings['inference threshold side list']: #looking at low pseudo or high pseudo labels 
                if inference_threshold_side == 'low':
                    prev_threshold = 0.10
                elif inference_threshold_side == 'high':
                    prev_threshold = 1
            
                for inference_threshold in settings['inference threshold list']: #threshold for performing pseudo vs. real classification 
                    dict_df = dict()
                    dict_df['train'] = pd.DataFrame(list(itemgetter(*train_filenames)(src_features_dict)))
                    dict_df['val'] = pd.DataFrame(list(itemgetter(*val_filenames)(val_features_dict)))
                    labels_df = dict()
                    labels_df['train'] = pd.DataFrame(list(itemgetter(*train_filenames)(src_labels_dict)))
                    labels_df['val'] = pd.DataFrame(list(itemgetter(*val_filenames)(val_labels_dict)))
                    
                    if goal in ['validate tgt data']:
                        """ Retrieve Desired Subset of Unlablled Data """
                        probs_df = pd.read_csv(os.path.join(path_to_wilds_folder,'camelyon17/test/camelyon17_test_Probs.csv'),index_col=0)
                        prob_col = 'Prob'
                        
                        if inference_threshold_side == 'low':
                            if switch == False:
                                pseudo_label = 0
                                real_label = 1
                            elif switch == True:
                                pseudo_label = 1
                                real_label = 0
                            
                            nsamples = 1000 # Number of data points to sample from each probability interval
                            bool1 = probs_df[prob_col] > prev_threshold
                            bool2 = probs_df[prob_col] <= inference_threshold
                            boolComb = bool1 & bool2
                            prev_threshold = inference_threshold
                            """ Retrieve Unlabelled Data """
                            subset_probs_df = probs_df[boolComb]
                            subset_probs_df['index'] = subset_probs_df.index
                            tgt_files = subset_probs_df['index'].sample(n=nsamples,random_state=inference_seed)
                            tgt_data_subset = pd.DataFrame(list(itemgetter(*tgt_files)(tgt_features_dict)))
                            tgt_labels_subset = pd.Series([pseudo_label]*len(tgt_data_subset))
                            """ Retrieve Matched Labelled Data from Different CLass """
                            src_labels_df = pd.DataFrame.from_dict(src_labels_dict,orient='index')
                            src_data_subset = src_labels_df[src_labels_df[0] == real_label]
                            src_data_subset['index'] = src_data_subset.index
                            src_files = src_data_subset['index'].sample(n=nsamples,random_state=inference_seed)                            
                            src_data_subset = pd.DataFrame(list(itemgetter(*src_files)(src_features_dict)))
                            src_labels_subset = pd.Series([real_label]*len(src_data_subset))
                        elif inference_threshold_side == 'high':
                            if switch == False:
                                pseudo_label = 1
                                real_label = 0
                            elif switch == True:
                                pseudo_label = 0
                                real_label = 1
                            nsamples = 1000 # Number of data points to sample from each probability interval
                            bool1 = probs_df[prob_col] < prev_threshold
                            bool2 = probs_df[prob_col] >= inference_threshold
                            boolComb = bool1 & bool2
                            prev_threshold = inference_threshold
                            """ Retrieve Unlabelled Data """
                            subset_probs_df = probs_df[boolComb]
                            tgt_files = subset_probs_df.index.sample(n=nsamples,random_state=inference_seed)
                            tgt_data_subset = pd.DataFrame(list(itemgetter(*tgt_files)(tgt_features_dict)))
                            tgt_labels_subset = pd.Series([pseudo_label]*len(tgt_data_subset))
                            """ Retrieve Matched Labelled Data from Different CLass """
                            src_labels_df = pd.DataFrame.from_dict(src_labels_dict,orient='index')
                            src_data_subset = src_labels_df[src_labels_df[0] == real_label]
                            src_files = src_data_subset.index.sample(n=nsamples,random_state=inference_seed)
                            src_data_subset = pd.DataFrame(list(itemgetter(*src_files)(src_features_dict)))
                            src_labels_subset = pd.Series([real_label]*len(src_data_subset))
                        """ Combine Pseudo Unlabelled Data and Real Labelled Data """
                        data_subset = pd.concat((src_data_subset,tgt_data_subset),0).sample(n=nsamples*2,replace=False,random_state=0)
                        labels_subset = pd.concat((src_labels_subset,tgt_labels_subset),0).sample(n=nsamples*2,replace=False,random_state=0)
                        dict_df['train'] = data_subset
                        labels_df['train'] = labels_subset
                    
                    train_inputs2, val_inputs2 = pd.DataFrame(dict_df['train']), pd.DataFrame(dict_df['val'])
                    train_outputs2, val_outputs2 = labels_df['train'], labels_df['val']
                    
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
                                    curr_df = pd.DataFrame([model_type,inference_threshold_side,inference_threshold,inference_seed,val_acc,val_auc,val_prec,val_rec]).T  
                                curr_df.columns = cols
                                results_df = pd.concat((results_df,curr_df),0)
                            
                            count += 1
    
    if goal in ['train and evaluate model','evaluate model','validate tgt data']:
        return results_df, None #val_probs_df
        
    elif goal in ['perform inference']:
        return None, probs_df

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
                'model_type':model_type                 
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
    model_type = 'DenseNet121' # options: HAM10000 | DeepDerm
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
        for inference_threshold_side_list,inference_threshold_list in zip([['low']],[np.arange(0.15,0.75,0.05)]): #(0.05,0.5,0.05) for HAM10000 # (0.1,1.1,0.1) for DeepDerm # (0.1,0.5,0.1) for HAM10000 fairness experiments 
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
def get_sudo_info(model_type='HAM10000',metric='Val AUC',wild_data_setting='out-of-domain',validation_label_noise=0):
    df = pd.read_csv('Validating_Unlabelled_Data_Thresholds.csv',index_col=0)
    #bool1 = df['Wild Data Setting'] == wild_data_setting
    #boolComb = bool1 #& bool2
    #df = df[boolComb]
    low_label, high_label = 'Pseudo Negative', 'Pseudo Positive'
    df['Setup'] = df['Setup'].replace({'Pseudo Low Label':low_label,'Pseudo High Label':high_label})    
    """ NOTE - I swapped order of calculation b/c original model was trained on swapped labels - it just flips the sudo value """
    g = df.groupby(by=['Threshold']).apply(lambda row:row[row['Setup'] == high_label].mean() - row[row['Setup'] == low_label].mean())
    g['Threshold'] = g.index
    #g['Val AUC Abs'] = g[metric].abs()
    g['Color'] = g[metric].apply(lambda auc: 'royalblue' if auc > 0 else 'forestgreen')
    return df, g
    
def generate_violin_plot(model_type='HAM10000',metric='Val AUC',wild_data_setting='out-of-domain',validation_label_noise=0):
    df, g = get_sudo_info(model_type,metric,wild_data_setting,validation_label_noise)
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

def generate_sudo_plot(model_type='HAM10000',metric='Val AUC',wild_data_setting='out-of-domain',validation_label_noise=0):
    df, g = get_sudo_info(model_type,metric,wild_data_setting,validation_label_noise)
    fig,axes = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':(0.05,0.95)})
    axes[0].remove()
    ax = axes[1]
    sns.barplot(x=np.arange(len(g)),y=metric,data=g,ec='black',ax=ax,palette=g['Color'])
    ax.set_xlabel('Threshold')
    ax.set_ylabel(r'$\mathrm{SUDO}_{\mathrm{%s}}$' % metric.split(' ')[1])
    ax.set_xticklabels(list(map(lambda el:'%.2f' % el,sorted(g['Threshold'].unique().tolist()))))
    return fig
    
def get_correlation(model_type='HAM10000',metric='Val AUC',wild_data_setting='out-of-domain',validation_label_noise=0):
    """ Correlation Plot """
    df, g = get_sudo_info(model_type,metric,wild_data_setting,validation_label_noise)
    probs_df = pd.read_csv('test/camelyon17_test_Probs.csv',index_col=0)
    
    prob_col = 'Prob' #option: Calib Prob
    bins = np.arange(0.15,0.80,0.05) #np.arange(0,0.5,0.05) #np.arange(0,1.1,0.1)
    labels = np.arange(0.15,0.75,0.05) #np.arange(0.05,0.5,0.05)
    probs_df['Range'] = pd.cut(probs_df[prob_col],bins=bins,labels=labels)
    counts = probs_df['Range'].value_counts().tolist()
    acc = probs_df.groupby(by=['Range'])['Label'].mean().reset_index(drop=True)
    acc.index = g.index
    g['Proportion High'] = 1-acc # b/c I think original model was trained with flipped labels (0 was 1 and 1 was 0)
    
    g['Count Weight'] = np.array(counts)/sum(counts)
    g['Bin Acc'] = acc
    bin_probs = probs_df.groupby(by=['Range'])[prob_col].mean()
    bin_probs.index = g.index
    g['Bin Prob'] = bin_probs
    
    corr = g[[metric,'Proportion High']].corr()
    corr = corr[metric]['Proportion High']
    return df, g, corr

def generate_correlation_plot(model_type='HAM10000',metric='Val AUC',wild_data_setting='out-of-domain',validation_label_noise=0):
    df, g, corr = get_correlation(model_type,metric,wild_data_setting,validation_label_noise)
    m = LinearRegression()
    m.fit(pd.DataFrame(g[metric]),pd.DataFrame(g['Proportion High']))
    
    line = m.predict(pd.DataFrame(np.arange(g[metric].min()-0.05,g[metric].max()+0.05,0.05)))
    fig,axes = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':(0.05,0.95)})
    axes[0].remove()
    ax = axes[1]
    sns.scatterplot(g[metric],g['Proportion High'],ci=None,s=200,ec='k',color='goldenrod',ax=ax,label=r'$\rho$ = %.2f' % corr)
    sns.lineplot(np.arange(g[metric].min()-0.05,g[metric].max()+0.05,0.05),line.squeeze(),lw=2,color='royalblue',ax=ax)#,label=r'$\rho$=0.95')
    plt.xlabel(r'$\mathrm{SUDO}_{\mathrm{%s}}$' % metric.split(' ')[1])
    plt.ylabel('Proportion of positive instances')
    return fig
    #plt.ylim([0,1.1])
    
def generate_histogram_plot(model_type='HAM10000',wild_data_setting='out-of-domain'):
    """ Histogram Plot """
    probs_df = pd.read_csv('test/camelyon17_test_Probs.csv',index_col=0)
    fig,axes = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':(0.05,0.95)})
    axes[0].remove()
    ax = axes[1]
    probs_df['Label'] = probs_df['Label'].replace({1:'Negative',0:'Positive'})
    palette = {'Negative': 'royalblue', 'Positive': 'forestgreen', 2: 'black'}
    sns.histplot(x='Prob',hue='Label',data=probs_df,palette=palette,bins=50,ax=ax)
    legend = ax.get_legend()
    handles = legend.legendHandles
    ax.legend(handles[::-1],['Negative','Positive'],loc='upper center',bbox_to_anchor=(0.5,1.2),frameon=True,ncol=2,facecolor='white')
    ax.set_xlabel('AI-based probability')
    ax.set_ylabel('Number of instances')
    ax.set_xlim([0,1])
    return fig
    
#%%
""" Reliability-Completeness Curves """
def generate_reliability_curve(model_type='HAM10000'):
    df = pd.read_csv('Validating_Unlabelled_Data_Thresholds.csv',index_col=0).reset_index(drop=True)
    
    low_label, high_label = 'Pseudo Negative', 'Pseudo Positive'
    df['Setup'] = df['Setup'].replace({'Pseudo Low Label':low_label,'Pseudo High Label':high_label})    
    g = df.groupby(by=['Threshold']).apply(lambda row:row[row['Setup'] == low_label].mean() - row[row['Setup'] == high_label].mean())
    g['Threshold'] = g.index
    g['Val AUC Abs'] = g['Val AUC'].abs()
    #g['Pred Acc'] = ols.predict(pd.DataFrame(g['Val AUC'])) # this is just so we can get closer to some absolute accuracy metric (not essential for comparing models)
    
    probs_df = pd.read_csv('DDI_Probs_%s.csv' % model_type,index_col=0)
    probs_df['Range'] = pd.cut(probs_df[prob_col],bins=[0]+g.index.tolist(),labels=g.index.tolist())#,duplicates='drop',ordered='False')
    counts = probs_df['Range'].value_counts().tolist()
    g['Counts'] = counts
    
    ### END ###
    
    low_thresh_list = [0.10,0.20,0.30,0.40,0.50]
    high_thresh_list = [0.75,0.70,0.65,0.60,0.50]
    
    rel = []
    com = []
    for l,r in zip(low_thresh_list,high_thresh_list):
        bool1 = g['Threshold'] <= l
        bool2 = g['Threshold'] >= r
        boolcomb = bool1 | bool2
        subset = g[boolcomb]
        subset1 = g[bool1]
        subset2 = g[bool2]
        curr_rel1 = subset1['Val AUC Abs'].mean() # add 1- if using OLS predictor
        curr_rel2 = 1-subset2['Val AUC Abs'].mean() # remove 1- if using OLS predictor
        rel.append((curr_rel1 + curr_rel2) / 2)
        curr_com = subset['Counts'].sum() / g['Counts'].sum()
        com.append(curr_com)
    
    from sklearn.metrics import auc
    rel = [1] + rel
    com = [0] + com
    area = auc(com,rel)
    
    rc_df = pd.DataFrame([rel,com]).T
    rc_df.columns = ['Reliability','Completeness']
    f = interpolate.interp1d(rc_df['Completeness'],rc_df['Reliability'],kind='quadratic')
    xnew = np.linspace(0,1,50)
    ynew = f(xnew)
    rc_df = pd.DataFrame([xnew,ynew]).T
    rc_df.columns = ['Completeness','Reliability']
    return rc_df, area

def generate_reliability_completeness_plot(rc_df,areas):
    fig,axes = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':(0.05,0.95)})
    axes[0].remove()
    ax = axes[1]
    #ax.fill_between(rc_df['Completeness'],0,rc_df['Reliability'],hatch='X',facecolor='none',ec='orange')
    #sns.scatterplot(x='Completeness',y='Reliability',hue='Model',data=rc_df,s=100,ec='k',ax=ax,color='orange')#,label='AURCC = %.2f' % area)
    sns.lineplot(x='Completeness',y='Reliability',hue='Model',data=rc_df,lw=3,ax=ax,palette=['goldenrod','darkorchid'])#,label='AURCC = %.2f' % area)
    ax.set_ylim([0,1.1])
    ax.set_xlabel('Completeness of predictions')
    ax.set_ylabel('Reliability of predictions')
    handles,labels = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_linewidth(3)
    labels = [label + ' - ' '%.3f' % area for label,area in zip(labels,areas)]
    ax.legend(handles,labels,loc='lower right',title='AURCC')
    return fig
    #sns.scatterplot(x=[0.797],y=[0.639],s=200,ec='k',ax=ax,color='black')
#%%
sns.set(font_scale=2)
sns.set_style('ticks')
plot_type = 'sudo' # options: histogram | sudo | correlation 
metric = 'Val AUC' # options: Val AUC | Val Precision | Val Recall | Val ACC
save_images = True
save_path = os.path.join(path_to_wilds_folder,'camelyon17/results')

if plot_type == 'histogram':
    fig = generate_histogram_plot(model_type)
elif plot_type == 'correlation':
    fig = generate_correlation_plot(model_type,metric=metric)
elif plot_type == 'sudo':
    fig = generate_sudo_plot(model_type,metric=metric)
    
if save_images == True:
    save_name = 'camelyon17_%s_%s.png' % (plot_type,metric.split(' ')[1].lower())
    fig.savefig(os.path.join(save_path,save_name))

#%%
""" Generate pair of RC curves """
rc_df1,area1 = generate_reliability_curve('HAM10000')
rc_df1['Model'] = 'HAM10000'
rc_df2,area2 = generate_reliability_curve('DeepDerm')
rc_df2['Model'] = 'DeepDerm'
rc_df = pd.concat((rc_df1,rc_df2),0)
areas = [area1,area2]
fig = generate_reliability_completeness_plot(rc_df,areas)
save_name = 'ddi_rc_curve.png'
fig.savefig(os.path.join(save_path,save_name))
