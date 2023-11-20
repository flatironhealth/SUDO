#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 20:43:14 2021

@author: dani.kiyasseh

This script is used to:
1) define helper functions for preparing the data
"""

import os 
os.chdir('/Users/dani.kiyasseh/Desktop/Scripts')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder,  LabelEncoder
import random
from tqdm import tqdm
tqdm.pandas()
label_encoder = LabelEncoder()

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans

from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

os.chdir('/Users/dani.kiyasseh/Desktop/Data')

#%%
class Impute(object):
    
    def __init__(self,train_df,demo_list,imputation_type):
        if imputation_type == 'median':
            stats = train_df.groupby(by=demo_list).median().reset_index()
        elif imputation_type == 'mean':
            stats = train_df.groupby(by=demo_list).mean().reset_index()            
        self.demo_list = demo_list
        self.stats = stats
        
    def retrieve_val(self,x,lab):
        if self.demo_list == ['age','gender','race']:
            val = self.retrieve_val_age_gender_race(x,lab)
        elif self.demo_list == ['age','gender']:
            val = self.retrieve_val_age_gender(x,lab)
        elif self.demo_list == ['age']:
            val = self.retrieve_val_age(x,lab)
        elif self.demo_list == ['gender']:
            val = self.retrieve_val_gender(x,lab)
        return val

    def retrieve_val_gender(self,x,lab):
        bool2 = self.stats['gender'] == x['gender'].iloc[0]
        boolcomb = bool2
        
        val = self.stats[lab][boolcomb]
        if len(val) > 0:
            val = val.item()
        else:
            val = float('nan')    
        
        return val     
    
    def retrieve_val_age(self,x,lab):
        bool1 = self.stats['age'] == x['age'].iloc[0]
        boolcomb = bool1
        
        val = self.stats[lab][boolcomb]
        if len(val) > 0:
            val = val.item()
        else:
            val = float('nan')    
        
        return val                 

    def retrieve_val_age_gender(self,x,lab):
        bool1 = self.stats['age'] == x['age'].iloc[0]
        bool2 = self.stats['gender'] == x['gender'].iloc[0]
        boolcomb = bool1 & bool2
        
        val = self.stats[lab][boolcomb]
        if len(val) > 0:
            val = val.item()
        else:
            val = float('nan')    
        
        return val            

    def retrieve_val_age_gender_race(self,x,lab):
        bool1 = self.stats['age'] == x['age'].iloc[0]
        bool2 = self.stats['gender'] == x['gender'].iloc[0]
        bool3 = self.stats['race'] == x['race'].iloc[0]
        boolcomb = bool1 & bool2 & bool3
        
        val = self.stats[lab][boolcomb]
        if len(val) > 0:
            val = val.item()
        else:
            val = float('nan')    
        
        return val

#%%
def impute_held_out_set(train_df,val_df,imputation_type):
    #train_df, val_df, and test_df contain all columns
    #final_lab_index = np.where([col == 'Structured' for col in train_df.columns])[0][0]
    #labs_to_keep = train_df.columns[2:final_lab_index]
    indices = val_df.index
    columns = train_df.columns
    col_indices = np.where(train_df.isna().sum() > 0)[0] #only iterate over columns with nan and those that need imputation
    labs_to_keep = [columns[index] for index in col_indices]
    
    val_df = val_df.copy().reset_index(drop=True)
    entire_list = [['age','gender','race'],['age','gender'],['age'],['gender']]
    for i,demo_list in enumerate(entire_list):
        print('%i Pass Imputation...' % i)
        imputer = Impute(train_df,demo_list,imputation_type)
        for lab in labs_to_keep:
            imputed_df = val_df.groupby(by=demo_list).apply(lambda x:x[lab].fillna(imputer.retrieve_val(x,lab)))
            val_df[lab] = imputed_df.reset_index()[lab]
            #print(val_df[lab].isna().sum())
    
    val_df.index = indices
    return val_df

#%%
def load_scaler(scaler_type):
    """ Load scaler to scale inputs with 
    
    Args:
        scaler_type (string): type of scaler
        
    Returns:
        scaler (scaler object):
    """
    if scaler_type == 'Standard':
        scaler = StandardScaler()
    elif scaler_type == 'MinMax':
        scaler = MinMaxScaler()
    return scaler

def scale_inputs(scaler,inputs,phase,modality='structured'):
    """ Scale the input features 
    
    Args:
        scaler (scaler object): 
        inputs (pd.DataFrame): matrix of inputs [ N x D ] N = Number of Samples, D = Number of Features
                               Rows: depending on setup, this reflects features for a single instance
        phase (string): determine whether to fit and transform or only transform inputs 
    
    Returns:
        inputs (pd.DataFrame): scaled inputs [ N x D ]
    """
        
    if modality == 'structured':
        # cols = inputs.columns
        # cols_to_not_scale = [col for col in cols if 'Collected' in col]
        # cols_to_scale = list(set(cols) - set(cols_to_not_scale))
        cols = inputs.columns
        vars_to_scale = cols #labs
        cols_to_scale = [col for col in cols if col in vars_to_scale]
        cols_to_not_scale = list(set(cols) - set(cols_to_scale))
        inputs_to_scale = inputs.loc[:,cols_to_scale]
        inputs_to_not_scale = inputs.loc[:,cols_to_not_scale]
    else:
        inputs_to_scale = inputs
        inputs_to_not_scale = pd.DataFrame()
    
    if phase == 'train':
        inputs_scaled = scaler.fit_transform(inputs_to_scale)
    else:
        inputs_scaled = scaler.transform(inputs_to_scale)

    inputs_scaled = pd.DataFrame(inputs_scaled,index=inputs_to_scale.index,columns=inputs_to_scale.columns)        
    inputs = pd.concat((inputs_scaled,inputs_to_not_scale),1)
    return inputs

def scale_inputs_for_inference(scaler_type,inputs,modality='structured',dataset_type='',nfeatures='',suffix=''):
    """ Scale the input features. 
        Note this is only used for inference in special cases. 
    
    Args:
        scaler (scaler object): 
        inputs (pd.DataFrame): matrix of inputs [ N x D ] N = Number of Samples, D = Number of Features
                               Rows: depending on setup, this reflects features for a single instance
        phase (string): determine whether to fit and transform or only transform inputs 
    
    Returns:
        inputs (pd.DataFrame): scaled inputs [ N x D ]
    """
    if modality == 'structured':
        # cols = inputs.columns
        # cols_to_not_scale = [col for col in cols if 'Collected' in col]
        # cols_to_scale = list(set(cols) - set(cols_to_not_scale))
        cols = inputs.columns
        vars_to_scale = cols #labs
        cols_to_scale = [col for col in cols if col in vars_to_scale]
        cols_to_not_scale = list(set(cols) - set(cols_to_scale))
        inputs_to_scale = inputs.loc[:,cols_to_scale]
        inputs_to_not_scale = inputs.loc[:,cols_to_not_scale]
    else:
        inputs_to_scale = inputs
        inputs_to_not_scale = pd.DataFrame()
    
    from joblib import load 
    print('Scaling Inputs...')
    scaler = load('StandardScaler%s_%s%s.joblib' % (str(nfeatures),dataset_type,suffix))
    inputs_scaled = scaler.transform(inputs_to_scale)
    inputs_scaled = pd.DataFrame(inputs_scaled,index=inputs_to_scale.index,columns=inputs_to_scale.columns)        
    inputs = pd.concat((inputs_scaled,inputs_to_not_scale),1)
    return inputs

def load_reducer(reducer_type):
    if reducer_type == 'PCA':
        reducer = PCA()
    elif reducer_type == 'IPCA':
        reducer = IncrementalPCA()
    return reducer

def reduce_inputs(reducer,inputs,phase):
    if phase == 'train':
        inputs = reducer.fit_transform(inputs)
    else:
        inputs = reducer.transform(inputs)
    return inputs

#%%
def load_model(model_type,goal='train and evaluate model',verbose=0,dataset_type='',nfeatures=''):
    """ Load model that is used for classification (or other tasks)
    
    Args:
        model_type (string): type of model we want
        goal (string): goal we want to achieve (e.g., train and evaluate, perform inference)
    
    Returns:
        model (model object): actual model that can be trained later on
    """
    
    print('Loading Model...')
    if goal in ['train and evaluate model','validate unlabelled data','validate tgt data']:
        if model_type == 'LR':
            model = LogisticRegression(max_iter=1000,solver='lbfgs',penalty='l2',verbose=verbose)
        elif model_type == 'RF':
            model = RandomForestClassifier(n_estimators=1000,verbose=verbose)
        elif model_type == 'XGB':
            #model = GradientBoostingClassifier(n_estimators=1000,verbose=2)
            model = xgb.XGBClassifier()
        elif model_type == 'SVC':
            model = LinearSVC(verbose=True)
        elif model_type == 'Kmeans':
            model = MiniBatchKMeans(n_clusters=2,verbose=verbose)
    elif goal in ['evaluate model','calibrate model','perform inference']:
        from joblib import load
        model = load('%s%s_%s.joblib' % (model_type,str(nfeatures),dataset_type)) #already saved model
        
    return model

def train_model(model,train_inputs,train_outputs,model_type='LR'):
    """ Train the model and learn the parameters 
    
    Args:
        model (model object): 
        train_inputs (pd.DataFrame): matrix of inputs [ N x D ] N = Number of Samples, D = Number of Features
        train_outputs (pd.DataFrame): matrix of outputs [ N x 1 ]  
    
    Returns:
        None 
    """
    print('Fitting Model...')
    if model_type in ['LR','RF','SVC']:
        model.fit(train_inputs,train_outputs)
    elif model_type in ['Kmeans']:
        model.fit(train_inputs)
        
def evaluate_model(model,inputs,outputs,df,eval_type='instance',task='0 vs. 1. vs. 234'):
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
    cols_to_groupby = eval_type.split('-')
    
    if task in ['01 vs. 234','0']:
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
    

    downweight = 2096/(2096 + 9710) #for the validation set
    term1 = -downweight * np.dot(1-outputs,np.log(1-probs+1e-16))
    term2 = -np.dot(outputs,np.log(probs+1e-16))
    bce = (term1 + term2)/len(outputs)
    
    acc = accuracy_score(outputs,preds)     
    auc = roc_auc_score(outputs,probs,multi_class='ovr')
    prec = precision_score(outputs,preds,average='macro')
    rec = recall_score(outputs,preds,average='macro')
    
    probs_df['label'] = outputs
    probs_df.columns = ['instance','date','prob','label']
    
    return acc, auc, prec, rec, bce, probs_df

def evaluate_multi_as_binary(model,inputs,outputs,df,eval_type='instance'):
    preds = model.predict(inputs)
    preds[preds == 1] = 0
    preds[preds == 2] = 1
    outputs[outputs == 1] = 0
    outputs[outputs == 2] = 1
    
    probs = model.predict_proba(inputs)
    probs = probs[:,-1] #positive probability
    cols_to_groupby = eval_type.split('-')
        
    if eval_type in ['instance']:
        preds_df = pd.DataFrame(preds,columns=['preds'],index=df.index)
        preds_df[cols_to_groupby] = df[cols_to_groupby]
        preds_df = preds_df.groupby(by=cols_to_groupby).progress_apply(lambda x:np.argmax(np.histogram((x>0.5).astype(int),[0,1,2])[0])).reset_index() #specific to 3 class problem
        preds = preds_df.iloc[:,-1].values
    
        probs_df = pd.DataFrame(probs,index=df.index)
        probs_df[cols_to_groupby] = df[cols_to_groupby]
        probs_df = probs_df.groupby(by=cols_to_groupby).progress_apply(lambda x:x.mean())
        probs = probs_df.loc[:,0].values #specific to 3 class problem

        outputs_df = pd.DataFrame(outputs,index=df.index)
        outputs_df[cols_to_groupby] = df[cols_to_groupby] #order matters for this line 
        outputs_df = outputs_df.groupby(by=cols_to_groupby).mean().reset_index()
        outputs = outputs_df.iloc[:,-1]
    
    acc = accuracy_score(outputs,preds)     
    auc = roc_auc_score(outputs,probs,multi_class='ovr')
    
    return acc, auc

#%%

#""" STAGE 2 - Obtain Data Splits (Based on Patient ID) """
def split_into_train_val_test_sets(df):
    """ Split data into training, validation, and test sets based on patient ID
    
    Args:
        df (pd.DataFrame): dataframe which consists of all data (both inputs and outputs) to be split [ N x M ] N = Number of Samples, M = Columns with All Data
    
    Returns:
        dict_df (dict): dictionary contaning the splits and the corresponding matrices
                        E.g., dict_df['train'] contains the data matrix for training [ N1 x M ] where N1 < N
    """
    train_ratio, val_ratio = 0.70, 0.10
    patients = df['instance'].unique()
    npatients = len(patients)
    train_npatients, val_npatients = int(npatients*train_ratio), int(npatients*val_ratio)
    random.seed(0)
    shuffled_patients = random.sample(list(patients),npatients)
    phases = ['train','val','test']
    
    final_indices = list(np.cumsum([train_npatients,val_npatients])) + [npatients]
    
    dict_df = dict()
    start_index = 0
    for phase,final_index in zip(phases,final_indices):
        curr_patients = shuffled_patients[start_index:final_index]
        curr_df = df[df['instance'].isin(curr_patients)]
        
        if phase == 'train': #shuffle training instances!
            random.seed(0)
            indices = random.sample(list(range(curr_df.shape[0])),curr_df.shape[0])
            curr_df = curr_df.iloc[indices,:]
            
        dict_df[phase] = curr_df
        start_index = final_index
    
    """ Confirm that Patients do Not Overlap """
    assert set(dict_df['train']['instance']).intersection(set(dict_df['val']['instance'])) == set()
    
    return dict_df

def quantify_lab_missingness(df):
    """ Quantify Percentage of Patient Visits with Missing Variables 
    
        We can use this to:
            1) determine which lab variables to exclude entirely
            2) determine which lab variables we can impute (e.g., mean, median)
    
    """
    # cols = df.columns.tolist() #check this again
    # #final_col_index = np.where([col == 'Structured' for col in cols])[0][0]
    # first_col_index = np.where([col == 'erythropoietin' for col in cols])[0][0]
    # labs = cols[first_col_index:] #final_col_index]
    # df_missing = pd.Series(index=labs)
    # for lab in labs:
    #     missing_percentage = df[lab].isna().sum()/df.shape[0] * 100
    #     df_missing[lab] = missing_percentage
    
    # cols = df.columns.tolist() #check this again
    # index1 = np.where([col == 'date' for col in cols])[0].item() + 1
    # labs = cols[index1:]
    
    cols = df.columns
    curr_labs = [col for col in cols if col in labs]
    df_missing_col = pd.Series(index=curr_labs)
    for lab in curr_labs:
        missing_percentage = df[lab].isna().sum()/df.shape[0] * 100
        df_missing_col[lab] = missing_percentage
    return df_missing_col

def drop_labs(df,df_missing,missingness):
    missing_threshold = missingness
    labs_to_keep = df_missing[df_missing < missing_threshold].index.tolist()
    labs_to_remove = set(df_missing.index.tolist()) - set(labs_to_keep)
    df = df.drop(list(labs_to_remove),1,inplace=False)    
    return df 

def quantify_row_missingness(df):
    # cols = df.columns.tolist()
    # index1 = np.where([col == 'date' for col in cols])[0].item() + 1
    # ncols = len(cols) - index1 #- 3
    cols = df.columns
    curr_labs = [col for col in cols if col in labs]
    ncols = len(curr_labs)
    df_missing_row = df.loc[:,curr_labs].progress_apply(lambda x: (x.isna().sum()/ncols) * 100,axis=1)
    return df_missing_row

def drop_rows(df,df_missing_row,missingness_row):
    drop_indices = df_missing_row[df_missing_row > missingness_row].index
    df = df.drop(drop_indices,0,inplace=False)
    return df 

#%%

def isNaN(el):
    return el != el

def load_lab_data():
    """ Load Labs Data """
    df2 = pd.read_csv('instance-all.csv')
    return df2



