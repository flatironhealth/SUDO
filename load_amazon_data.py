#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 17:12:04 2021

@author: dani.kiyasseh

This script is used to:
1) load the Multi-Domain Sentiment dataset for feature extraction
"""

# [USER MUST MODIFY] path_to_amazon_data
path_to_amazon_data = '/Users/dani.kiyasseh/Desktop/Data/processed_acl'

import os
from tqdm import tqdm 
import random
from operator import itemgetter
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer

def load_amazon_reviews():
    path = path_to_amazon_data
    reviews = dict()
    for domain in ['books','dvd','electronics','kitchen']:
        reviews[domain] = dict()
        
        for label in ['negative','positive']:
            with open(os.path.join(path,domain,'%s.review' % label),'r') as f:
                data = []
                for line in f:
                    data.append(line.strip())
        
            curr_reviews = []
            for review in data:
                fts_and_cnts = review.split(' ')[:-1] #last entry is label
                fts = list(map(lambda el:el.split(':')[0],fts_and_cnts))
                cnts = list(map(lambda el:el.split(':')[1],fts_and_cnts))
                curr_reviews.append(fts)
                
            reviews[domain][label] = curr_reviews
    return reviews

#%%
def split_reviews_into_phases(reviews):
    random.seed(0)
    indices = random.sample(list(range(1000)),1000) #1000 is number of reviews per category
    train_frac, val_frac = 0.6, 0.2
    ntrain, nval = int(train_frac*len(indices)), int(val_frac*len(indices))
    train_indices, val_indices, test_indices = indices[:ntrain], indices[ntrain:ntrain+nval], indices[ntrain+nval:]
    
    phase_reviews = dict()
    phase_labels = dict()
    for domain in reviews.keys():
        phase_reviews[domain] = defaultdict(list)
        phase_labels[domain] = defaultdict(list)
        for phase,indices in zip(['train','val','test'],[train_indices,val_indices,test_indices]):
            for label in reviews[domain].keys():
                curr_reviews = list(itemgetter(*indices)(reviews[domain][label]))
                phase_reviews[domain][phase].extend(curr_reviews)
                phase_labels[domain][phase].extend([label]*len(curr_reviews))
    return phase_reviews, phase_labels

def retrieve_vocabs(phase_reviews,nfeatures):
    domain_vocabs = dict()
    for domain in phase_reviews.keys():
        all_tokens = np.concatenate(phase_reviews[domain]['train'])
        top_tokens_and_cnts = sorted(Counter(all_tokens).items(),key=lambda el:el[1])[-nfeatures:]
        tokens = list(map(lambda entry:entry[0],top_tokens_and_cnts))
        domain_vocabs[domain] = tokens
    return domain_vocabs

def get_vectorizers(domain_vocabs):
    vecs_dict = dict()
    for domain in domain_vocabs.keys():
        vecs_dict[domain] = CountVectorizer(vocabulary=domain_vocabs[domain])
    return vecs_dict
    
#%%

def get_data(phase_reviews,vec,domain):
    data = dict()
    data[domain] = dict()
    for phase in phase_reviews[domain].keys():
        reviews = list(map(lambda review:' '.join(review),phase_reviews[domain][phase]))
        features = vec.transform(reviews).toarray()
        data[domain][phase] = features
    return data



