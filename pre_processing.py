#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:01:09 2020

@author: a.gogohia
"""

import os
import numpy as np
from sklearn.datasets import (
    load_boston,
    load_wine,
    load_breast_cancer
    )

os.chdir('/Users/a.gogohia/Documents/GitHub/data_type_agnostic_imputation/data')

def load_data(data_import, return_X_y=False):
    if return_X_y is True:
        X, y = data_import(return_X_y=True)
        return X, y
    else:
        X = data_import(return_X_y=False)
        return X

def convert_to_strings(dataset):
    X = dataset.data
    y = dataset.target
    X_names = ['<{}>'.format(w) for w in dataset.feature_names] # get column names
    X_str = np.vectorize(str)(X) # convert all elements to strings
    f = lambda line: " ".join([ch for ch in line]) # function to separate each character in string
    X_str = np.vectorize(f)(X_str) # apply separator function to each element in array
    y_str = np.vectorize(str)(y) # convert all elements to strings
    y_str = np.vectorize(f)(y_str) # apply separator function to each element in array
    y_str = y_str.tolist()
    # join column name with repective cell
    whole_list=[]
    for i in range(X_str.shape[0]):
        sublist=[]
        for j in range(len(X_names)):
            sublist.append(X_names[j] + ' ' + X_str[i,j])
        whole_list.append(' '.join(sublist))
    return whole_list, y_str

def create_source_factors(dataset):
    X = dataset.data
    # get names of data columns
    X_names = ['<{}>'.format(w) for w in dataset.feature_names]
    # create dataset with column names
    X_named = np.vstack((X_names,X))
    source_factors = []
    for i in range(1,X_named.shape[0]):
        placeholder = []
        for j in range(X_named.shape[1]):
            for ch in X_named[i,j]:
                placeholder.append(str(X_named[0,j]))
            placeholder.append(str(X_named[0,j])) # add one additional sf
        source_factors.append(' '.join(placeholder))
    return source_factors

def shuffle_data(source, target, source_factors):
    c = list(zip(source, target, source_factors))
    np.random.shuffle(c)    
    source, target, source_factors = zip(*c)
    return source, target, source_factors

def create_files(source, target, source_factors, file_dir, num_dev=0.1):
    
    num_samples = len(source)
    num_dev = round(num_samples*num_dev)
    
    # split training and validation data
    train_samples = source[:num_samples-num_dev] # training source data
    dev_samples = source[num_samples-num_dev:] # validation source data
    
    train_target = target[:num_samples-num_dev] # training target data
    dev_target = target[num_samples-num_dev:] # validation target data
    
    train_source_factors = source_factors[:num_samples-num_dev] # training source factors
    dev_source_factors = source_factors[num_samples-num_dev:] # validation target factors
    
    
    file_dir=str(file_dir)
    # write files
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
            
    with open("{}/train.source".format(file_dir), "w") as source1:
        for sample in train_samples:
            source1.write(sample + "\n")
    
    with open("{}/train.target".format(file_dir), "w") as target1:
        for sample in train_target:
            target1.write(sample + "\n")
            
    with open("{}/dev.source".format(file_dir), "w") as source1:
        for sample in dev_samples:
            source1.write(sample + "\n")
    
    with open("{}/dev.target".format(file_dir), "w") as target1:
        for sample in dev_target:
            target1.write(sample + "\n")
            
    with open("{}/train.source_factors".format(file_dir), "w") as source_f1:
        for sample in train_source_factors:
            source_f1.write(sample + "\n")
            
    with open("{}/dev.source_factors".format(file_dir), "w") as source_f1:
        for sample in dev_source_factors:
            source_f1.write(sample + "\n")

dataset_dict = {'boston_data': load_boston, 
                'wine_data': load_wine,
                'breast_cancer_data': load_breast_cancer,
                }

for name, loader in dataset_dict.items():
    dataset = load_data(loader, False)
    print(name)
    source_factors = create_source_factors(dataset)
    source, target = convert_to_strings(dataset)
    
    source, target, source_factors = shuffle_data(source, target, source_factors)
    
    num_dev=30
        
    create_files(source, target, source_factors, name)

