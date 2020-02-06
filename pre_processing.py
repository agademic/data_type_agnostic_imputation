#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:01:09 2020

@author: a.gogohia
"""

import os
import numpy as np
import sklearn
import pandas as pd
from sklearn.datasets import (
    load_boston,
    load_wine,
    )

os.chdir('/Users/a.gogohia/Documents/GitHub/data_type_agnostic_imputation/data')

def load_data(data_import, return_X_y=False):
    if return_X_y is True:
        try:
            X, y = data_import(return_X_y=True)

        except:
            X, y = sklearn.datasets.fetch_openml(str(data_import), return_X_y=True)
        return X, y
    else:
        try:
            X = data_import(return_X_y=False)
        except:
            X = sklearn.datasets.fetch_openml(str(data_import), return_X_y=False)
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

def create_numerical_data(num_samples = 10000,
                num_dev = 0.1,
                target_type='quadratic',
                c_1=2,
                c_2=3,
                noise=0.01,
                file_dir='data'):
    """
    Function to create synthetic data. This function generates a list with randomly
    generated numerical values (normal distributed), another list with quadratic values from the first 
    list and source factors (labels) for the first list.
    
    Input: 
        num_samples: int, total number of values to be generated
        num_dev: float, amount of validation data
        target_type: string, 'quadratic' for quadratic targets to be generated or 
        'linear' for linear target values
    """
    if target_type == 'quadratic':
        
        # generate source values
        x = abs(np.random.normal(0, 2, num_samples))
        # generate target values
        y = x*x+np.random.normal(0, 1)*noise
            
    else:
            
        # generate source values
        x = np.random.normal(0, 2, num_samples)
        c_1 = c_1
        c_2 = c_2
        y = x*c_1+c_2+np.random.normal(0, 1)*noise
        
    # create datafame 
    df = pd.DataFrame({'col1': '<source>', 'col2': x, 'col3': y})
    # round floats to 8 decimals and format to strings, keep trailing zeros
    df1 = df['col2'].apply(lambda x: '{:.8f}'.format(round(x, 8)))
    # split tokens by character and combine with source factor
    df2 = df1.apply(lambda line: " ".join([ch for ch in line]))
    df3 = df['col1'] + ' ' + df2
    source = df3.tolist()
    
    # split target tokens by character
    target = df['col3'].apply(lambda x: '{:.8f}'.format(round(x, 8)))
    target = target.apply(lambda line: " ".join([ch for ch in line]))
    target = target.tolist()
    
    # create list with source factors
    source_factors = []
    for line in df1:
        elem = ' '.join(['<source>'] * (len(line)+1))
        source_factors.append(elem)
        
    return source, target, source_factors

def create_files(source, target, source_factors, file_dir, num_dev=0.1):
    """
    Function to create input files for the sockeye model. It takes the datasets
    as input and creates distinct training and test files for source, target 
    and source factors.

    Parameters
    ----------
    source : list
        List with source variables.
    target : list
        List with .
    source_factors : list
        DESCRIPTION.
    file_dir : str, file_path
        Directory path where files should be saved.
    num_dev : float, optional
        Amount of validation data to be created. The default is 0.1.

    Returns
    -------
    Creates files for source, target and source factor variables in given directory.

    """
    
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
                'blood_transfusion_data': 'blood-transfusion-service-center',
                'german_credit_data': 'credit-g'
                }

for name, loader in dataset_dict.items():
    dataset = load_data(loader, False)
    print(name)
    source_factors = create_source_factors(dataset)
    source, target = convert_to_strings(dataset)
    
    source, target, source_factors = shuffle_data(source, target, source_factors)
        
    create_files(source, target, source_factors, name)

# create quadratic and linear numerical values with different noise factors
noise_list = [0.001, 0.01, 0.1, 1]
for noise in noise_list:
    source, target, source_factors = create_numerical_data(noise=noise)
    create_files(source, target, source_factors, 'quadratic_with_noise_{noise}'.format(noise=noise))

for noise in noise_list:
    source, target, source_factors = create_numerical_data(target_type='linear', noise=noise)
    create_files(source, target, source_factors, 'linear_with_noise_{noise}'.format(noise=noise))
