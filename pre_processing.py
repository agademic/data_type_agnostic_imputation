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
    
def generate_missing_mask(dataset, percent_missing=10, missingness='MCAR'):
    arr = dataset.data
    arr = np.concatenate([arr, np.reshape(dataset.target, (dataset.target.shape[0], 1))], axis=1)
    X = arr
    if missingness=='MCAR':
        # missing completely at random
        mask = np.random.rand(*X.shape) < percent_missing / 100.
    elif missingness=='MAR':
        # missing at random, missingness is conditioned on a random other column
        # this case could contain MNAR cases, when the percentile in the other column is 
        # computed including values that are to be masked
        mask = np.zeros(X.shape)
        n_values_to_discard = int((percent_missing / 100) * X.shape[0])
        # for each affected column
        for col_affected in range(X.shape[1]):
            # select a random other column for missingness to depend on
            depends_on_col = np.random.choice([c for c in range(X.shape[1]) if c != col_affected])
            # pick a random percentile of values in other column
            if n_values_to_discard < X.shape[0]:
                discard_lower_start = np.random.randint(0, X.shape[0]-n_values_to_discard)
            else:
                discard_lower_start = 0
            discard_idx = range(discard_lower_start, discard_lower_start + n_values_to_discard)
            values_to_discard = X[:,depends_on_col].argsort()[discard_idx]
            mask[values_to_discard, col_affected] = 1
    elif missingness == 'MNAR':
        # missing not at random, missingness of one column depends on unobserved values in this column
        mask = np.zeros(X.shape)
        n_values_to_discard = int((percent_missing / 100) * X.shape[0])
        # for each affected column
        for col_affected in range(X.shape[1]):
            # pick a random percentile of values in other column
            if n_values_to_discard < X.shape[0]:
                discard_lower_start = np.random.randint(0, X.shape[0]-n_values_to_discard)
            else:
                discard_lower_start = 0
            discard_idx = range(discard_lower_start, discard_lower_start + n_values_to_discard)
            values_to_discard = X[:,col_affected].argsort()[discard_idx]
            mask[values_to_discard, col_affected] = 1
    return pd.DataFrame(mask) > 0

def generate_missingness(dataset, mask, target_column=-1, nan_value='N'):
    """
    Function to create missingness in a specific dataset with a given missing mask.
    Returns a source and a test dataset as bunch objects. The target column can
    be specified.

    Parameters
    ----------
    dataset : sklearn.utils.bunch object
        Dataset to be filled with missing values.
    mask : pandas.DataFrame object
        Missing values mask, Boolean.
    target_column : int, optional
        Column which should be treated as target. The default is -1.
    nan_value : str, optional
        Value which should replace NaN values. The default is 'N'.

    Returns
    -------
    dataset_source : sklearn.utils.bunch object
        Dataset bunch object for further preprocessing, used for training.
    dataset_test : sklearn.utils.bunch object
        Dataset bunch object for further preprocessing, used for testing.

    """
    # convert to array to merge source and target values
    arr = dataset.data
    arr = np.concatenate([arr, np.reshape(dataset.target, (dataset.target.shape[0], 1))], axis=1)
    # convert to DataFrame to easily select mask
    df = pd.DataFrame(arr)
    # convert to integers if type categorical
    #df.iloc[:, target_column] = pd.to_numeric(df.iloc[:,target_column])
    df_nans = df[~mask]
    #df_source = df_nans[np.isfinite(df_nans.iloc[:,target_column])]
    df_source = df_nans[~df_nans.iloc[:,target_column].isnull()]
    #df_source.fillna(nan_value, inplace=True)
    df_test = df_nans.drop(df_source.index)
    #df_test.fillna(nan_value, inplace=True)
    # create bunch objects
    shape = dataset.data.shape
    df_dropped = df_source.drop(df_source.columns[target_column], axis=1)
    # need to reshape data to (n, 1) in case there is only one column
    dataset_source = sklearn.utils.Bunch(data=np.array(df_dropped).reshape(-1, shape[1]),
                                         feature_names=dataset.feature_names,
                                         target=np.array(df_source.iloc[:,target_column]))
    df_dropped = df_test.drop(df_test.columns[target_column], axis=1)
    dataset_test = sklearn.utils.Bunch(data=np.array(df_dropped).reshape(-1, shape[1]).reshape(-1, shape[1]),
                                       feature_names=dataset.feature_names,
                                       target=np.array(df.iloc[df_test.index, target_column]))

    return dataset_source, dataset_test

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
            for ch in str(X_named[i,j]):
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
        x = np.reshape(x, (x.shape[0],1))
        # generate target values
        y = x*x+np.random.normal(0, 1)*noise
        y = np.reshape(y, (y.shape[0],1))
            
    else:
            
        # generate source values
        x = np.random.normal(0, 2, num_samples)
        x = np.reshape(x, (x.shape[0],1))
        c_1 = c_1
        c_2 = c_2
        y = x*c_1+c_2+np.random.normal(0, 1)*noise
        y = np.reshape(y, (y.shape[0],1))
    
    dataset = sklearn.utils.Bunch(data=x,feature_names=np.array(['source']), target=y)
    
    # # create datafame 
    # df = pd.DataFrame({'col1': '<source>', 'col2': x, 'col3': y})
    # # round floats to 8 decimals and format to strings, keep trailing zeros
    # df1 = df['col2'].apply(lambda x: '{:.8f}'.format(round(x, 8)))
    # # split tokens by character and combine with source factor
    # df2 = df1.apply(lambda line: " ".join([ch for ch in line]))
    # df3 = df['col1'] + ' ' + df2
    # source = df3.tolist()
    
    # # split target tokens by character
    # target = df['col3'].apply(lambda x: '{:.8f}'.format(round(x, 8)))
    # target = target.apply(lambda line: " ".join([ch for ch in line]))
    # target = target.tolist()
    
    # # create list with source factors
    # source_factors = []
    # for line in df1:
    #     elem = ' '.join(['<source>'] * (len(line)+1))
    #     source_factors.append(elem)
        
    return dataset

def create_files(source, target, source_factors, file_dir, num_dev=0.1, set_type='train'):
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
    
    if set_type == 'train':
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
            os.makedirs(file_dir)
                
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
                
    elif set_type == 'test':
        test_samples = source
        test_target = target
        test_source_factors = source_factors
        
        file_dir = str(file_dir)
        # write files
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
            
        with open("{}/test.source".format(file_dir), "w") as source1:
            for sample in test_samples:
                source1.write(sample + "\n")
                
        with open("{}/test.target".format(file_dir), "w") as target1:
            for sample in test_target:
                target1.write(sample + "\n")
                
        with open("{}/test.source_factors".format(file_dir), "w") as source_f1:
            for sample in test_source_factors:
                source_f1.write(sample + "\n")
        
    else:
        print('Please specify the dataset type: "train" or "test".')

