#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:40:10 2019

@author: a.gogohia
"""

import os
os.chdir('/Users/a.gogohia/OneDrive/Dokumente/Master Data Science/Master Thesis/Master Program')
import numpy as np
import pandas as pd

def create_data(num_samples = 10000,
                num_dev = 1000,
                target_type='quadratic',
                c_1=2,
                c_2=3,
                file_dir='data'):
    """
    Function to create synthetic data. This function generates a list with randomly
    generated numerical values (normal distributed), another list with quadratic values from the first 
    list and source factors (labels) for the first list.
    
    Input: 
        n: number of values to be generated
        target_type: 'quadratic' for quadratic targets to be generated or 
        'linear' for linear target values
    """
    if target_type == 'quadratic':
        
        # generate source values
        x = abs(np.random.normal(0, 2, num_samples))
        # generate target values
        y = x*x+np.random.normal(0, 1)*0.01
            
    else:
            
        # generate source values
        x = np.random.normal(0, 2, num_samples)
        c_1 = c_1
        c_2 = c_2
        y = x*c_1+c_2+np.random.normal(0, 1)*0.01
        
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
    source_factor = []
    for line in df1:
        elem = ' '.join(['<source>'] * (len(line)+1))
        source_factor.append(elem)
    
    # split training and validation data
    train_samples = source[:num_samples-num_dev] # training source data
    dev_samples = source[num_samples-num_dev:] # validation source data
    
    train_target = target[:num_samples-num_dev] # training target data
    dev_target = target[num_samples-num_dev:] # validation target data
    
    train_source_factors = source_factor[:num_samples-num_dev] # training source factors
    dev_source_factors = source_factor[num_samples-num_dev:] # validation target factors
    
    # write files
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
            
    with open("{}/train.source".format(file_dir), "w") as source:
        for sample in train_samples:
            source.write(sample + "\n")
    
    with open("{}/train.target".format(file_dir), "w") as target:
        for sample in train_target:
            target.write(sample + "\n")
            
    with open("{}/dev.source".format(file_dir), "w") as source:
        for sample in dev_samples:
            source.write(sample + "\n")
    
    with open("{}/dev.target".format(file_dir), "w") as target:
        for sample in dev_target:
            target.write(sample + "\n")
            
    with open("{}/train.source_factors".format(file_dir), "w") as source_f:
        for sample in train_source_factors:
            source_f.write(sample + "\n")
            
    with open("{}/dev.source_factors".format(file_dir), "w") as source_f:
        for sample in dev_source_factors:
            source_f.write(sample + "\n")

    #return source, target, source_factor



create_data(num_samples=10000,num_dev=1000, target_type='linear', file_dir='data_linear/data')
