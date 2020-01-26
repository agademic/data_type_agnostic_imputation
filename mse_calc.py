#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:54:29 2019

@author: a.gogohia
"""

import os
import numpy as np

def open_list(filename):
    """
    Function, that opens a file as a list and converts its string formatted
    values to numerical values.
    
    Parameters
    ----------
    filename : str, file path
        Directory path with list to be translated.

    Returns
    -------
    translated_list : list
        List with translated numerical values.
        
    """
    
    opened_list = [line.rstrip('\n') for line in open(filename)]
    
    t1 = []
    for i in range(len(opened_list)):
        t1.append(opened_list[i].replace('<source>', ''))
        
    translated_list = []
    for i in range(len(opened_list)):
        translated_list.append(float(t1[i].replace(' ', '')))
        
    return translated_list

def mse(targetlist, predlist):
    """
    Function to calculate MSE between values stored in a list.

    Parameters
    ----------
    targetlist : list
        List with true target values.
    predlist : list
        List with predicted target values.

    Returns
    -------
    MSE : Mean Squared Error

    """
    MSE = ((np.asarray(targetlist)-np.asarray(predlist)) ** 2).mean()
    # print("The Mean Square Error is: " , MSE)
    return MSE

def progress_mse(directory, target):
    """
    Function to calculate the MSE between the predicted output from sockeye model
    for each checkpoint and the true target value.

    Parameters
    ----------
    directory : str, file path
        Directory path with saved sockeye model outputs.
    target : list
        List with true target values.

    Returns
    -------
    mse_list : list
        List with MSE values for each model checkpoint.

    """
    mse_list = []
    for filename in sorted(os.listdir(directory)): # iterate through all files in folder // The method listdir() returns a list containing the names of the entries in the directory given by path.
        if not filename.startswith('decode.output'): # skip unwanted files
            continue
        path = directory + '/' + filename # create full path to open the files
        doc = open_list(path) # load and clean doc
        #print(doc[0])
        MSE = mse(doc, target)
        print("The Mean Square Error for {} is: ".format(filename) , MSE)
        mse_list.append(MSE)
    return mse_list

target = open_list('/Users/a.gogohia/sockeye/data_quadratic/data/sockeye_model_large2/decode.target')
directory = '/Users/a.gogohia/sockeye/data_quadratic/data/sockeye_model_large2'

mse_list = progress_mse(directory, target)