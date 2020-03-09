#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:49:26 2020

@author: a.gogohia
"""

import os
from sklearn.metrics import accuracy_score
from mse_calc import (
    open_list,
    mse)

os.chdir('data')

categorical = ['german_credit_data', 'wine_data', 'blood_transfusion_data']
numerical = ['linear', 'quadratic', 'boston_data']
mseDict = {}
for dataset in numerical:
    print(dataset)
    try:
        types = os.listdir(str(dataset))
        for typ in types:
            print(typ)
            if not typ.startswith('.'): # ignore hidden folders
                missingness = os.listdir(str(dataset) + '/' + str(typ))
                for miss in missingness:
                    print(miss)
                    try:
                        targetList = open_list(f'{dataset}/{typ}/{miss}/test.target')
                        predList = open_list(f'{dataset}/{typ}/{miss}/translated_data')
                        test = mse(targetList, predList)
                        print(test)
                        mseDict.update({
                            f'{dataset}/{typ}/{miss}': test})
                    except:
                        continue
    except:
        continue
    
accDict = {}
for dataset in categorical:
    print(dataset)
    try:
        types = os.listdir(str(dataset))
        for typ in types:
            print(typ)
            if not typ.startswith('.'): # ignore hidden folders
                missingness = os.listdir(str(dataset) + '/' + str(typ))
                for miss in missingness:
                    print(miss)
                    try:
                        targetList = open_list(f'{dataset}/{typ}/{miss}/test.target')
                        predList = open_list(f'{dataset}/{typ}/{miss}/translated_data')
                        test = accuracy_score(targetList, predList)
                        print(test)
                        accDict.update({
                            f'{dataset}/{typ}/{miss}': test})
                    except:
                        continue
    except:
        continue