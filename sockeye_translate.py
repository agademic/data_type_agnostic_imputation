#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:53:19 2020

@author: a.gogohia
"""


import os
import subprocess

os.chdir('data')
datasets = os.listdir()

for dataset in datasets:
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
                        command_prepr = f'python -m sockeye.translate -m {dataset}/{typ}/{miss}/sockeye_model_xxl -i {dataset}/{typ}/{miss}/test.source -if {dataset}/{typ}/{miss}/test.source_factors --use-cpu -o {dataset}/{typ}/{miss}/translated_data_xxl'
                        print(command_prepr)
                        process = subprocess.Popen(command_prepr.split(), stdout=subprocess.PIPE)
                        output, error = process.communicate()
                    except:
                        continue
    except:
        continue
