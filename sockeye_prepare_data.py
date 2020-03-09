#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:45:08 2020

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
            missingness = os.listdir(str(dataset) + '/' + str(typ))
            for miss in missingness:
                print(miss)
                try:
                    command_prepr = f'python -m sockeye.prepare_data -s {dataset}/{typ}/{miss}/train.source -t {dataset}/{typ}/{miss}/train.target -sf {dataset}/{typ}/{miss}/train.source_factors -o {dataset}/{typ}/{miss}/prepared_sockeye_data'
                    print(command_prepr)
                    process = subprocess.Popen(command_prepr.split(), stdout=subprocess.PIPE)
                    output, error = process.communicate()
                except:
                    continue
    except:
        continue