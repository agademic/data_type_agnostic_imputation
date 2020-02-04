#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:45:08 2020

@author: a.gogohia
"""

import os
import subprocess

os.chdir('/Users/a.gogohia/Documents/GitHub/data_type_agnostic_imputation/data')
datasets = os.listdir('/Users/a.gogohia/Documents/GitHub/data_type_agnostic_imputation/data')

for dataset in datasets:
    try:
        command_prepr = 'python -m sockeye.prepare_data -s {name}/train.source -t {name}/train.target -sf {name}/train.source_factors -o {name}/prepared_sockeye_data'.format(name=dataset)
        print(command_prepr)
        process = subprocess.Popen(command_prepr.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
    except:
        continue