#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:56:44 2020

@author: a.gogohia
"""

import os
import subprocess

os.chdir('/Users/a.gogohia/Documents/GitHub/data_type_agnostic_imputation/data')
datasets = os.listdir('/Users/a.gogohia/Documents/GitHub/data_type_agnostic_imputation/data')

for dataset in datasets:
    try:
        command_prepr = 'python -m sockeye.train -d {name}/prepared_sockeye_data/ -vs {name}/dev.source -vt {name}/dev.target -vsf {name}/dev.source_factors --num-embed 32 --source-factors-num-embed 16 --transformer-model-size 32 --transformer-feed-forward-num-hidden 16 --num-layers 4 --metrics perplexity accuracy --use-cpu --batch-type sentence --max-num-checkpoint-not-improved 3 --batch-size 4 -o {name}/sockeye_model_large'.format(name=dataset)
        print(command_prepr)
        process = subprocess.Popen(command_prepr.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
    except:
        continue
    
# model does not overwrite output, add --overwrite-output