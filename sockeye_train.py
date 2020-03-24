#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:56:44 2020

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
                        command_prepr = f'python -m sockeye.train -d {dataset}/{typ}/{miss}/prepared_sockeye_data/ -vs {dataset}/{typ}/{miss}/dev.source -vt {dataset}/{typ}/{miss}/dev.target -vsf {dataset}/{typ}/{miss}/dev.source_factors --source-factors-num-embed 16 --transformer-feed-forward-num-hidden 128 --optimized-metric accuracy --metrics accuracy --batch-type sentence --batch-size 64 --max-num-checkpoint-not-improved 2 -o {dataset}/{typ}/{miss}//sockeye_model_xxl'
                        print(command_prepr)
                        process = subprocess.Popen(command_prepr.split(), stdout=subprocess.PIPE)
                        output, error = process.communicate()
                    except:
                        continue
    except:
        continue
    
# model does not overwrite output, add --overwrite-output
        #f'python -m sockeye.train -d {dataset}/{typ}/{miss}/prepared_sockeye_data/ -vs {dataset}/{typ}/{miss}/dev.source -vt {dataset}/{typ}/{miss}/dev.target -vsf {dataset}/{typ}/{miss}/dev.source_factors --num-embed 32 --source-factors-num-embed 16 --transformer-model-size 32 --transformer-feed-forward-num-hidden 16 --num-layers 4 --metrics perplexity accuracy --use-cpu --batch-type sentence --max-num-checkpoint-not-improved 3 --batch-size 4 -o {dataset}/{typ}/{miss}//sockeye_model_large'
