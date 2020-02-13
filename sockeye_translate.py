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
    try:
        types = os.listdir(str(dataset))
        print(types)
        for typ in types:
            try:
                command_prepr = 'python -m sockeye.translate -m {name}/sockeye_model_large -i {name}/test.source -if {name}/test.source_factors --use-cpu -o {name}/translated_data'.format(name=typ)
                print(command_prepr)
                process = subprocess.Popen(command_prepr.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
            except:
                continue
    except:
        continue