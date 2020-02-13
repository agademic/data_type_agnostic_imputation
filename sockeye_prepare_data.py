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
    try:
        types = os.listdir(str(dataset))
        print(types)
        for typ in types:
            try:
                command_prepr = 'python -m sockeye.prepare_data -s {name}/train.source -t {name}/train.target -sf {name}/train.source_factors -o {name}/prepared_sockeye_data'.format(name=typ)
                print(command_prepr)
                process = subprocess.Popen(command_prepr.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
            except:
                continue
    except:
        continue