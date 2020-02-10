#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:11:29 2020

@author: a.gogohia
"""

from pre_processing import *

dataset_dict = {'boston_data': load_boston, 
                'wine_data': load_wine,
                'blood_transfusion_data': 'blood-transfusion-service-center',
                'german_credit_data': 'credit-g'
                }

for name, loader in dataset_dict.items():
    dataset = load_data(loader, False)
    print(name)
    source_factors = create_source_factors(dataset)
    source, target = convert_to_strings(dataset)
    
    source, target, source_factors = shuffle_data(source, target, source_factors)
        
    create_files(source, target, source_factors, name)

# create quadratic and linear numerical values with different noise factors
noise_list = [0.001, 0.01, 0.1, 1]
for noise in noise_list:
    source, target, source_factors = create_numerical_data(noise=noise)
    create_files(source, target, source_factors, 'quadratic_with_noise_{noise}'.format(noise=noise))

for noise in noise_list:
    source, target, source_factors = create_numerical_data(target_type='linear', noise=noise)
    create_files(source, target, source_factors, 'linear_with_noise_{noise}'.format(noise=noise))
