#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:11:29 2020

@author: a.gogohia
"""

from pathlib import Path
import numpy as np

#import numpy as np
#import pandas as pd
#import sklearn
from sklearn.datasets import (
    load_boston,
    load_wine,
    )

from pre_processing import (
    load_data,
    generate_missing_mask,
    generate_missingness,
    convert_to_strings,
    create_source_factors,
    shuffle_data,
    create_numerical_data,
    create_files)
from sklearn_imputation import sklearn_impute

"""
Top main file for setting the parameters for the experiment.
1. Define and load datasets for training.
    a. For synthetical data define:
        i. Type of data (quadratic or linear).
        ii. Amount of data.
        iii. Noise in target data.
2. Define missingness mask (type and percentage of missingness) for datasets.
3. Apply missingness mask (generate_missingness) to datasets and create training
and test set.
    a. Set target column for training and testing.
4. Generate source factors for datasets.
5. Convert dataset values to strings (convert_to_strings).
6. Create files for training, validation and testing (create_files(set_type="train" or "test")).
7. Run model data preparation and training file.

"""
np.random.seed(8)

# dataset = load_data(load_boston)
# dataset = load_boston()
# mask = generate_missing_mask(dataset)
# t, t1 = generate_missingness(dataset, mask)
# dataset = create_numerical_data()
# mask = generate_missing_mask(dataset)

# dataset = load_data('blood-transfusion-service-center')
# mask = generate_missing_mask(dataset)
# t, t1 = generate_missingness(dataset, mask)

def data_setup():
    imputation_scores = []
    # load real world datassets
    for name, loader in dataset_dict.items():
        dataset = load_data(loader, False)
        print(name)
        for miss_type in missingness['missingness_type']:
            print(miss_type)
            for miss_percent in missingness['missingness_percent']:
                print(miss_percent)
                mask = generate_missing_mask(dataset, percent_missing=miss_percent, missingness=miss_type)
                training_set, test_set = generate_missingness(dataset, mask)
                # write training set
                source_factors = create_source_factors(training_set)
                source, target = convert_to_strings(training_set)
                source, target, source_factors = shuffle_data(source, target, source_factors)
                #file_name = str(name + '/' + miss_type)
                file_name = Path('data/' + f'{name}/{miss_type}/{miss_percent}')
                create_files(source, target, source_factors, file_dir=file_name, set_type='train')
                # write testing set
                source_factors = create_source_factors(test_set)
                source, target = convert_to_strings(test_set)
                create_files(source, target, source_factors, file_dir=file_name, set_type='test')
                # compute mse using imputation methods and save file
                scores = [f'{name}, {miss_type}, {miss_percent}', sklearn_impute(training_set, test_set, name=name)]
                imputation_scores.append(scores)
                #imputation_scores.to_pickle(Path(file_name.__str__() + 'imputation_score.pkl'))
                print(imputation_scores)
    # create quadratic and linear numerical values with different noise factors
    for target_type in parameters_syth_data['type']:
        print(target_type)
        dataset = create_numerical_data(target_type=target_type)
        for miss_type in missingness['missingness_type']:
            print(miss_type)
            for miss_percent in missingness['missingness_percent']:
                print(miss_percent)
                mask = generate_missing_mask(dataset, percent_missing=miss_percent, missingness=miss_type)
                training_set, test_set = generate_missingness(dataset, mask)
                # write training set
                source_factors = create_source_factors(training_set)
                source, target = convert_to_strings(training_set)
                source, target, source_factors = shuffle_data(source, target, source_factors)
                #file_name = str(target_type + '/' + miss_type)
                file_name = Path('data/' + f'{target_type}/{miss_type}/{miss_percent}')
                create_files(source, target, source_factors, file_dir=file_name, set_type='train')
                # write testing set
                source_factors = create_source_factors(test_set)
                source, target = convert_to_strings(test_set)
                create_files(source, target, source_factors, file_dir=file_name, set_type='test')
                # compute mse using imputation methods and save file          
                scores = [f'{target_type}, {miss_type}, {miss_percent}', sklearn_impute(training_set, test_set, name=name)]
                imputation_scores.append(scores)
                #imputation_scores.to_pickle(Path(file_name.__str__() + 'imputation_score.pkl'))
                print(imputation_scores)
    
    with open('scores.txt', 'w') as f:
        for item in scores:
            f.write("%s\n" % item)

dataset_dict = {
    'boston_data': load_boston, 
    'wine_data': load_wine,
    'blood_transfusion_data': 'blood-transfusion-service-center',
    'german_credit_data': 'credit-g'
                }

parameters_syth_data = {
    'type': ['quadratic', 'linear'],
    'amount': [100, 1000, 10000],
    'noise': [0.001, 0.01, 0.1, 1]
    }

missingness = {
    'missingness_type': ['MCAR', 'MAR', 'MNAR'],
    'missingness_percent': [10, 30]}

data_setup()
# noise_list = [0.001, 0.01, 0.1, 1]
# for noise in noise_list:
#     source, target, source_factors = create_numerical_data(noise=noise)
#     create_files(source, target, source_factors, 'quadratic_with_noise_{noise}'.format(noise=noise))

# for noise in noise_list:
#     source, target, source_factors = create_numerical_data(target_type='linear', noise=noise)
#     create_files(source, target, source_factors, 'linear_with_noise_{noise}'.format(noise=noise))
