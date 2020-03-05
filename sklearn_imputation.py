#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:25:19 2020

@author: a.gogohia
"""
import numpy as np
import pandas as pd

from mse_calc import mse

from sklearn.datasets import (
    load_boston,
    load_wine,
    )

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import BayesianRidge

from pre_processing import (
    load_data,
    generate_missing_mask,
    generate_missingness,
    convert_to_strings,
    create_source_factors,
    shuffle_data,
    create_numerical_data,
    create_files)

def sklearn_impute(training_set, test_set, name='dataset_name'):
    """
    Function to impute missing values in a specified column in a given target
    data set. Function takes a training set with no missing values in specififed
    column to fit the imputation methods. Afterwards the method predicts the
    values in the target column on the target data set.
    
    The function returns the calculated mse value for applied imputation methods.

    Parameters
    ----------
    training_set : sklearn.utils.bunch object
        training set to fit imputation method, no missing values in target column.
    test_set : sklearn.utils.bunch object
        target set to impute missing values in specified target column, target
        column has only missing values.
    estimator : sklearn regressor model
        Chosen regression model to impute missing values.

    Returns
    -------
    float
        Mean squared error value for predicted column.

    """
    estimators = [
    #KNeighborsRegressor(n_neighbors=15),
    RandomForestRegressor(random_state=0)
    ]
    
    
    training_set = np.concatenate([training_set.data, training_set.target.reshape(-1,1)], axis=1)
    test_set = np.concatenate([test_set.data, test_set.target.reshape(-1,1)], axis=1)
    test_set_labels = test_set[:,-1].copy() 
    test_set[:,-1].fill(np.nan) # set last column to nan
    if len(np.unique(training_set[:,-1])) < 20: # check if target column categorical
        # score = pd.DataFrame()
        # encode categorical values
        # le = LabelEncoder()
        # training_set[:,-1] = le.fit_transform(training_set[:,-1])
        # test_set[:,-1] = le.fit_transform(test_set[:,-1])
        # first impute all missing values in X with SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(training_set[:,:-1])
        training_set[:,:-1] = imp.transform(training_set[:,:-1])
        imp.fit(test_set[:,:-1])
        test_set[:,:-1] = imp.transform(test_set[:,:-1])
        # now apply classifier on imputed dataset
        clf = RandomForestClassifier()
        # perform grid search for classifier
        parameters = {
        'n_estimators': [2, 10, 100],
        'max_features': [int(np.sqrt(training_set[:,:-1].shape[-1])), training_set[:,:-1].shape[-1]]
                }
        clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5)
        clf.fit(training_set[:,:-1], training_set[:,-1])
        test_set[:,-1] = clf.predict(test_set[:,:-1])
        # compute score
        score = ['Accuracy: ', clf.score(test_set[:,:-1], test_set_labels)] 
        
    else:
        # score = pd.DataFrame()
        # train imputer model on training set
        #est = RandomForestRegressor(random_state=0)        
        parameters = {
        'n_estimators': [2, 10, 100],
        'max_features': [int(np.sqrt(training_set[:,:-1].shape[-1])), training_set[:,:-1].shape[-1]]
                }
        est = GridSearchCV(RandomForestRegressor(random_state=0), parameters, cv=5)
        imp = IterativeImputer(estimator=est, random_state=0) # choose method
        imp.fit(training_set) # fit on training data
        y_pred = imp.transform(test_set) # impute test data
        score = ['MSE: ', mse(test_set_labels, y_pred[:,-1])]
        # result = [
        #     'estimator', f'{estimator}',
        #     'mse', mse(test_set[:,-1], y_pred[:,-1])]
        # results.append(result)
    
    return score

# dataset = load_boston(return_X_y=False)
# mask = generate_missing_mask(dataset, missingness='MAR')
# training_set, test_set = generate_missingness(dataset, mask)

# mse_score = sklearn_impute(training_set, test_set)

dataset = load_boston()
mask = generate_missing_mask(dataset, missingness='MAR')
training_set, test_set = generate_missingness(dataset, mask)


