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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

from pre_processing import (
    load_data,
    generate_missing_mask,
    generate_missingness,
    convert_to_strings,
    create_source_factors,
    shuffle_data,
    create_numerical_data,
    create_files)

def sklearn_impute(training_set, test_set):
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
    BayesianRidge(),
    DecisionTreeRegressor(max_features='sqrt', random_state=0),
    ExtraTreesRegressor(n_estimators=10, random_state=0),
    KNeighborsRegressor(n_neighbors=15),
    LinearRegression(),
    RandomForestRegressor(random_state=0)
    ]
    
    
    training_set = np.concatenate([training_set.data, training_set.target.reshape(-1,1)], axis=1)
    test_set = np.concatenate([test_set.data, test_set.target.reshape(-1,1)], axis=1)
    test_set_missing = test_set.copy() 
    test_set_missing[:,-1].fill(np.nan) # set last column to nan
    if type(training_set[0,-1]) == str:
        le = LabelEncoder()
        training_set[:,-1] = le.fit_transform(training_set[:,-1])
        test_set[:,-1] = le.fit_transform(test_set[:,-1])
    
    
    score_iterative_imputer = pd.DataFrame()
    for estimator in estimators:
        # train imputer model on training set
        imp = IterativeImputer(estimator=estimator, random_state=0) # choose method
        imp.fit(training_set) # fit on training data
        y_pred = imp.transform(test_set_missing) # impute test data
        score_iterative_imputer[estimator.__class__.__name__] = [mse(test_set[:,-1], y_pred[:,-1])]
        # result = [
        #     'estimator', f'{estimator}',
        #     'mse', mse(test_set[:,-1], y_pred[:,-1])]
        # results.append(result)
    
    return score_iterative_imputer

# dataset = load_boston(return_X_y=False)
# mask = generate_missing_mask(dataset, missingness='MAR')
# training_set, test_set = generate_missingness(dataset, mask)

# mse_score = sklearn_impute(training_set, test_set)

# dataset = sklearn.datasets.fetch_openml('blood-transfusion-service-center', return_X_y=False)
# mask = generate_missing_mask(dataset, missingness='MAR')
# training_set, test_set = generate_missingness(dataset, mask)
