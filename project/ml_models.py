#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------------------------------

def tune_svc(X, y, param_grid, scoring, seed):
    '''
    Uses grid search and cross-validation to discover the best meta-parameters for a Support Vector
    Classifier. Prints out the best parameters and best score (using the specified scoring method).
    Returns the best parameters.
    '''
    # Scale variables:
    #scaler = RobustScaler()
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    # SVC:
    svc = SVC()

    # Cross-validation and grid search:
    cv = StratifiedShuffleSplit(n_splits = 100, test_size = 0.3, random_state = seed * 2)
    gscv = GridSearchCV(svc, param_grid = param_grid, scoring = scoring, cv = cv, verbose = 1, 
        n_jobs = -1)

    # Fit model:
    gscv.fit(scaled_X, y)

    best_params = gscv.best_params_
    best_score = gscv.best_score_
    print "Best parameters:", best_params
    print "Best CV score:", best_score

    return best_params, best_score


def make_svc_pipeline(params):
    '''
    Builds a pipeline that scales featurse then fits an SVC, using the parameters discovered by
    'tune_svc'.
    Returns the pipeline as a classifier.
    '''
    clf = Pipeline(steps = [('scaler', StandardScaler()), ('svc', SVC())])
    clf.set_params(svc__C = params['C'], svc__gamma = params['gamma'])
    return clf

# --------------------------------------------------------------------------------------------------

def tune_rf(X, y, param_grid, scoring, seed):
    '''
    Uses grid search and cross-validation to discover the best meta-parameters for a Random Forest
    Classifier. Prints out the best parameters and best score (using the specified scoring method).
    Returns the best parameters.
    '''
    # Scale variables:
    #scaler = RobustScaler()
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    # SVC:
    rf = RandomForestClassifier()

    # Cross-validation and grid search:
    cv = StratifiedShuffleSplit(n_splits = 100, test_size = 0.3, random_state = seed * 2)
    gscv = GridSearchCV(rf, param_grid = param_grid, scoring = scoring, cv = cv, verbose = 1, 
        n_jobs = -1)

    # Fit model:
    gscv.fit(scaled_X, y)

    best_params = gscv.best_params_
    best_score = gscv.best_score_
    print "Best parameters:", best_params
    print "Best CV score:", best_score

    return best_params, best_score


def make_rf_pipeline(params):
    '''
    Builds a pipeline that scales featurse then fits an SVC, using the parameters discovered by
    'tune_svc'.
    Returns the pipeline as a classifier.
    '''
    clf = Pipeline(steps = [('scaler', StandardScaler()), ('rf', RandomForestClassifier())])
    clf.set_params(rf__max_features = params['max_features'], 
        rf__n_estimators = params['n_estimators'])
    return clf


def rf_feature_selection(X, y, params, feature_names):
    rf = RandomForestClassifier(max_features = params['max_features'], 
        n_estimators = params['n_estimators'])
    rf.fit(X, y)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis = 0)
    indices = np.argsort(importances)[::-1]
    names = [feature_names[n] for n in indices]
    print indices

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(X[0])), importances[indices], color = "r", yerr = std[indices], 
        align = 'center') 
    locs, xlabels = plt.xticks(range(len(X[0])), names)
    plt.xlim([-1, len(X[0])])
    plt.setp(xlabels, rotation = 90, verticalalignment = 'bottom')
    plt.show()
    return names

