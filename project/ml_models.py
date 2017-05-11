#!/usr/bin/python

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline

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
    cv = StratifiedShuffleSplit(n_splits = 10, test_size = 0.3, random_state = seed * 2)
    gscv = GridSearchCV(svc, param_grid = param_grid, scoring = 'f1', cv = cv, verbose = 3, 
        n_jobs = -1)

    # Fit model:
    gscv.fit(scaled_X, y)

    print "Best parameters:", gscv.best_params_

    return gscv.best_params_


def make_svc_pipeline(X, y, ):

# SVC parameter grid:
param_grid = {'gamma': np.logspace(-9, 5, 15, base = 2.), 'C': np.logspace(-9, 5, 15, base = 2.)}