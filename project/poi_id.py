#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from prep_dataset import *
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from ml_models import *

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### The features below are a mix of the original features found in the dataset and features created
### or transformed by me:

# # All original variables except loan_advances:
# features_list = ['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 
#     'deferral_payments', 'other', 'expenses', 'director_fees', 'total_payments', 
#     'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value',
#     'from_messages', 'to_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 
#     'shared_receipt_with_poi']

# # All original variables except loan_advances, using log variants
# features_list = ['poi', 'salary', 'bonus', 'log_bonus', 'log_long_term_incentive', 'log_deferred_income', 
#     'deferral_payments', 'log_other', 'expenses', 'director_fees', 'total_payments', 
#     'exercised_stock_options', 'restricted_stock', 'log_restricted_stock_deferred', 'log_total_stock_value',
#     'from_messages', 'to_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 
#     'shared_receipt_with_poi']

# # Removing variable deferral_payments
# features_list = ['poi', 'salary', 'bonus', 'log_bonus', 'log_long_term_incentive', 'log_deferred_income', 
#     'log_other', 'expenses', 'total_payments', 'director_fees', 'log_restricted_stock_deferred', 
#     'exercised_stock_options', 'restricted_stock', 'log_total_stock_value',
#     'from_messages', 'to_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 
#     'shared_receipt_with_poi']

# # All original variables except loan_advances, using log variants, replacing email features with aggregates
# features_list = ['poi', 'salary', 'bonus', 'log_bonus', 'log_long_term_incentive', 'log_deferred_income', 
#     'log_other', 'expenses', 'total_payments', 'director_fees', 'log_restricted_stock_deferred', 'deferral_payments',
#     'exercised_stock_options', 'restricted_stock', 'log_total_stock_value',
#     'sent_vs_received', 'total_emails', 'emails_with_poi']

# # Keep original email features, keep deferral_payments, drop total_payments and non-total stock features
# features_list = ['poi', 'salary', 'bonus', 'log_bonus', 'log_long_term_incentive', 'log_deferred_income', 
#     'log_other', 'expenses', 'director_fees', 'log_restricted_stock_deferred', 'deferral_payments',
#     'log_total_stock_value', 'from_messages', 'to_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 
#     'shared_receipt_with_poi']

# # Drop deferral_payments
# features_list = ['poi', 'salary', 'bonus', 'log_bonus', 'log_long_term_incentive', 'log_deferred_income', 
#     'log_other', 'expenses', 'director_fees', 'log_restricted_stock_deferred', 'log_total_stock_value', 
#     'from_messages', 'to_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 
#     'shared_receipt_with_poi']

# Keep deferral_payments, use new email features
features_list = ['poi', 'salary', 'bonus', 'log_bonus', 'log_long_term_incentive', 'log_deferred_income', 
    'log_other', 'expenses', 'director_fees', 'log_restricted_stock_deferred', 'log_total_stock_value', 
    'deferral_payments', 'sent_vs_received', 'total_emails', 'emails_with_poi']

# # Drop deferral_payments
# features_list = ['poi', 'salary', 'bonus', 'log_bonus', 'log_long_term_incentive', 'log_deferred_income', 
#     'log_other', 'expenses', 'director_fees', 'log_restricted_stock_deferred', 'log_total_stock_value', 
#     'sent_vs_received', 'total_emails', 'emails_with_poi']




### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Convert the data to a Pandas DataFrame and clean two errors (shifted columns for R. Belfer and
### S. Bhatnagar)
data_df = convert_and_clean_data(data_dict, fill_na = 1.e-5)

### Task 2: Remove outliers
data_df = drop_outliers(data_df, ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'])

### Task 3: Create new feature(s)
# Define list of features to apply log transformation to:
log_columns = ['bonus', 'deferred_income', 'long_term_incentive', 'other', 
    'restricted_stock_deferred', 'total_stock_value']
# Create new email features and apply log transformation:
data_df = create_new_features(data_df, log_columns)

### Store to my_dataset for easy export below 
my_dataset = data_df.to_dict(orient = 'index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# ************* Random Forest classifier *******************************
### Using cross-validation to determine the best parameters, therefore tasks 4 and 5 are performed
### together.
seed = 42
# Scoring method to use during model tuning:
scoring = 'f1'
# Grid of parameters to explore when tuning the model:
rf_grid = {'n_estimators': [3, 4, 5, 8], 'max_features': [2, 3, 4, 5]}
# Find optimal model parameters:
rf_params, rf_score = tune_rf(features, labels, rf_grid, scoring = scoring, seed = seed)
# Make pipeline using these parameters:
rf_clf = make_rf_pipeline(rf_params)

# ************* Support Vector Machine classifier **********************
# Grid of parameters to explore when tuning the model:
svc_grid = {'gamma': np.logspace(-9., 5., 15, base = 2.), 'C': np.logspace(-9., 5., 15, base = 2.)}
# Find optimal model parameters:
svc_params, svc_score = tune_svc(features, labels, svc_grid, scoring = scoring, seed = seed)
# Make pipeline using these parameters:
svc_clf = make_svc_pipeline(svc_params)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# Retain the model giving the best CV score:
if svc_score >= rf_score:
    clf = svc_clf
else:
    clf = rf_score

# Dump data into a pickle file:    
dump_classifier_and_data(clf, my_dataset, features_list)