#!/usr/bin/python

import pickle
import pandas as pd
import numpy as np

def convert_and_clean_data(data_dict, fill_na = 1.e-5):
    '''
    Takes a dataset as a dictionary, then converts it into a Pandas DataFrame for convenience. 
    Replaces all NA values by the value specified in 'fill_na' (or None).
    Cleans up data errors on two observations.
    Returns a Pandas DataFrame.
    '''
    # Convert to DataFrame
    data_df = pd.DataFrame.from_dict(data_dict, orient = 'index', dtype = float)
    if fill_na:
        data_df = data_df.fillna(fill_na)

    # Sort columns in correct order
    column_names = ['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 
    'deferral_payments', 'loan_advances','other', 'expenses', 'director_fees', 'total_payments', 
    'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value',
    'from_messages', 'to_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 
    'shared_receipt_with_poi']
    data_df = data_df[column_names]

    # Correct two data errors
    # Robert Belfer: Data shifted right by one column
    for j in xrange(1, 14):
        data_df.ix['BELFER ROBERT', j] = data_df.ix['BELFER ROBERT', j + 1]
    data_df.ix['BELFER ROBERT', 14] = 1.e-5
    # Sanjay Bhatnagar: Data shifted left by one column
    for j in xrange(14, 2, -1):
        data_df.ix['BHATNAGAR SANJAY', j] = data_df.ix['BHATNAGAR SANJAY', j - 1]
    data_df.ix['BHATNAGAR SANJAY', 1] = 1.e-5

    return data_df


def drop_outliers(data_df, outliers):
    '''
    'outliers' is a list of indexes for observations to be dropped.
    '''
    data_df = data_df.drop(outliers)
    return data_df


def create_new_features(data_df, log_columns):
    '''
    Creates new email-related features by aggregating some of the existing ones.
    Applies log transformation to the specified list of features.
    '''

    # Create 3 aggregate email features to help reduce dimensionality 
    data_df.loc[:, 'sent_vs_received'] = 1. * data_df.loc[:, 'from_messages'] / \
        data_df.loc[:, 'to_messages']
    data_df.loc[:, 'total_emails'] = data_df.loc[:, 'from_messages'] + data_df.loc[:, 'to_messages']
    data_df.loc[:, 'emails_with_poi'] = data_df.loc[:, 'from_this_person_to_poi'] + \
        data_df.loc[:, 'from_poi_to_this_person'] + data_df.loc[:, 'shared_receipt_with_poi']

    # Create log-transformed features from the features in list to make data look closer to normal
    for col in log_columns:
        # Some of the financial data is negative, which causes undefined values with log. Take abs:
        data_df.loc[:, 'log_' + col] = np.log(np.abs(data_df.loc[:, col]))

    return data_df


log_columns = ['bonus', 'deferred_income', 'long_term_incentive', 'other', 
    'restricted_stock_deferred', 'total_stock_value']


features_list = ['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 
    'deferral_payments', 'loan_advances','other', 'expenses', 'director_fees', 'total_payments', 
    'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 
    'total_stock_value', 'from_messages', 'to_messages', 'from_poi_to_this_person', 
    'from_this_person_to_poi', 'shared_receipt_with_poi']

