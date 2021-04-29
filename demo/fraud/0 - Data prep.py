#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import time

randomState = 12
np.random.seed(randomState)

################################
# Load the dataset
################################
start_time = time.time()

df = pd.read_csv('data/Fraud detection dataset.csv')

df = df.rename(columns={'oldbalanceOrg': 'oldBalanceOrig',
                        'newbalanceOrig': 'newBalanceOrig',
                        'oldbalanceDest': 'oldBalanceDest',
                        'newbalanceDest': 'newBalanceDest'})

end_time_load = time.time()

print("\nJust loaded data head :")
print(df.head())

################################
# Data cleaning
################################

# We keep only tranfer and cash out operations because they are the ony one with fraud
data = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]

# Eliminate columns shown to be irrelevant for analysis in the EDA
data = data.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

# Binary-encoding of labelled data in 'type'
data.loc[data.type == 'TRANSFER', 'type'] = 0
data.loc[data.type == 'CASH_OUT', 'type'] = 1
data.type = data.type.astype(int)  # convert dtype('O') to dtype(int)

################################
# Imputation of Latent Missing Values
################################
# The data has several transactions with zero balances in the destination account both before and after a non-zero
# amount is transacted. The fraction of such transactions, where zero likely denotes a missing value, is much larger
# in fraudulent (50%) compared to genuine transactions (0.06%).

# Since the destination account balances being zero is a strong indicator of fraud, we do not impute the account
# balance (before the transaction is made) with a statistic or from a distribution with a subsequent adjustment for
# the amount transacted. Doing so would mask this indicator of fraud and make fraudulent transactions appear genuine.
# Instead, below we replace the value of 0 with -1 which will be more useful to a suitable machine-learning (ML)
# algorithm detecting fraud.

data.loc[(data.oldBalanceDest == 0) & (data.newBalanceDest == 0) & (data.amount != 0), ['oldBalanceDest',
                                                                                        'newBalanceDest']] = - 1

# The data also has several transactions with zero balances in the originating account both before and after a non-zero
# amount is transacted. In this case, the fraction of such transactions is much smaller in fraudulent (0.3%) compared
# to genuine transactions (47%). Once again, from similar reasoning as above, instead of imputing a numerical value
# we replace the value of 0 with a null value.
data.loc[(data.oldBalanceOrig == 0) & (data.newBalanceOrig == 0) & (data.amount != 0), ['oldBalanceOrig',
                                                                                        'newBalanceOrig']] = np.nan

data['errorBalanceOrig'] = data.newBalanceOrig + data.amount - data.oldBalanceOrig
data['errorBalanceDest'] = data.oldBalanceDest + data.amount - data.newBalanceDest

################################
# Scaling
################################

columns_features = list(data.columns)
columns_features.remove("isFraud")

ct = ColumnTransformer([
    ('somename', StandardScaler(), columns_features)
], remainder='passthrough')

scaled_data = ct.fit_transform(data)
scaled_data = pd.DataFrame(scaled_data, columns=columns_features + ["isFraud"])

scaled_data = data

# move isFraud to 1st column
scaled_data.insert(0, "isFraud", scaled_data.pop("isFraud"))

print("\nScaled data head :")
print(scaled_data.head())

end_time_prep = time.time()

################################
# Splitting and saving
################################
# Split data between train and test set
train, test = train_test_split(scaled_data, test_size=0.2, random_state=randomState, stratify=scaled_data['isFraud'])

# Split train data for 2 data owner simulation
train1, train2 = train_test_split(train, test_size=0.6, random_state=randomState, stratify=train['isFraud'])

# Split train data for training and validation
train, val = train_test_split(train, test_size=0.2, random_state=randomState, stratify=train['isFraud'])
# Split train data for training and validation
train1, val1 = train_test_split(train1, test_size=0.2, random_state=randomState, stratify=train1['isFraud'])
# Split train data for training and validation
train2, val2 = train_test_split(train2, test_size=0.2, random_state=randomState, stratify=train2['isFraud'])

print("\nTrain, val and test shape")
print(train.shape[0], val.shape[0], test.shape[0])

print("\nTest data head :")
print(test.head())

################################
# Save each dataframe
################################

test.to_csv(r'../data/test.csv', index=False, header=False)

train.to_csv(r'../data/train.csv', index=False, header=False)

train1.to_csv(r'../data/train1.csv', index=False, header=False)
train2.to_csv(r'../data/train2.csv', index=False, header=False)

val1.to_csv(r'../data/val1.csv', index=False, header=False)
val2.to_csv(r'../data/val2.csv', index=False, header=False)

print("Dataset loading time : {}s".format(round(end_time_load - start_time, 2)))
print("Dataset prep time : {}s".format(round(end_time_prep - end_time_load, 2)))
print("Splitting and saving time : {}s".format(round(time.time() - end_time_prep, 2)))
