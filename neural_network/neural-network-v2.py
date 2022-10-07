# Step 1: Loading the Required Libraries and Modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, r2_score
from math import sqrt

# Step 2: Reading the Data and Performing Basic Data Checks

df = pd.read_csv('../data_processing/final_final_final.csv', index_col=0)
df.shape
df.describe().transpose()

# Step 3: Creating Arrays for the Features and the Response Variable

result_columns = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
target_column = ['ACL_k']
predictors = list(set(list(df.columns)) - set(result_columns))
df[predictors] = df[predictors] / df[predictors].max()
df.describe().transpose()

# Step 4: Creating the Training and Test Datasets

X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)
print(); print("Training set shape: ", X_train.shape)
print("Test set shape: ", X_test.shape)

# Step 5: Building, Predicting, and Evaluating the Neural Network Model

# Setting up the regressor
mlp = MLPRegressor(hidden_layer_sizes=(40, 40), activation='relu', solver='adam', max_iter=650)
mlp.fit(X_train, y_train.ravel())

# Predicting the regressor
expected_y = y_test
predicted_y = mlp.predict(X_test)

# Evaluating the regressor
print(); print(metrics.r2_score(expected_y, predicted_y))
print(metrics.mean_squared_log_error(expected_y, predicted_y))

print(); print("Completed")
