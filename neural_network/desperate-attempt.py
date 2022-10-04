# Step 1: Loading the Required Libraries and Modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, r2_score
from math import sqrt

# Step 2: Reading the Data and Performing Basic Data Checks

df = pd.read_csv('diabetes.csv')
print(df.shape)
print(df.describe().transpose())

# Step 3: Creating Arrays for the Features and the Response Variable

target_column = ['Outcome']
predictors = list(set(list(df.columns)) - set(target_column))
df[predictors] = df[predictors] / df[predictors].max()
print(df.describe().transpose())

# Step 4: Creating the Training and Test Datasets

X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape)
print(X_test.shape)

# Step 5: Building, Predicting, and Evaluating the Neural Network Model
# This example dataset is uses a classifier. While we need a regressor for our data, I have included that below.

mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train, y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print(confusion_matrix(y_train, predict_train))
print(classification_report(y_train, predict_train))

print(confusion_matrix(y_test, predict_test))
print(classification_report(y_test, predict_test))

'''
# Setting up the regressor
mlp = MLPRegressor()
mlp.fit(X_train, y_train)
print(mlp)

# Predicting the regressor
expected_y  = y_test
predicted_y = mlp.predict(X_test)

# Evaluating the regressor
print(metrics.r2_score(expected_y, predicted_y))
print(metrics.mean_squared_log_error(expected_y, predicted_y))
'''
