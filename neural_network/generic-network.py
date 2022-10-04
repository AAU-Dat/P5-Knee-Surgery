# Import needed libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Read and load dataset

dataset = pd.read_csv("../data_processing/final_final_final.csv", index_col=0)
knee_X = dataset.T.tail(-8).T

# Select y = ACL_k as the dependant variable

knee_Y = dataset.T[0].T

# Split the dataset to get training and testing data

X_train, X_test, y_train, y_test = train_test_split(knee_X, knee_Y, test_size=0.2)

# Train the model

regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the testing set

y_pred = regressor.predict(X_test)

# Print coefficients and intercept

print("Coefficients are: ", regressor.coef_)
print("Intercept is: ", regressor.intercept_)
