import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv("../data_processing/final_final_final.csv")

target_column = ['ACL_k']
predictors = list(set(list(raw_data.columns))-set(target_column))

X = raw_data[predictors].values
Y = raw_data[target_column].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=40)
print(X_train.shape)
print(X_test.shape)

# TODO : Create the Logic for the Network

mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train, Y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

# TODO : Train the Neural Network

# TODO : Show Results
