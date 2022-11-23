import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
matplotlib.use('Agg')

# Constants to change style in graph
MarkerSize = 0.1
DotColor = 'Blue'

# This function make sure that y has all the 276 columns
def gives_x_all_param_header():
    x = []
    for i in range(1, 24):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                       'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                       'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x

def graph_information(title, xlable, ylable, xleft, xright, ybottom, ytop):
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.xlim(xleft, xright)
    plt.ylim(ybottom, ytop)

# importing data
df = pd.read_csv('../data_processing/final_final_final.csv')

x = df[gives_x_all_param_header()]
y = df[['ACL_epsr']]

# This prints out the x and y, it's just for checking that everything is correct
print(x)
print(y)

# Make a for loop that iterates thourgh x numbers of models of x_test and ----------------------------------------------
# saves the best and worse r value and the average r value
regr = linear_model.LinearRegression()
regr.fit(x, y)

# creating train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2) # Remember to shuffle test

# creating a regression model
model = LinearRegression()

# fitting the model
model.fit(x_train, y_train)

# making predictions
predictions_test = model.predict(x_test)
predictions_train = model.predict(x_train)

# model evaluation fro predict_train
print('Predict_train for ACL_epsr')
print('r2 value is: ', r2_score(y_train, predictions_train))
print('Root Mean Squared Error (RMSE) : ', mean_squared_error(y_train, predictions_train, squared=False))
print('mean_absolute_error : ', mean_absolute_error(y_train, predictions_train))
print()

# model evaluation for predict_test
print('Predict_test for ACL_epsr')
print('r2 value is: ', r2_score(y_test, predictions_test))
print('Root Mean Squared Error (RMSE) : ', mean_squared_error(y_test, predictions_test, squared=False))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions_test))
# ----------------------------------------------------------------------------------------------------------------------

# prints out the graph for predictions_test
plt.scatter(y_test, predictions_test, color=DotColor, s=MarkerSize)
graph_information('ACL_epsr test model', 'Actual ACL_epsr value', 'Predicted ACL_epsr value', -0.10, 0.30, -0.10, 0.30)
plt.savefig('./figures/prediction_test_ACL_espr.png')

# prints out the graph for predictions_train
plt.figure() # This makes a new figure
plt.scatter(y_train, predictions_train, color=DotColor, s=MarkerSize)
graph_information('ACL_epsr train model', 'Actual ACL_epsr value', 'Predicted ACL_epsr value', -0.10, 0.30, -0.10, 0.30)
plt.savefig('./figures/prediction_train_ACL_espr.png')

