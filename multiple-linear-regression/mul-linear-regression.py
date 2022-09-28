import numpy as np
import pandas
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


# importing data
df = pd.read_csv('../data_processing/final_final_final.csv')

def tests_model(param_result, df):
    x = df[gives_x_all_param_header()]
    y = df[param_result]

    # This prints out the x and y, it's just for checking that everything is correct
    # print(x)
    # print(y)

    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    # creating train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)  # Remember to shuffle test

    # creating a regression model
    model = LinearRegression()

    # fitting the model
    model.fit(x_train, y_train)

    # making predictions
    predictions_train = model.predict(x_train) #gem
    predictions_test = model.predict(x_test) #gem

    names = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
    df = pandas.DataFrame(columns=['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr'])
    for x in range(0, 8):


# model evaluation fro predict_train
print('Predict_train')
print('r2 value is: ', r2_score(y_train, predictions_train))
print('Root Mean Squared Error (RMSE) : ', mean_squared_error(y_train, predictions_train, squared=False))
print()
# print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

# model evaluation for predict_test
print('Predict_test')
print('r2 value is: ', r2_score(y_test, predictions_test))
print('Root Mean Squared Error (RMSE) : ', mean_squared_error(y_test, predictions_test, squared=False))
# print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))
# ----------------------------------------------------------------------------------------------------------------------

# prints out the graph for predictions_test
plt.scatter(y_test, predictions_test, color=DotColor, s=MarkerSize)
# plt.plot(y_test, predictions, color="blue", linewidth=3)

# To save the graph for predictions_test
plt.savefig('./figures/prediction_test_ACL_espr.png')

# prints out the graph for predictions_train
plt.figure() # This makes a new figure
plt.scatter(y_train, predictions_train, color=DotColor, s=MarkerSize)

# To save the graph for predictions_train
plt.savefig('./figures/prediction_train_ACL_espr.png')
