```
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

# This function make sure that y has all the 276 columns
def gives_x_all_param_header():
    x = []
    for i in range(1, 24):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                       'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                       'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x

def tests_model(param_names, df_result, df, r2_train_column, r2_test_column, rmse_train_column, rmse_test_column):
    x = df[gives_x_all_param_header()]
    y = df[param_names]

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
    predictions_train = model.predict(x_train)
    predictions_test = model.predict(x_test)

    r2_train = r2_score(y_train, predictions_train)
    r2_test = r2_score(y_test, predictions_test)
    rmse_train = mean_squared_error(y_train, predictions_train, squared=False)
    rmse_test = mean_squared_error(y_test, predictions_test, squared=False)

    df_result = df_result.append({r2_train_column: r2_train, r2_test_column: r2_test,
                                  rmse_train_column: rmse_train, rmse_test_column: rmse_test}, ignore_index=True)

    return df_result

def output_to_terminal_predict_train(y_train, predictions_train):
    # model evaluation fro predict_train
    print('Predict_train')
    print('r2 value is: ', r2_score(y_train, predictions_train))
    print('Root Mean Squared Error (RMSE) : ', mean_squared_error(y_train, predictions_train, squared=False))
    print()

def output_to_terminal_predict_test(y_test, predictions_test):
    # model evaluation for predict_test
    print('Predict_test')
    print('r2 value is: ', r2_score(y_test, predictions_test))
    print('Root Mean Squared Error (RMSE) : ', mean_squared_error(y_test, predictions_test, squared=False))
# ----------------------------------------------------------------------------------------------------------------------

def save_graph_train(y_train, predictions_train):
    plt.figure() # This makes a new figure
    plt.scatter(y_train, predictions_train, color=DotColor, s=MarkerSize)
    plt.savefig('./figures/prediction_train_ACL_epsr.png')

def save_graph_test(y_test, predictions_test):
    plt.figure()  # This makes a new figure
    plt.scatter(y_test, predictions_test, color=DotColor, s=MarkerSize)
    plt.savefig('./figures/prediction_test_ACL_epsr.png')

def generates_columns(name):
    return [name + '_r2_train', name + '_r2_test', name + '_rmse_train', name + '_rmse_test']


# Global variables
param_names = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']

columns = []
# Generates names to columns
for x in range(0, 8):
    columns.extend(generates_columns(param_names[x]))

# Constants to change style in graph
MarkerSize = 0.1
DotColor = 'Blue'

# importing data
df = pd.read_csv('../data_processing/final_final_final.csv')

df_r2_rmse = pd.DataFrame(columns=[columns])

for i in range(0, 32, 4):
    for j in range(0, 10):
       df_r2_rmse = tests_model(param_names[int(i/4)], df_r2_rmse, df, columns[i], columns[i+1], columns[i+2], columns[i+3])

print(df_r2_rmse)
```