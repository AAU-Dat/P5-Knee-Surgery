import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from scipy.constants._codata import val
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')

# Constants to change style in graph
MarkerSize = 0.1
DotColor = 'Blue'

# Constants to change x & y axises in the saved graphs
#xleft_k     = -2500
#xright_k    = 30000
ybottom_k   = -2500
ytop_k      = 30000

#xleft_epsr  = -0.10
#xright_epsr = 0.30
ybottom_epsr= -0.10
ytop_epsr   = 0.30

# This function make sure that y has all the 276 columns
def gives_x_all_param_header():
    x = []
    for i in range(1, 4):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x

def graph_information_k_value(title, xlable, ylable, ybottom, ytop):
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    #plt.xlim(xleft, xright)
    plt.ylim(ybottom, ytop)

def graph_information_epsr_value(title, xlable, ylable, ybottom, ytop):
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    #plt.xlim(xleft, xright)
    plt.ylim(ybottom, ytop)

def plt_graph_test(y_test, predictions_test, header_name):
    plt.figure()  # This makes a new figure
    plt.scatter(y_test, predictions_test, color=DotColor, s=MarkerSize)

    if 'k' in header_name:
        graph_information_k_value(f'{header_name} test model', 'Actual ' f'{header_name} value', 'Predicted ' f'{header_name} value', ybottom_k, ytop_k)
    else:
        graph_information_epsr_value(f'{header_name} test model', 'Actual ' f'{header_name} value', 'Predicted ' f'{header_name} value', ybottom_epsr, ytop_epsr)

    plt.savefig('./polynomial-regression-figures/prediction_test_' f'{header_name}.png')

def plt_graph_train(y_train, predictions_train, header_name):
    plt.figure()  # This makes a new figure
    plt.scatter(y_train, predictions_train, color=DotColor, s=MarkerSize)

    if 'k' in header_name:
        graph_information_k_value(f'{header_name} train model', 'Actual ' f'{header_name} value', 'Predicted ' f'{header_name} value', ybottom_k, ytop_k)
    else:
        graph_information_epsr_value(f'{header_name} train model', 'Actual ' f'{header_name} value', 'Predicted ' f'{header_name} value', ybottom_epsr, ytop_epsr)

    plt.savefig('./polynomial-regression-figures/prediction_train_' f'{header_name}.png')

def train_test_model(header):
    x = df[gives_x_all_param_header()]
    y = df[header]

    poly = PolynomialFeatures(degree=2, include_bias=False)

    X_ = poly.fit_transform(x)

    regr = linear_model.LinearRegression()

    regr.fit(X_, y)

    # creating train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80, test_size=0.20, shuffle=True)

    # creating a regression model
    model = LinearRegression()

    # fitting the model
    model.fit(x_train, y_train)

    # making predictions
    predictions_test = model.predict(x_test)
    predictions_train = model.predict(x_train)

    plt_graph_train(y_train, predictions_train, header)
    plt_graph_test(y_test, predictions_test, header)

def dynamic_train_test_model(header):
    x = df[gives_x_all_param_header()]
    y = df[header]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_ = poly.fit_transform(x)

    regr = linear_model.LinearRegression()
    regr.fit(X_, y)

    # creating train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80, test_size=0.20, shuffle=True)

    # creating a regression model
    model = LinearRegression()

    # fitting the model
    model.fit(x_train, y_train)

    # making predictions
    predictions_test = model.predict(x_test)
    predictions_train = model.predict(x_train)

    r2_train = r2_score(y_train, predictions_train)
    rmse_train = mean_squared_error(y_train, predictions_train, squared=False)
    mae_train = mean_absolute_error(y_train, predictions_train)

    r2_test = r2_score(y_test, predictions_test)
    rmse_test = mean_squared_error(y_test, predictions_test, squared=False)
    mae_test = mean_absolute_error(y_test, predictions_test)

    return [r2_train, rmse_train, mae_train, r2_test, rmse_test, mae_test]

def update_dict(dict, header, list_data):
    l_train = dict[header + '_train']
    l_test = dict[header + '_test']
    l_train.append({'R2': list_data[0], 'RMSE': list_data[1], 'MAE': list_data[2]})
    l_test.append({'R2': list_data[3], 'RMSE': list_data[4], 'MAE': list_data[5]})

def get_stats(header, list):
    return f'{header};{np.max(list)};{np.min(list)};{np.mean(list)}\n'

def find_stats(dict, header):
    r2_train = [x['R2'] for x in dict[header + '_train']]
    rmse_train = [x['RMSE'] for x in dict[header + '_train']]
    mae_train = [x['MAE'] for x in dict[header + '_train']]
    r2_test = [x['R2'] for x in dict[header + '_test']]
    rmse_test = [x['RMSE'] for x in dict[header + '_test']]
    mae_test = [x['MAE'] for x in dict[header + '_test']]

    res = []
    res.append(get_stats(header + '_train_R2', r2_train))
    res.append(get_stats(header + '_train_RMSE', rmse_train))
    res.append(get_stats(header + '_train_MAE', mae_train))
    res.append(get_stats(header + '_test_R2', r2_test))
    res.append(get_stats(header + '_test_RMSE', rmse_test))
    res.append(get_stats(header + '_test_MAE', mae_test))
    return res


# importing data and converting it to float32
df = pd.read_csv('../data_processing/final_final_final.csv').astype(np.float32)

y_head = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
results = dict()
rounds = 50
file = open('./polynomial-regression-figures-results.csv', 'w')
file.write('ID;Max;Min;Avg\n')

for header in y_head:
    results[header + '_train'] = []
    results[header + '_test'] = []
    train_test_model(header)
    print('\n' + header + ':')

    for j in range(rounds):
        print("\r" f'{j + 1} / {rounds}', end='')
        list_data = dynamic_train_test_model(header)
        update_dict(results, header, list_data)

    res = find_stats(results, header)
    file.writelines(res)

file.close()