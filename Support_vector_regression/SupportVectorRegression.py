import pandas as pd
import numpy as np
import sklearn.svm as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Constants to change style in graph
MarkerSize = 0.1
DotColor = 'Blue'

# Constants to change x & y axises in the saved graphs
k_x_left     = -2500
k_x_right    = 30000
k_y_bottom   = -2500
k_y_top      = 30000

epsr_x_left   = -0.10
epsr_x_right  = 0.30
epsr_y_bottom = -0.10
epsr_y_top    = 0.30


def gives_header_array():
    x = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
    for i in range(1, 24):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x


def print_graph(target, prediction, target_index, on_what_data):
    # set data
    plt.figure()
    plt.scatter(target, prediction, color=DotColor, s=MarkerSize)

    # set names
    plt.title(f'{header[target_index]} {on_what_data} model')
    plt.xlabel(f'Actual {header[target_index]} value')
    plt.ylabel(f'Predicted {header[target_index]} value')

    # set axies
    # plt.xlim(x_left, x_right)
    # plt.ylim(y_bottom, y_top)

    # saves/shows graph
    # plt.show()
    plt.savefig(f'./Support_vector_regression_figures/prediction_{on_what_data}_{header[target_index]}.png')


def make_SVR_graph(target_index, print=False):
    # find relevant data
    x = df[header[9:285]]
    y = df[header[target_index]]

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, shuffle=True)

    # making the model and fitting the model to the data
    linear_svr_model = sk.LinearSVR()
    linear_svr_model.fit(x_train, y_train)

    # predicting results with both test and train
    predictions_train = linear_svr_model.predict(x_train)
    predictions_test = linear_svr_model.predict(x_test)

    # plotting the graph
    if(print):
        print_graph(y_train, predictions_train, target_index, 'train')
        print_graph(y_test, predictions_test, target_index, 'test')

    #calculating correctness values
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


# importing data
df = pd.read_csv('../data_processing/final_final_final.csv', index_col=0)
results = dict()
rounds = 20
file = open('./Support_vector_regression_figures-results.csv', 'w')
file.write('ID;Max;Min;Avg\n')
header = gives_header_array()
for x in range(0, 8):
    results[header[x] + '_train'] = []
    results[header[x] + '_test'] = []
    make_SVR_graph(x, True)
    print(header[x] + ':')

    for j in range(rounds):
        print("'\r" f'{j + 1} / {rounds}\n', end='')
        list_data = make_SVR_graph(x)
        update_dict(results, header[x], list_data)

    res = find_stats(results, header[x])
    file.writelines(res)

file.close()


# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR
