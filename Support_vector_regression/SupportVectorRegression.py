import pandas as pd
import numpy as np
import sklearn.svm as sk
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline



# Constants to change style in graph
MarkerSize = 0.1
DotColor = 'Blue'

# Constants to change x & y axises in the saved graphs
k_x_left     = -2500
k_x_right    = 30000
k_y_bottom   = 9500
k_y_top      = 9700

epsr_x_left   = -0.10
epsr_x_right  = 0.30
epsr_y_bottom = -0.10
epsr_y_top    = 0.30


def gives_header_array():
    header_array = ['0', 'ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
    for i in range(1, 24):
        header_array.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return header_array


def print_a_graph(target, prediction, target_index, on_what_data, test_size):
    # set data
    plt.figure()
    plt.scatter(target, prediction, color=DotColor, s=MarkerSize)
    # plt.plot([0, 10000], [0, 10000], color='red')

    # set names
    plt.title(f'{header[target_index]} {on_what_data} model')
    plt.xlabel(f'Actual {header[target_index]} value')
    plt.ylabel(f'Predicted {header[target_index]} value')

    # set axies
    # plt.xlim(x_left, x_right)
    # plt.ylim(k_y_bottom, k_y_top)

    # saves/shows graph
    # plt.show()
    plt.savefig(f'./Support_vector_regression_figures/prediction_{on_what_data}_{header[target_index]}_{100-test_size*100}_{test_size*100}.png')


def make_svr_graph(target_index, test_size, c, print_graph=False):
    # find relevant data
    x = df[header[9:285]]
    y = df[header[target_index]]

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=1-test_size, test_size=test_size, shuffle=True)

    # making the model and fitting the model to the data
    pipe = Pipeline([('scaler', StandardScaler()), ('svc', sk.LinearSVR(max_iter=4000 * c, C=c))])
    pipe.fit(x_train, y_train)

    # linear_svr_model = make_pipeline(StandardScaler(), sk.LinearSVR(random_state=0, tol=1e-5))
    # linear_svr_model.fit(x, y)


    # predicting results with both test and train
    # predictions_train = linear_svr_model.predict(x_train)
    # predictions_test = linear_svr_model.predict(x_test)

    predictions_train = pipe.predict(x_train)
    predictions_test = pipe.predict(x_test)

    # plotting the graph
    if print_graph:
        print_a_graph(y_train, predictions_train, target_index, 'train', test_size)
        print_a_graph(y_test, predictions_test, target_index, 'test', test_size)

    # calculating correctness values
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
df = pd.read_csv('../data_processing/final_final_final.csv')
header = gives_header_array()

results = dict()
rounds = 20

for size_of_test in range(10, 20, 10):
    file = open(f'./Support_vector_regression_figures-results_{100-size_of_test}_{size_of_test}.csv', 'w')
    file.write('ID;Max;Min;Avg\n')

    for i in range(1, 9):
        print('\n' + header[i] + ':')
        results[header[i] + '_train'] = []
        results[header[i] + '_test'] = []
        make_svr_graph(i, size_of_test/100, 500, True)

        for j in range(rounds):
            print("\r" f'{j + 1} / {rounds}', end='')
            list_data = make_svr_graph(i, size_of_test/100, 500)
            update_dict(results, header[i], list_data)

        res = find_stats(results, header[i])
        file.writelines(res)
    file.close()




# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR
