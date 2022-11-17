import pandas as pd
import numpy as np
import sklearn.svm as sk
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def gives_header_array():
    header_array = ['0', 'ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
    for i in range(1, 24):
        header_array.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return header_array


# test train validation ratios
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1

# data
df = pd.read_csv('../data_processing/final_final_final.csv')
header = gives_header_array()


def print_a_graph(target, prediction, target_index, on_what_data, test_size):
    # set data
    plt.figure()
    plt.scatter(target, prediction, color=DotColor, s=MarkerSize)
    plt.plot(to_k if is_k(target_index) else to_espr, to_k if is_k(target_index) else to_espr, color='red')

    # set names
    plt.title(f'{header[target_index]} {on_what_data} model')
    plt.xlabel(f'Actual {header[target_index]} value')
    plt.ylabel(f'Predicted {header[target_index]} value')

    # set axies
    # plt.xlim(x_left, x_right)
    # plt.ylim(k_y_bottom, k_y_top)

    # saves/shows graph
    # plt.show()
    plt.savefig(f'./Support_vector_regression_figures/{100-test_size*100}_{test_size*100}/{header[target_index]}_{on_what_data}_{100-test_size*100}_{test_size*100}.png')
    plt.close()


def is_k(target_index):
    return target_index % 2 == 1


def main(target_index):
    # Todo make data ready
    # find relevant data
    x = df[header[9:285]]
    y = df[header[target_index]]

    # Train gets the train_ratio of the data set (train = 80%, test = 20% - af det fulde datasæt)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio, random_state=69)

    # Both Validation and Test get 50% each of the remainder (val = 10%, test = 10% - af det fulde datasæt)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + val_ratio), random_state=69)
    x, y = np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val))

    # Todo find hyper param
    pipe = Pipeline([('scaler', StandardScaler()), ('svc', sk.LinearSVR())])

    list = []
    for x in range(50, 1050, 50):
        list.append(x)

    parameter_grid = {'svc__c': list}

    gridsearch = GridSearchCV(estimator=pipe, param_grid=parameter_grid, scoring="neg_root_mean_squared_error", cv=5, verbose=3, n_jobs=3)

    results = gridsearch.fit(x, y)
    # Todo make best model
    # Todo evaluate model
    # Todo return evaluation
    # Todo

main(1)