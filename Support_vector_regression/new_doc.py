import pandas as pd
import numpy as np
import sklearn.svm as sk
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize


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
df = pd.read_csv('../data_processing/final_final_final.csv').astype(np.float32)
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
    x_all, y_all = np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val))

    # Todo find hyper param
    pipe = Pipeline([('scaler', StandardScaler()), ('svc', sk.LinearSVR(max_iter=10000))])

    list = []
    for i in range(50, 150, 50):
        list.append(i)

    parameter_grid = {'svc__C': list}

    gridsearch = GridSearchCV(estimator=pipe, param_grid=parameter_grid, scoring="r2", cv=5, verbose=3, n_jobs=3)

    results = gridsearch.fit(x_all, y_all)

    print(f"{results.best_params_}\t{results.best_score_}")

    # Todo make best model
    print("making final model")
    final_model = Pipeline([('scaler', StandardScaler()), ('svc', sk.LinearSVR(max_iter=10000, C=results.best_params_["svc__C"]))])
    final_model.fit(x_all, y_all)

    prediction = final_model.predict(x_test)

    # Todo evaluate model
    r2 = r2_score(y_test, prediction)
    rmse = mean_squared_error(y_test, prediction, squared=False)
    mae = mean_absolute_error(y_test, prediction)

    # Todo return evaluation
    file = open(f'./Support_vector_regression_results_{header[target_index]}.csv', 'w')
    file.write(f"Best C value {results.best_params_}\nR2:{r2}\nrmse:{rmse}\nmae{mae}")
    file.close()

    # Todo

main(1)