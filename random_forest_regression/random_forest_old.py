import csv
import random
import scipy as sp
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from time import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedKFold, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
import pandas as pd
import numpy as np
#import tensorflow as tf
from fast_ml.utilities import display_all
from fast_ml.feature_selection import get_constant_features

#Program skal bygges under keras for at vi kan bruge tensorflow. Se jamie branch.
#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

df = pd.read_csv('../data_processing/final_final_final.csv')
ligament_headers = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
rfr_criterion = ["squared_error", "poisson"]

random_generator = np.random.RandomState(69)

default_parameters = {"n_estimators": 100, "max_depth": None, "min_sample_split": 2, "max_features": 1.0}
n_estimators_default = default_parameters["n_estimators"]
max_depth_default = default_parameters["max_depth"]
min_sample_split_default = default_parameters["min_sample_split"]
max_features_default = default_parameters["max_features"]


def gives_x_all_param_header():
    x = []
    for i in range(1, 24):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x

def save_to_list(r_2, mae, rmse, estimators, max_features, ligament, mode):
    return {"mode": mode, "r_2": r_2, "mae": mae, "rmse": rmse, "estimators": estimators, "max_features": max_features, "ligament": ligament}

def write_best_scores_for_all_knees_to_file(mode, test_size, list_of_result_dictionaries, configurations):
    #
    # iterate entries in dictionary
    # log the number of the entry with higher-than-before r2, or mae, or rmse
    # write the 3 entries to file in an understandable manner
    #
    highest_r2, lowest_mae, lowest_rmse = 0, float('inf'), float('inf')
    for e in list_of_result_dictionaries:
        if e.get("r_2") > highest_r2:
            highest_r2 = e.get("r_2")
            highest_r2_record = e
        if e.get("mae") < lowest_mae:
            lowest_mae = e.get("mae")
            lowest_mae_record = e
        if e.get("rmse") < lowest_rmse:
            lowest_rmse = e.get("rmse")
            lowest_rmse_record = e

    file = open("random_forest_results.txt", "a")
    file.write(f'Mode: {mode}, TestSize: {test_size}, Configurations: {configurations}\nHighest r_2: {highest_r2_record}\nHighest MAE: {lowest_mae_record}\nHighest RMSE: {lowest_rmse_record}\n\n')
    file.close()

#
def last_model_performed_best(list_of_model_results):
    last_result = list_of_model_results[-1]
    last_result_has_lowest_rmse = all(score < last_result for score in list_of_model_results)
    return last_result_has_lowest_rmse

def train_test_return_results(n_trees=100, max_depth=None, min_sample_split=2, max_features=1.0, min_samples_leaf=1, ligament_index=0, min_weight_fraction_leaf=0.0, max_leaf_nodes=None, bootstrap=True):
    x = df[gives_x_all_param_header()]
    y = df[ligament_headers[ligament_index]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=random_generator)

    pipe = Pipeline([('scaler', StandardScaler()), (
    'RFR', RFR(n_estimators=n_trees, max_features=max_features, max_depth=max_depth, min_samples_split=min_sample_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, min_weight_fraction_leaf=min_weight_fraction_leaf, bootstrap=bootstrap,  verbose=3, n_jobs=7))])
    pipe.fit(x_train, y_train)

    y_predict_test = pipe.predict(x_test)
    y_predict_train = pipe.predict(x_train)

    r2_train = r2_score(y_train, y_predict_train)
    mae_train = mean_absolute_error(y_train, y_predict_train)
    rmse_train = mean_squared_error(y_train, y_predict_train, squared=False)
    r2_test = r2_score(y_test, y_predict_test)
    mae_test = mean_absolute_error(y_test, y_predict_test)
    rmse_test = mean_squared_error(y_test, y_predict_test, squared=False)

    return [r2_train, r2_test, mae_train, mae_test, rmse_train, rmse_test]

def file_exists(file_path):
    print(f"File at {file_path} exists!\n")

def write_to_file(content, file_path):
    file = open(file_path, "a")
    file.write(f"{content}\n")
    file.close()

def write_list_to_csv(list, file_path):
    with open(file_path, 'a', newline="") as fd:
        writer = csv.writer(fd, delimiter=";", quoting=csv.QUOTE_NONE, escapechar="")
        writer.writerow(list)
def investigate_hyperparameters(n_trees_range, max_depth_range, min_sample_split_range, max_features_range, ligament_index_range):
    # train all configurations of one parameter.
    # do it for all parameters
    # do it for all ligaments (or at least 2 for now)
    path_of_results = "./hyperparameter_poking.csv"
    file_exists(path_of_results)
    write_to_file("ID;r2_train;r2_test;mae_test;mae_train;rmse_train;rmse_test;n_estimators;max_depth;min_sample_split;max_features", "./hyperparameter_poking.csv")
    for ligament in range(*ligament_index_range):
        for few_trees in range(1, 6):
            results = [ligament_headers[ligament]]
            results.extend(train_test_return_results(n_trees=few_trees, max_depth=None, min_sample_split=2, max_features=1.0, ligament_index=ligament))
            results.extend([few_trees, "MAX", 2, 1.0])
            write_list_to_csv(results, path_of_results)
        for num_trees in range(*n_trees_range):
            results = [ligament_headers[ligament]]
            results.extend(train_test_return_results(n_trees=num_trees, max_depth=None, min_sample_split=2, max_features=1.0, ligament_index=ligament))
            results.extend([num_trees, "MAX", 2, 1.0])
            write_list_to_csv(results, path_of_results)
            #write_to_file(f"{num_trees};MAX;2;1.0", path_of_results)
        for depth in range(*max_depth_range):
            results = [ligament_headers[ligament]]
            results.extend(train_test_return_results(n_trees=100, max_depth=depth, min_sample_split=2, max_features=1.0, ligament_index=ligament))
            results.extend([100, depth, 2, 1.0])
            write_list_to_csv(results, path_of_results)
            #write_to_file(f"100;{depth};2;1.0", path_of_results)
        for min_sample_split in np.arange(*min_sample_split_range):
            results = [ligament_headers[ligament]]
            results.extend(train_test_return_results(n_trees=100, max_depth=None, min_sample_split=min_sample_split, max_features=1.0, ligament_index=ligament))
            results.extend([100, "MAX", min_sample_split, 1.0])
            write_list_to_csv(results, path_of_results)
            #write_to_file(f"100;MAX;{min_sample_split};1.0", path_of_results)
        for max_features in np.arange(*max_features_range):
            results = [ligament_headers[ligament]]
            results.extend(train_test_return_results(n_trees=100, max_depth=None, min_sample_split=2, max_features=max_features, ligament_index=ligament))
            results.extend([100, "MAX", 2, max_features])
            write_list_to_csv(results, path_of_results)
            #write_to_file(f"100;MAX;2;{min_sample_split}", path_of_results)

def investigate_sub_100_trees(n_trees_range, ligament_index_range, min_samples_leaf_range):
    path_of_results = "./hyperparameter_poking2.csv"
    file_exists(path_of_results)
    write_to_file("ID;r2_train;r2_test;mae_test;mae_train;rmse_train;rmse_test;n_estimators;max_depth;min_sample_split;max_features;min_samples_leaf", "./hyperparameter_poking2.csv")
    for ligament in range(*ligament_index_range):
        '''
        for small_num_trees in range(1, 6):
            results = [ligament_headers[ligament]]
            results.extend(train_test_return_results(n_trees=small_num_trees, max_depth=None, min_sample_split=2, max_features=1.0, ligament_index=ligament))
            results.extend([small_num_trees, "MAX", 2, 1.0, 1])
            write_list_to_csv(results, path_of_results)
        for num_trees in range(*n_trees_range):
            results = [ligament_headers[ligament]]
            results.extend(train_test_return_results(n_trees=num_trees, max_depth=None, min_sample_split=2, max_features=1.0, ligament_index=ligament))
            results.extend([num_trees, "MAX", 2, 1.0, 1])
            write_list_to_csv(results, path_of_results)
        '''
        for min_samples in range(*min_samples_leaf_range):
            results = [ligament_headers[ligament]]
            results.extend(train_test_return_results(n_trees=100, max_depth=None, min_sample_split=2, max_features=1.0, ligament_index=ligament, min_samples_leaf=min_samples))
            results.extend([100, "MAX", 2, 1.0, min_samples])
            write_list_to_csv(results, path_of_results)


def investigate_max_leaf_nodes(ligament_index_range, max_leaf_nodes_range):
    path_of_results = "./hyperparameter_poking_max_leaf_nodes.csv"
    file_exists(path_of_results)
    write_to_file("ID;r2_train;r2_test;mae_test;mae_train;rmse_train;rmse_test;max_leaf_nodes","./hyperparameter_poking_investigate_max_leaf_nodes.csv")
    for ligament in range(*ligament_index_range):
        for max_leaf_nodes in range(*max_leaf_nodes_range):
            results = [ligament_headers[ligament]]
            results.extend(train_test_return_results(n_trees=100, max_depth=None, min_sample_split=2, max_features=1.0, ligament_index=ligament, max_leaf_nodes=max_leaf_nodes))
            results.extend([max_leaf_nodes])
            write_list_to_csv(results, path_of_results)


def train_single_forest(ligament_index, estimators, max_features, test_size, max_depth, bootstrap=True, min_samples_leaf=1, min_samples_split=2):
    x = df[gives_x_all_param_header()]
    y = df[ligament_headers[ligament_index]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=random.seed(69))

    time_before_train = time()
    pipe = Pipeline([('scaler', StandardScaler()), ('RFR', RFR(n_estimators=estimators, max_features=max_features, max_depth=max_depth, bootstrap=bootstrap, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, verbose=3, n_jobs=7))])
    pipe.fit(x_train, y_train)

    y_predict_test = pipe.predict(x_test)
    y_predict_train = pipe.predict(x_train)

    r2_train = r2_score(y_train, y_predict_train)
    mae_train = mean_absolute_error(y_train, y_predict_train)
    rmse_train = mean_squared_error(y_train, y_predict_train, squared=False)
    r2_test = r2_score(y_test, y_predict_test)
    mae_test = mean_absolute_error(y_test, y_predict_test)
    rmse_test = mean_squared_error(y_test, y_predict_test, squared=False)

    print(f'Time elapsed: {time()-time_before_train}')
    print(f'rmse_train;rmse_test;r2_train;r2_test;mae_train;mae_test')
    print(f'{rmse_train};{rmse_test};{r2_train};{r2_test};{mae_train};{mae_test}')

def random_forest_random_parameters(estimators_range, max_features_range, n_configurations, ligament_index, test_size):
    x = df[gives_x_all_param_header()]
    y = df[ligament_headers[ligament_index]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=random.seed(69))
    list_of_results_train = list(dict())
    list_of_results_test = list(dict())
    list_of_rmse_test_scores = list()

    for c in range(n_configurations):
        estimators = random.randint(estimators_range[0], estimators_range[1])
        max_features = random.uniform(max_features_range[0], max_features_range[1])
        #pipe = Pipeline([('scaler', StandardScaler()), (
        #'RFR', RFR(n_estimators=estimators, max_features=max_features))])
        regressor = RFR(n_estimators=estimators, max_features=max_features)

        #regressor.fit(x_train, y_train)
        regressor.fit(x_train, y_train)

        y_predict_test = regressor.predict(x_test)
        y_predict_train = regressor.predict(x_train)

        r2_train = r2_score(y_train, y_predict_train)
        mae_train = mean_absolute_error(y_train, y_predict_train)
        rmse_train = mean_squared_error(y_train, y_predict_train, squared=False)

        r2_test = r2_score(y_test, y_predict_test)
        mae_test = mean_absolute_error(y_test, y_predict_test)
        rmse_test = mean_squared_error(y_test, y_predict_test, squared=False)

        list_of_results_train.append(save_to_list(mode="train", r_2=r2_train, mae=mae_train, rmse=rmse_train, estimators=estimators, max_features=max_features,
            ligament=ligament_headers[ligament_index]))

        list_of_results_test.append(save_to_list(mode="test", r_2=r2_test, mae=mae_test, rmse=rmse_test, estimators=estimators, max_features=max_features,
            ligament=ligament_headers[ligament_index]))

        list_of_rmse_test_scores.append(rmse_test)
        if last_model_performed_best(list_of_rmse_test_scores): best_model = regressor

    write_best_scores_for_all_knees_to_file("train", test_size, list_of_results_train, n_configurations)
    write_best_scores_for_all_knees_to_file("test", test_size, list_of_results_test, n_configurations)
    # PLOT HERE

def investigate_min_weight_fraction_leaf(ligament_index_range, min_weight_fraction_leaf_range):
    path_of_results = "./hyperparameter_poking_min_weight_fraction_leaf.csv"
    file_exists(path_of_results)
    write_to_file("ID;r2_train;r2_test;mae_test;mae_train;rmse_train;rmse_test;max_leaf_nodes",
                  "./hyperparameter_poking_min_weight_fraction_leaf.csv")
    for ligament in range(*ligament_index_range):
        for min_weight_fraction_leaf in np.arange(*min_weight_fraction_leaf_range):
            results = [ligament_headers[ligament]]
            results.extend(train_test_return_results(n_trees=100, max_depth=None, min_sample_split=2, max_features=1.0,
                                                     ligament_index=ligament, min_weight_fraction_leaf=min_weight_fraction_leaf))
            results.extend([min_weight_fraction_leaf])
            write_list_to_csv(results, path_of_results)


def investigate_bootstrap(ligament_index_range, bootstrap):
    path_of_results = "./hyperparameter_poking_bootstrap.csv"
    file_exists(path_of_results)
    write_to_file("ID;r2_train;r2_test;mae_test;mae_train;rmse_train;rmse_test;bootstrap",
                  "./hyperparameter_poking_bootstrap.csv")
    for ligament in range(*ligament_index_range):
        results = [ligament_headers[ligament]]
        results.extend(train_test_return_results(n_trees=100, max_depth=None, min_sample_split=2, max_features=1.0, ligament_index=ligament, bootstrap=bootstrap))
        results.extend([bootstrap])
        write_list_to_csv(results, path_of_results)

def investigate_min_sample_split(ligament_index_range, min_sample_split_range):
    path_of_results = "./hyperparameter_poking_min_sample_split.csv"
    file_exists(path_of_results)
    write_to_file("ID;r2_train;r2_test;mae_test;mae_train;rmse_train;rmse_test;min_sample_split",
                  "./hyperparameter_poking_min_sample_split.csv")
    for ligament in range(*ligament_index_range):
        for min_sample_split in np.arange(*min_sample_split_range):
            results = [ligament_headers[ligament]]
            results.extend(train_test_return_results(n_trees=100, max_depth=None, min_sample_split=min_sample_split, max_features=1.0, ligament_index=ligament))
            results.extend([min_sample_split])
            write_list_to_csv(results, path_of_results)


def make_default_forests():
    for ligament_index in range(0,8):
        x = df[gives_x_all_param_header()]
        y = df[ligament_headers[ligament_index]]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=random.seed(69))

        time_before_train = time()
        pipe = Pipeline([('scaler', StandardScaler()), ('RFR', RFR(verbose=3, n_jobs=8))])
        pipe.fit(x_train, y_train)

        y_predict_test = pipe.predict(x_test)
        y_predict_train = pipe.predict(x_train)

        r2_train = r2_score(y_train, y_predict_train)
        mae_train = mean_absolute_error(y_train, y_predict_train)
        rmse_train = mean_squared_error(y_train, y_predict_train, squared=False)
        r2_test = r2_score(y_test, y_predict_test)
        mae_test = mean_absolute_error(y_test, y_predict_test)
        rmse_test = mean_squared_error(y_test, y_predict_test, squared=False)

        file = open('./default_scores.csv', 'a')
        if ligament_index == 0:
            file.writelines("ID;r2_train;r2_test;mae_test;mae_train;rmse_test;rmse_train\n")
        file.writelines(f"{ligament_headers[ligament_index]};{r2_train};{r2_test};{mae_test};{mae_train};{rmse_test};{rmse_train}\n\n")
        file.close()


train_single_forest(estimators=100, max_features=1.0, ligament_index=0, test_size=0.2, max_depth=None, bootstrap=True, min_samples_leaf=1, min_samples_split=100)
#investigate_hyperparameters(n_trees_range=(10, 201, 10), max_depth_range=(1, 51, 5), min_sample_split_range=(2, 11, 1), max_features_range=(0.2, 1.2, 0.2), ligament_index_range=(0, 8))
#investigate_sub_100_trees(n_trees_range=(10, 100, 10), ligament_index_range=(0, 1, 1), min_samples_leaf_range=(1, 101, 10))
#investigate_max_leaf_nodes((0,1,1), max_leaf_nodes_range=(1500, 2001, 500))
#investigate_min_weight_fraction_leaf((0,1,1), min_weight_fraction_leaf_range=(0.0, 0.51, 0.1))
#investigate_bootstrap(ligament_index_range=(0, 8, 1), bootstrap=False)
#investigate_bootstrap(ligament_index_range=(0, 8, 1), bootstrap=True)
#investigate_min_sample_split(ligament_index_range=(0, 1, 1), min_sample_split_range=(100, 201, 100))



