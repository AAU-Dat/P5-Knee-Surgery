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
from sklearn.decomposition import PCA

df = pd.read_csv('../data_processing/final_final_final.csv').astype('float32')
ligament_headers = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
rfr_criterion = ["squared_error", "poisson"]

random_generator = 69

train_ratio = 0.8
test_ratio = 0.1
validation_ratio = 0.1

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

def test_dimensionality_reduction(ligament_header, calculate=False, write_to_file=False):

    # Read the data
    x = df[gives_x_all_param_header()]
    y = df[ligament_header]

    # Train gets the train_ratio of the data set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio, random_state=random_generator)

    # Both Validation and Test get 50% each of the remainder
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio), random_state=random_generator)

    x_all, y_all = np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val))

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ("RFRegressor", RFR(n_estimators=100, max_depth=None))
    ])

    # Making list with all the steps
    list_param = []
    for i in range(100, 261, 20):
        list_param.append(i)

    param_grid = {"pca__n_components": list_param}

    # Grid search
    rf_grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='r2', n_jobs=5, verbose=3)

    # Fit the model
    result = rf_grid_search.fit(x_all, y_all)

    print('Best score: ', result.best_score_)
    print('Best parameters: ', result.best_params_)

    # Train the model with the best parameters
    best_model_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=result.best_params_['pca__n_components'])),
        ("RFRegressor", RFR(n_estimators=100, max_depth=None))
    ])

    # fit the model
    best_model_rf.fit(x_all, y_all)

    # Predict the test set
    y_predict_test = best_model_rf.predict(x_test)
    y_predict_train = best_model_rf.predict(x_all)

    if calculate:
        r2_train = r2_score(y_all, y_predict_train)
        mae_train = mean_absolute_error(y_all, y_predict_train)
        rmse_train = mean_squared_error(y_all, y_predict_train, squared=False)

        r2_test = r2_score(y_test, y_predict_test)
        mae_test = mean_absolute_error(y_test, y_predict_test)
        rmse_test = mean_squared_error(y_test, y_predict_test, squared=False)

    if write_to_file:
        file = open('./rf_dim_reduction_best.csv', 'a')
        if ligament_header == 'ACL_k':
            file.writelines("Holdout trial scores: ID;r2_train;r2_test;mae_test;mae_train;rmse_test;rmse_train\n")
        file.writelines(f"Ligament attribute: {ligament_header}\n")
        file.writelines(f"best hypermodel: {ligament_header}\nbest hypermodel rmse: {-result.best_score_}\n")
        file.writelines(f'Best Dimension for: {ligament_header} is: {result.best_params_["pca__n_components"]}\n')
        file.writelines(f"Holdout trial scores: {ligament_header};{r2_train};{r2_test};{mae_test};{mae_train};{rmse_test};{rmse_train}\n\n")
        file.close()

        #cvResultsFile = open('./rf_di_reduction_cvResults.csv', 'a')
        #cvResultsFile.writelines(result.cv_results_)
        #file.close()
        #resdf = pd.DataFrame(result.cv_results_)
        #resdf.to_csv('./rf_di_reduction_cvResults.csv', mode='a')

for header in ligament_headers:
    print('\n' + header + ':')
    test_dimensionality_reduction(header, calculate=True, write_to_file=True)
