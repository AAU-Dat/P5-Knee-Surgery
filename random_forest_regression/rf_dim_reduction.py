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

train_ratio = 0.8
test_ratio = 0.1
validation_ratio = 0.1

def test_dimensionality_reduction(ligaments_range):
    list_param = []
    for i in range(50, 276, 1):
        list_param.append(i)

    for ligament in range(*ligaments_range):
        x = df[gives_x_all_param_header()]
        y = df[ligament_headers[ligament]]
        # Train gets the train_ratio of the data set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio, random_state=69)
        # Both Validation and Test get 50% each of the remainder
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                        test_size=test_ratio / (test_ratio + validation_ratio),
                                                        random_state=69)
        x_all, y_all = np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val))

        RFRegressor = RFR()
        pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ("RFRegressor", RFRegressor)])

        param_grid = {"pca__n_components": list_param}

        rf_grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring="neg_root_mean_squared_error", cv=ShuffleSplit(n_splits=1, test_size=0.2), verbose=3, n_jobs=-1)
        result = rf_grid_search.fit(x_all, y_all)

        final_regressor = RFR(verbose=3, n_jobs=7)  # params in here

        best_model = Pipeline([('scaler', StandardScaler()),
                               ('pca', PCA(n_components=result.best_params_['pca__n_components'])),
                               ("final_regressor", final_regressor)])

        best_model.fit(x_all, y_all)

        y_predict_test = best_model.predict(x_test)
        y_predict_train = best_model.predict(x_all)

        r2_train = r2_score(y_all, y_predict_train)
        mae_train = mean_absolute_error(y_all, y_predict_train)
        rmse_train = mean_squared_error(y_all, y_predict_train, squared=False)
        r2_test = r2_score(y_test, y_predict_test)
        mae_test = mean_absolute_error(y_test, y_predict_test)
        rmse_test = mean_squared_error(y_test, y_predict_test, squared=False)

        file = open('./rf_dim_reduction_best.csv', 'a')
        if ligament == 0:
            file.writelines("Holdout trial scores: ID;r2_train;r2_test;mae_test;mae_train;rmse_test;rmse_train\n")
        file.writelines(f"Ligament attribute: {ligament_headers[ligament]}\n")
        file.writelines(f"best hypermodel: {result.best_params_}\nbest hypermodel rmse: {-result.best_score_}\n")
        file.writelines(f"Holdout trial scores: {ligament_headers[ligament]};{r2_train};{r2_test};{mae_test};{mae_train};{rmse_test};{rmse_train}\n\n")
        file.close()

        #cvResultsFile = open('./rf_di_reduction_cvResults.csv', 'a')
        #cvResultsFile.writelines(result.cv_results_)
        #file.close()
        resdf = pd.DataFrame(result.cv_results_)
        resdf.to_csv('./rf_di_reduction_cvResults.csv', mode='a')


test_dimensionality_reduction(ligaments_range=(0, 8, 1))
