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

df = pd.read_csv('../data_processing/final_final_final.csv').astype('float32')
ligament_headers = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
rfr_criterion = ["squared_error", "poisson"]
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1
random_generator = 69

def gives_x_all_param_header():
    x = []
    for i in range(1, 24):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x


def make_concatenated_split(df):
    # Read the data
    x = df[gives_x_all_param_header()]
    y = df[ligament_headers]
    # Train gets the train_ratio of the data set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio, random_state=69)
    # Both Validation and Test get 50% each of the remainder
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio),
                                                    random_state=69)
    x_all, y_all = np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val))
    return x_all, y_all, x_test, y_test


def get_scores(prediction, truth):
    r2 = r2_score(truth, prediction)
    mae = mean_absolute_error(truth, prediction)
    rmse = mean_squared_error(truth, prediction, squared=False)
    return {"r2": r2, "mae": mae, "rmse": rmse}


def write_scores_to_file(val_scores, test_scores, ligament, filepath, header="", make_header=False, ):
    file = open(filepath, 'a')
    if make_header == True:
        file.writelines(f"{header}\n")
    file.writelines(f"Ligament attribute: {ligament}\n")
    file.writelines(
        f"{ligament};{val_scores['r2']};{test_scores['r2']};{val_scores['mae']};"
        f"{test_scores['mae']};{val_scores['rmse']};{test_scores['rmse']}\n\n")
    file.close()


def test_default_tree(ligament, filepath, make_header=False):
    x_all, y_all, x_test, y_test = make_concatenated_split(df)
    validation_regressor = Pipeline([('scaler', StandardScaler()), ("RFRegressor", RFR(n_jobs=6, verbose=3))])
    validation_regressor.fit(x_all, y_all)

    y_predict_test = validation_regressor.predict(x_test)
    y_predict_train = validation_regressor.predict(x_all)
    train_test_scores = get_scores(prediction=y_predict_test, truth=y_test)
    train_val_scores = get_scores(prediction=y_predict_train, truth=y_all)

    write_scores_to_file(train_val_scores, train_test_scores, ligament, filepath, make_header=make_header,
                         header="ID;r2_train;r2_test;mae_test;mre_test;rmse_train;rmse_test")


# Writes default hyperparameter test scores for all 8 attributes.
ligament_being_tested = "ACL_k"#for ligament_being_tested in ligament_headers:
test_default_tree(ligament_being_tested, filepath="./default_scores.csv", make_header=True)








