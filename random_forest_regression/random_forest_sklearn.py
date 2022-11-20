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
#from fast_ml.utilities import display_all
#from fast_ml.feature_selection import get_constant_features

df = pd.read_csv('../data_processing/final_final_final.csv').astype(np.float32)
ligament_headers = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
rfr_criterion = ["squared_error", "poisson"]

def gives_x_all_param_header():
    x = []
    for i in range(1, 24):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x


#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=rand)

rand = np.random.RandomState(69)

def main(ligaments_range):
    for ligament in range(*ligaments_range):
        x = df[gives_x_all_param_header()]
        y = df[ligament_headers[ligament]]

        train_ratio = 0.8
        test_ratio = 0.1
        validation_ratio = 0.1
        # Train gets the train_ratio of the data set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio, random_state=69)
        # Both Validation and Test get 50% each of the remainder
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                        test_size=test_ratio / (test_ratio + validation_ratio),
                                                        random_state=69)
        x_all, y_all = np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val))

        RFRegressor = RFR()
        pipe = Pipeline([('scaler', StandardScaler()), ("RFRegressor", RFRegressor)])

        n_estimators = [int(x) for x in np.linspace(start=20, stop=150, num=int(130 / 5))]
        max_features = [float(x) for x in np.linspace(start=0.001, stop=1.0, num=50)]
        max_depth = [int(x) for x in np.linspace(start=10, stop=120, num=int(110 / 5))]
        min_samples_split = [int(x) for x in np.linspace(start=2, stop=20, num=19)]  # keep in here
        min_samples_leaf = [int(x) for x in np.linspace(start=1, stop=20, num=20)]  # reduced to 20

        random_grid = {'RFRegressor__n_estimators': n_estimators,
                       'RFRegressor__max_features': max_features,
                       'RFRegressor__max_depth': max_depth,
                       'RFRegressor__min_samples_split': min_samples_split,
                       'RFRegressor__min_samples_leaf': min_samples_leaf}

        rf_randomSearch = RandomizedSearchCV(estimator=pipe, param_distributions=random_grid, scoring="neg_root_mean_squared_error", n_iter=50, cv=5, verbose=3, n_jobs=-1)
        result = rf_randomSearch.fit(x_all, y_all)

        final_regressor = RFR(
            n_estimators=result.best_params_["RFRegressor__n_estimators"],
            max_features=result.best_params_['RFRegressor__max_features'],
            max_depth=result.best_params_['RFRegressor__max_depth'],
            min_samples_split=result.best_params_['RFRegressor__min_samples_split'],
            bootstrap=result.best_params_['RFRegressor__min_samples_leaf'],
            verbose=3, n_jobs=7
        )  # params in here

        final_pipe = Pipeline([('scaler', StandardScaler()), ("final_regressor", final_regressor)])
        final_pipe.fit(x_all, y_all)

        y_predict_test = final_pipe.predict(x_test)
        y_predict_train = final_pipe.predict(x_all)

        r2_train = r2_score(y_all, y_predict_train)
        mae_train = mean_absolute_error(y_all, y_predict_train)
        rmse_train = mean_squared_error(y_all, y_predict_train, squared=False)
        r2_test = r2_score(y_test, y_predict_test)
        mae_test = mean_absolute_error(y_test, y_predict_test)
        rmse_test = mean_squared_error(y_test, y_predict_test, squared=False)

        file = open('./random_search_results_rfr.csv', 'a')
        if ligament == 0:
            file.writelines("Holdout trial scores: ID;r2_train;r2_test;mae_test;mae_train;rmse_test;rmse_train\n")
        file.writelines(f"Ligament attribute: {ligament_headers[ligament]}\n")
        file.writelines(f"best hypermodel: {result.best_params_}\nbest hypermodel rmse: {-result.best_score_}\n")
        file.writelines(f"Holdout trial scores: {ligament_headers[ligament]};{r2_train};{r2_test};{mae_test};{mae_train};{rmse_test};{rmse_train}\n\n")
        file.close()

        #cvResultsFile = open('./random_search_cvResults.csv', 'a')
        #cvResultsFile.writelines(result.cv_results_)
        #file.close()
        resdf = pd.DataFrame(result.cv_results_)
        resdf.to_csv('./random_search_cvResults.csv', mode='a')


main(ligaments_range=(0, 8, 1))
