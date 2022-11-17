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

df = pd.read_csv('../data_processing/final_final_final.csv')
ligament_headers = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
rfr_criterion = ["squared_error", "poisson"]

rand = np.random.RandomState(69)


def gives_x_all_param_header():
    x = []
    for i in range(1, 24):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x


x = df[gives_x_all_param_header()]
y = df[ligament_headers[0]]  # ACL_k
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=rand)

train_ratio = 0.8
test_ratio = 0.1
validation_ratio = 0.1
# Train gets the train_ratio of the data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)
# Both Validation and Test get 50% each of the remainder
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio))
x, y = np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val))

RFRegressor = RFR()
pipe = Pipeline([('scaler', StandardScaler()), ("RFRegressor", RFRegressor)])

n_estimators = [int(x) for x in np.linspace(start=1, stop=50, num=50)] # number of trees in the random forest
max_features = ['sqrt'] # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=10)] # maximum number of levels allowed in each decision tree
min_samples_split = [2, 4, 6, 8, 10] # minimum sample number to split a node
min_samples_leaf = [1, 2, 3, 4, 5] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False]

random_grid = {'RFRegressor__n_estimators': n_estimators,
               'RFRegressor__max_features': max_features,
               'RFRegressor__max_depth': max_depth,
               'RFRegressor__min_samples_split': min_samples_split,
               'RFRegressor__min_samples_leaf': min_samples_leaf,
               'RFRegressor__bootstrap': bootstrap}

scoring = {"rmse": "neg_root_mean_squared_error"}

rf_randomSearch = RandomizedSearchCV(estimator=pipe, param_distributions=random_grid, scoring="neg_root_mean_squared_error", n_iter=1, cv=2, verbose=3, random_state=69, n_jobs=7)
result = rf_randomSearch.fit(x, y)

print("we got this far...")
print(-result.best_score_)
print("from these results...")
print(result.best_params_)

# Now make single forest with the best parameters.
# Then get the best params
print("right before final_regressor")
final_regressor = RFR(
   n_estimators=result.best_params_["RFRegressor__n_estimators"],
    max_features=result.best_params_['RFRegressor__max_features'],
    max_depth=result.best_params_['RFRegressor__max_depth'],
    min_samples_split=result.best_params_['RFRegressor__min_samples_split'],
    bootstrap=result.best_params_['RFRegressor__bootstrap'],
    verbose=3, n_jobs=7
) # params in here

print("after making final_regressor")
final_regressor.fit(x, y)
print("after fitting final_regressor")

y_predict_test = final_regressor.predict(x_test)
y_predict_train = final_regressor.predict(x)

r2_train = r2_score(y, y_predict_train)
mae_train = mean_absolute_error(y, y_predict_train)
rmse_train = mean_squared_error(y, y_predict_train, squared=False)
r2_test = r2_score(y_test, y_predict_test)
mae_test = mean_absolute_error(y_test, y_predict_test)
rmse_test = mean_squared_error(y_test, y_predict_test, squared=False)

print(r2_train, r2_test, mae_test, mae_train, rmse_test, rmse_train)