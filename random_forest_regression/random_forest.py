import scipy as sp
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from time import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedKFold, ShuffleSplit
from scipy.stats import randint
import pandas as pd
import numpy as np
#import tensorflow as tf

#Program skal bygges under keras for at vi kan bruge tensorflow. Se jamie branch.
#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

df = pd.read_csv('../data_processing/final_final_final.csv')
ligament_headers = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
rfr_criterion = ["squared_error", "poisson"]

def gives_x_all_param_header():
    x = []
    for i in range(1, 24):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x

def write_results_to_file(r_2, mae, mse, estimators, max_features, ligament):
    file = open("random_forest_results.txt", "a")
    file.write(f'r_2: {r_2}, MAE: {mae}, MSE: {mse}, maxfeatures: {max_features}, estimators: {estimators}, ligament: {ligament}\n')
    file.close()

def print_status(estimators, max_features, ligament):
    print(f'Finished max_features= {max_features}, estimators= {estimators} ligament={ligament}')

#
# retain best entries for best values:
#
#def evaluate_best_config(estimators, max_features, ligament, r_2, mae, mse):


def save_to_list(r_2, mae, mse, estimators, max_features, ligament):
    return {"r_2": r_2, "mae": mae, "mse": mse, "estimators": estimators, "max_features": max_features, "ligament": ligament}

# åh ja returnerer ikke noget haha!
def write_best_scores_for_all_knees_to_file(list_of_result_dictionaries):
    #
    # iterate entries in dictionary
    # log the number of the entry with higher-than-before r2, or mae, or mse
    # write the 3 entries to file in an understandable manner
    #
    highest_r2, lowest_mae, lowest_mse = 0, float('inf'), float('inf')
    for e in list_of_result_dictionaries:
        if e.get("r_2") > highest_r2:
            highest_r2 = e.get("r_2")
            highest_r2_record = e
        if e.get("mae") < lowest_mae:
            lowest_mae = e.get("mae")
            lowest_mae_record = e
        if e.get("mse") < lowest_mse:
            lowest_mse = e.get("mse")
            lowest_mse_record = e

    file = open("random_forest_results.txt", "a")
    file.write(f'Highest r_2: {highest_r2_record}\nHighest MAE: {lowest_mae_record}\nHighest MSE: {lowest_mse_record}\n\n')
    file.close()



def random_forest_all_parameters(estimators, ligaments):
    x = df[gives_x_all_param_header()]
    list_of_results = list(dict())

    for l in range(ligaments):
        y = df[ligament_headers[l]]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)
        for i in range(1, estimators+1):
            for j in range(1, 12):
                regressor = RFR(n_estimators=i, max_features=0.45+(j*0.05))
                regressor.fit(x_train, y_train)
                y_pred = regressor.predict(x_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                #write_results_to_file(r_2=r2, mae=mae, mse=mse, estimators=i, max_features=0.45+(j*0.05), ligament=ligament_headers[l])
                list_of_results.append(save_to_list(r_2=r2, mae=mae, mse=mse, estimators=i, max_features=0.45+(j*0.05), ligament=ligament_headers[l]))
                print_status(max_features=0.45+(j*0.05), estimators=i, ligament=ligament_headers[l])

    # List er tom? wtf 😒
    write_best_scores_for_all_knees_to_file(list_of_results)

random_forest_all_parameters(1, 1)

'''
# Make parameters for random search.
parameters_range = {"n_estimators": range(1, 2),
                    "max_features": np.arange(0.5, 1.05, 0.05)}

scoring = {"negative MAE": "neg_mean_absolute_error",
           "r_2": "r2",
           "negative MSE": "neg_mean_squared_error"}

# Create regressor with standard settings.
regressor = RFR()

# Define how many hyperparameter-configurations random search tries.
randomsearch_iterations = 1  # how many configurations are we trying out

#cross_validation = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)

# Create random-search object with the attributes defined above.
randomsearch = RandomizedSearchCV(regressor,
                                  param_distributions=parameters_range,
                                  n_iter=randomsearch_iterations,
                                  scoring=scoring,
                                  n_jobs=1,
                                  refit="negative MSE",
                                  cv=ShuffleSplit(test_size=0.20, n_splits=1, random_state=0))

# Define x as all machine headers.
x = df[gives_x_all_param_header()]

# Define y as ACL_k.
y = df[ligament_headers[0]]

# Create 80-20 test-train split for machine data and ACL_k.
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# Save the time for right before random-search starts
start_time = time()

# Start the random-search. Its parameters were defined above.
result = randomsearch.fit(x, y)

# Report back the results of random-search.
print("Random-search took %.2f seconds for %d configurations of parameter settings." % ((time() - start_time),
                                                                                        randomsearch_iterations))

# can do best_score, best_params etc.
print('Best estimator across all params:\n', randomsearch.best_estimator_)
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

'''

