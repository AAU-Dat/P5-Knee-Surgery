from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf

#Program skal bygges under keras for at vi kan bruge tensorflow. Se jamie branch.
#import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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


def random_forest_all_parameters(estimators, ligaments):
    x = df[gives_x_all_param_header()]
    y = df[ligament_headers]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)

    for i in range(1, estimators+1):
        for j in range(1, 12):
                for l in range(ligaments):
                    regressor = RFR(n_estimators=i, max_features=0.45+(j*0.05))
                    regressor.fit(x_train, y_train)
                    y_pred = regressor.predict(x_test)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)

                    write_results_to_file(r_2=r2, mae=mae, mse=mse, estimators=i, max_features=j, ligament=ligament_headers[l])
                    print_status(max_features=0.45+(j*0.05), estimators=i, ligament=ligament_headers[l])



random_forest_all_parameters(10, 1)



