import pandas as pd
import numpy as np
import sklearn.svm as sk
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing


def gives_header_array():
    header_array = ['0', 'ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
    for i in range(1, 24):
        header_array.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return header_array


def make_svr_graph(target_index, test_size, print_graph=False):
    # find relevant data
    x = df[header[9:285]] #285 is max
    y = df[header[target_index]]

    # normalizing the input data
    # x = preprocessing.normalize(x)

    # scaler = StandardScaler()
    # scaler.fit(x)
    # x = scaler.transform(x)

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=1-test_size, test_size=test_size, shuffle=True)

    # making the model and fitting the model to the data
    linear_svr_model = sk.LinearSVR(max_iter=10000, C=50)
    linear_svr_model.fit(x_train, y_train)

    # predicting results with both test and train
    predictions_train = linear_svr_model.predict(x_train)
    predictions_test = linear_svr_model.predict(x_test)

    return r2_score(y_test, predictions_test)




df = pd.read_csv('../data_processing/final_final_final.csv')
header = gives_header_array()

print(make_svr_graph(1, 0.2))
