import pandas as pd
import numpy as np
import sklearn.svm as sk
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline


def gives_header_array():
    header_array = ['0', 'ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
    for i in range(1, 24):
        header_array.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return header_array


def make_svr_graph(target_index, test_size, max_iter, c):
    # find relevant data
    x = df[header[9:285]] #285 is max
    y = df[header[target_index]]

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=1-test_size, test_size=test_size, shuffle=True)

    pipe = Pipeline([('scaler', StandardScaler()), ('svc', sk.LinearSVR(max_iter=100, C=c))])
    pipe.fit(x_train, y_train)

    predictions_train = pipe.predict(x_train)
    predictions_test = pipe.predict(x_test)

    return r2_score(y_test, predictions_test)


def test(using_k, start, end, steps, header_start):
    for x in range(header_start, 9, 2):
        print(header[x])
        res = []
        res.append('c\tr2')
        for i in range(start, end, steps):
            print("\r" f'{i} / {end}', end='')
            if using_k:
                res.append(str(i) + '\t' + str(make_svr_graph(x, test_size, i*5000, i)) + '\n')
            else:
                res.append(str(i/100) + '\t' + str(make_svr_graph(x, test_size, 500, i/100)) + '\n')

        file = open(f'{header[x]}.csv', 'w')
        file.writelines(res)
        file.close()


df = pd.read_csv('../../data_processing/final_final_final.csv')
header = gives_header_array()
test_size = 0.2

start_k = 50
end_k = 1050
steps_k = 50

start_espr = 5
end_espr = 100
steps_espr = 5

k = 1
espr = 2

test(True, start_k, end_k, steps_k, k)
test(False, start_espr, end_espr, steps_espr, espr)
