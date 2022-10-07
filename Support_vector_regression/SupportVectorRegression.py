import pandas as pd
import numpy as np
import sklearn.svm as sk
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def gives_header_array():
    x = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
    for i in range(1, 24):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x


# importing data
df = pd.read_csv('../data_processing/final_final_final.csv', index_col=0)
header = gives_header_array()

x = df[header[0]]
y = df[header[9:285]]



#StandardScaler()

linear_svr_model = sk.LinearSVR()
linear_svr_model.fit(x, y)

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')



# https://www.youtube.com/watch?v=iSLJ6i-BpPg&ab_channel=CodeWarriors
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR
