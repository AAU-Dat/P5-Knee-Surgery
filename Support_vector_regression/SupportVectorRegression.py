import pandas as pd
import sklearn.svm as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Constants to change style in graph
MarkerSize = 0.1
DotColor = 'Blue'

# Constants to change x & y axises in the saved graphs
k_x_left     = -2500
k_x_right    = 30000
k_y_bottom   = -2500
k_y_top      = 30000

epsr_x_left   = -0.10
epsr_x_right  = 0.30
epsr_y_bottom = -0.10
epsr_y_top    = 0.30


def gives_header_array():
    x = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
    for i in range(1, 24):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x


def print_graph(target, prediction, target_index, on_what_data):
    '''
    things that change in each graph
    title
    data
    model prediction
    y aksen
    '''
    # set data
    plt.figure()
    plt.scatter(target, prediction, color=DotColor, s=MarkerSize)

    # set names
    plt.title(f'{header[target_index]} {on_what_data} model')
    plt.xlabel(f'Actual {header[target_index]} value')
    plt.ylabel(f'Predicted {header[target_index]} value')

    # set axies
    # plt.xlim(x_left, x_right)
    # plt.ylim(y_bottom, y_top)

    # saves/shows graph
    plt.show()
    # plt.savefig('./multiple-linear-regression-figures/prediction_test_' f'{header[target_index]}.png')

def make_SVR_graph(target_index):
    # find relevant data
    x = df[header[9:285]]
    y = df[header[target_index]]

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, shuffle=True)

    # making the model and fitting the model to the data
    linear_svr_model = sk.LinearSVR()
    linear_svr_model.fit(x_train, y_train)

    # predicting results with both test and train
    predictions_train = linear_svr_model.predict(x_train)
    predictions_test = linear_svr_model.predict(x_test)

    # plotting the graph
    print_graph(y_train, predictions_train, target_index, 'train')
    print_graph(y_test, predictions_test, target_index, 'test')


# importing data
df = pd.read_csv('../data_processing/final_final_final.csv', index_col=0)
header = gives_header_array()
for x in range(0, 8):
    make_SVR_graph(x)



# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR
