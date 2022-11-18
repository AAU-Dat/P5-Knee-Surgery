import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# gives header names of the data as is read as a string array
def gives_header_array():
    header_array = ['0', 'ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
    for i in range(1, 24):
        header_array.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return header_array


# settings for print_a_graph
# Change dots in graph
MarkerSize = 0.1
DotColor = 'Blue'

# settings for the line in the graph for each method
# example
# line_settings = [[], [0, 20000], [-0.25, 0.25], [0, 20000], [-0.25, 0.25], [0, 20000], [-0.25, 0.25], [0, 20000], [-0.25, 0.25]]
line_settings = [[], [], [], [], [], [], [], []]

# Constants for the x axis in the graph
# example
# x_axis_settings = [[], [0, 30000], [-0.5, 0.5], [0, 30000], [-0.5, 0.5], [0, 30000], [-0.5, 0.5], [0, 30000], [-0.5, 0.5]]
x_axis_settings = [[], [], [], [], [], [], [], [], []]

# Constants for the y axis in the graph
# example
# y_axis_settings = [[], [0, 30000], [-0.5, 0.5], [0, 30000], [-0.5, 0.5], [0, 30000], [-0.5, 0.5], [0, 30000], [-0.5, 0.5]]
y_axis_settings = [[], [], [], [], [], [], [], [], []]


# prints a graph with the given data for the given targetindex of header at the savestring variable
def print_a_graph(target_data, prediction_data, target_index, save_string):
    # set data
    header = gives_header_array()
    plt.figure()
    plt.scatter(target_data, prediction_data, color=DotColor, s=MarkerSize)
    plt.plot(line_settings[target_index], line_settings[target_index], color='red')

    # set names
    plt.title(f'{header[target_index]} model')
    plt.xlabel(f'Actual {header[target_index]} value')
    plt.ylabel(f'Predicted {header[target_index]} value')

    # set axies
    plt.xlim(x_axis_settings[taget_index][1], x_axis_settings[taget_index][2])
    plt.ylim(y_axis_settings[taget_index][1], y_axis_settings[taget_index][2])

    # saves/shows graph
    # plt.show()
    plt.savefig(save_string)
    plt.close()


# pending on how we want to write to the file
# save the results of r2 rmse mae in a file at save_string
def save_results(target_index, target_data, predicted_data, save_string):
    header = gives_header_array()

    r2 = r2_score(target_data, predicted_data)
    rmse = mean_squared_error(target_data, predicted_data, squared=False)
    mae = mean_absolute_error(target_data, predicted_data)

    file = open(save_string, 'a')
    if target_index == 1:
        file.write('name;R2;RMSE;MAE\n')
    file.write(f'{header[target_index]};{r2};{rmse};{mae}\n')
    file.close()


# settings for train_val_test_split

train_ratio = 0.7
test_ratio = 0.1
val_ratio = 0.2
random_seed = 69

# returns the train val test split for all methods
def train_val_test_split(data):
    header = gives_header_array()
    x = data[header[9:285]]
    y = data[header[target_index]]

    # Train gets the train_ratio of the data set (train = 80%, test = 20% - af det fulde datasæt)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio, random_state=random_seed)

    # Both Validation and Test get 50% each of the remainder (val = 10%, test = 10% - af det fulde datasæt)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + val_ratio), random_state=random_seed)

    return x_train, y_train, x_val, y_val, x_test, y_test
# example of how to use it
# x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(your_dataframe)
