import numpy as np
import csv
import math
import pandas as pd


files = 1
min_knee_measurements = 23
results = 8
measurements = 12


# Loads raw data from data files 1 to 8
def load_data(stopfile):
    # 6561328 is the size of all datafiles
    data = np.zeros((6561328, 31))
    count = 0
    for x in range(0, stopfile):
        with open("./raw_data/data" + str(x + 1) + ".csv") as csvfile:
            print(csvfile.name)
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                data[count] = row
                count = count + 1
    print("Done with load_data()")
    return data


def remove_incomplete_measurements(data):
    currently_checking_for = data[0, 0]
    succession_start = 0
    complete_data = []

    for x in range(0, len(data)):
        if currently_checking_for == data[x, 0]:
            if x - succession_start == min_knee_measurements-1:
                complete_data.extend(data[succession_start:x + 1])
        else:
            currently_checking_for = data[x][0]
            succession_start = x
    print("Done with remove_incomplete_measurements()")
    return complete_data


def transpose_data(data):
    rearranged_data = np.zeros((math.ceil(len(data) / min_knee_measurements), results + (min_knee_measurements * measurements)))
    for i in range(0, len(data), min_knee_measurements):
        temp_i = int(i / min_knee_measurements)
        rearranged_data[temp_i][0:results] = data[i][0:results]
        for x in range(0, min_knee_measurements):
            rearranged_data[temp_i][results + measurements * x:results + measurements * (x + 1)] = data[i + x][results:results + measurements]
    print("Done with transpose_data()")
    return rearranged_data


def headers():
    header = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
    for i in range(1, 24):
        header.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                       'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                       'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return header


def data_processing():
    data = load_data(files)
    data = np.delete(data, slice(20, 31), 1)
    data = remove_incomplete_measurements(data)
    data = transpose_data(data)
    df = pd.DataFrame(data, columns=headers())
    print('Done with pd.DataFrame()')
    df = df.drop_duplicates()
    df.to_csv('processed_data.csv')


data_processing()
