import numpy as np
import csv
import math
import pandas as pd


min_knee_measurements = 23

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


def remove_incomplete_meassurements(data):
    currently_checking_for = data[0, 0]
    succession_start = 0
    complete_data = []

    for x in range(0, len(data)):
        if currently_checking_for == data[x, 0]:
            if x - succession_start == 22:
                complete_data.extend(data[succession_start:x + 1])
        else:
            currently_checking_for = data[x][0]
            succession_start = x
    return complete_data


def transpose_data1(data):
    # 102963 is the amount of rows after we rearange the data
    reranged_data = np.zeros((102963, 284))
    for i in range(0, 2368149, 23):
        reranged_data[math.ceil(i / 23)][0:8] = data[i][0:8]
        for x in range(0, 23):
            reranged_data[math.ceil(i / 23), 8 + 12 * x:20 + 12 * x] = data[i + x, 8:20]
    print("Done with transpose_data()")
    return reranged_data


def transpose_data2(data):
    reranged_data = np.zeros((int(len(data)/min_knee_measurements), 284))
    for i in range(0, len(data), min_knee_measurements):
        temp_i = int(i / min_knee_measurements)
        reranged_data[temp_i][0:8] = data[i][0:8]
        for x in range(0, 23):
            reranged_data[temp_i][8 + 12 * x:20 + 12 * x] = data[i + x][8:20]
    print("Done with transpose_data()")
    return reranged_data


def headers():
    header = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
    for i in range(1, 24):
        header.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                       'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                       'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    print("Done with makes_header_in_file()")
    return header


def headers_minor():
    header = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
    for i in range(1, 2):
        header.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                       'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                       'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    print("Done with makes_header_in_file()")
    return header

def data_processing():
    data = load_data(8)
    data = np.delete(data, slice(20, 31), 1)
    data = remove_incomplete_meassurements(data)
    df1 = pd.DataFrame(data, columns=headers_minor())
    data = transpose_data2(data)
    df = pd.DataFrame(data, columns=headers())
    df = df.drop_duplicates()
    print('hej')
    df.to_csv('processed_data.csv')

data_processing()


'''
# set header
df = pd.DataFrame(transpose_data(), columns=makes_header_in_file())
df = df.drop_duplicates()
print('Duplicates is removed')

# Write to file
df.to_csv('final_final_final.csv')
print("The final_final_final.csv is added to raw_data")
'''