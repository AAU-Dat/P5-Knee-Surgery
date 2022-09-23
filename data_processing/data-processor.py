import numpy as np
import csv


# Loads raw data from data files 1 to 8
def load_data(stopfile):
    data = np.zeros((6561328, 31))
    count = 0
    for x in range(0, stopfile):
        with open("./raw_data/data" + str(x + 1) + ".csv") as csvfile:
            print(csvfile.name)
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                data[count] = row
                count = count + 1
    return data


# Remove the last columns containing metadata - 20 to 31
def remove_metadata(data_with_metadata):
    return np.delete(data_with_metadata, slice(20, 31), 1)


def remove_incomplete_data(data):
    currently_checking_for = data[0, 0]
    succession_start = 0
    row_count_on_complete_dataset = 0
    complete_data = np.zeros((1541000, 20))
    for x in range(0, len(data)):
        if currently_checking_for == data[x, 0]:
            if x - succession_start == 22:
                complete_data[row_count_on_complete_dataset:row_count_on_complete_dataset+23, 0:20] = data[succession_start:x+1, 0:20]
                row_count_on_complete_dataset += 23
        else:
            currently_checking_for = data[x][0]
            succession_start = x
        # else
            # update currently_checking_for
            # succession_start = x
    return complete_data
    # return complete_data

raw_data = load_data(1)
data_without_metadata = remove_metadata(raw_data)
complete_data = remove_incomplete_data(data_without_metadata)
print("hej")

'''
complete_data[row_count_on_complete_dataset][0:result_columns] = data[x][0:result_columns]
# check efter herfra
for y in range(23, 0, -1):
    complete_data[row_count_on_complete_dataset][8 + 12 * (23 - y):8 + 12 * (23 - y) + 12] = data[x - y][8:20]
row_count_on_complete_dataset += 1
'''