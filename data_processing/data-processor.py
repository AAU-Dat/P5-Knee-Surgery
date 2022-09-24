import numpy as np
import csv
import math


# Loads raw data from data files 1 to 8
def load_data(stopfile):
    #6561328 is the size of all datafiles
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


#finds each case with 23 sets of messurements in a row
def remove_incomplete_data(data):
    #size of all complete data 2368149 to verify use 2368150 with debug and see if the last row returns zeroes
    currently_checking_for = data[0, 0]
    succession_start = 0
    row_count_on_complete_dataset = 0
    complete_data = np.zeros((2368149, 20))
    for x in range(0, len(data)):
        if currently_checking_for == data[x, 0]:
            if x - succession_start == 22:
                complete_data[row_count_on_complete_dataset:row_count_on_complete_dataset+23, 0:20] = data[succession_start:x+1, 0:20]
                row_count_on_complete_dataset += 23
        else:
            currently_checking_for = data[x][0]
            succession_start = x
    return complete_data


#this func doesn't work yet
def rerange_data(data):
    #102963 is the amount of rows after we rearange the data
    reranged_data = np.zeros((102963, 284))
    for i in range(0, 2368149, 23):
        reranged_data[math.ceil(i / 23), 0:8] = data[i, 0:8]
        for x in range(0, 23):
            reranged_data[math.ceil(i / 23), 8 + 12 * x:20 + 12 * x] = data[i+x, 8:20]
    return reranged_data


#think this works but extremely inefficient
def remove_dublicates(data):
    data_without_dublicates = np.zeros((102963, 284))
    data_without_dublicates_row_count = 0

    for x in range(0, len(data)):
        for y in range(0, data_without_dublicates_row_count):
            #this removes all lines that are totally equal but is very expensive in our case
            # all(data[x, 0:284] == data_without_dublicates[y, 0:284]
            if all(data[x, 0:8] == data_without_dublicates[y, 0:8]):
                break
        else:
            data_without_dublicates[data_without_dublicates_row_count, 0:284] = data[x, 0:284]
            data_without_dublicates_row_count += 1

    return data_without_dublicates





#load all data
raw_data = load_data(1)
#remove meta data
data_without_metadata = remove_metadata(raw_data)
#find all cases with 23 instances
complete_data = remove_incomplete_data(data_without_metadata)
#rearange data
reranged_data = rerange_data(complete_data)
#remove doublicates
data_without_dublicates = remove_dublicates(reranged_data)
#set header

print("hej")