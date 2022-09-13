#row count = 6561328
#column count = 31

import pathlib
import csv
import numpy as np

data = np.zeros((6561328, 31))

# Loads every dataset into a big data set
for x in range(1):
    with open("./data/data" + str(x+1) + ".csv") as csvfile:

        print(csvfile.name)

        csvreader = csv.reader(csvfile)

        count = 0

        for row in csvreader:
            data[count] = row;
            count = count + 1
            #print(row[0])

# Removes any redundent rows that don't have 23 rows
previous = data[0][0]
index = 0
array_of_parameters = np.zeros((23, 31))

# Create a file called temp.csv, if temp.csv already exsist it override the previous file
knee_data = open("temp.csv", "x")

for row in data:
    if row[0] == previous:
        array_of_parameters[index] = row
        index += 1
        if index == 22:
            print("Added knee data to file")
            for i in array_of_parameters:
                knee_data += array_of_parameters[i]
            index = 0
    else:
        index = 0
    # This row we are looking at, is stored in the previous row
    previous = row[0]
