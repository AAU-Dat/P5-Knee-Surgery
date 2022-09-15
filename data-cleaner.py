#row count = 6561328
#column count = 31

import pathlib
import csv
import numpy as np

data = np.zeros((6561328, 31))

### Step 1 - Loads every dataset into a big data set ###
for x in range(8):
    with open("./data/data" + str(x+1) + ".csv") as csvfile:

        print(csvfile.name)

        csvreader = csv.reader(csvfile)

        count = 0

        for row in csvreader:
            data[count] = row;
            count = count + 1
            #print(row[0])

### step 1.5 - remove the last 11 metadata rows
data = np.delete(data, slice(20, 31, 1), 1)

### Step 2 - Removes any redundent rows that don't have 23 rows ###
previous = data[0][0]
array_of_parameters = np.zeros((23, 20))
index = 1
temp_row = data[0]

# Create a file called temp.csv, if temp.csv already exists it override the previous file
temp_knee_data = open("temp_knee_data2.csv", "w")

for row in data:
    array_of_parameters[index] = row
    if row[0] == previous and row[0] != 0:  # how to determine 0-row?
        index += 1
        if index == 23:
            #print("Added knee data to array")
            array_of_parameters[0] = temp_row
            np.savetxt(temp_knee_data, array_of_parameters, newline="\n", delimiter=",", fmt="%13f")
            index = 1
    else:
        array_of_parameters = np.zeros((23, 20))
        index = 1
        temp_row = row
    # This row we are looking at, is stored in the previous row
    previous = row[0]



### Step 3: transpose data

### Step 4: remove duplicates (with numpy.unique likely)