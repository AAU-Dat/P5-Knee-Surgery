import csv
import numpy as np
import math

data = np.zeros((2368149, 20))

with open("./temp_knee_data_1_to_8.csv") as csvfile:
    print(csvfile.name + " is opened.")
    csvreader = csv.reader(csvfile)

    #populate matrix with temp_knee_data
    count = 0
    for row in csvreader:
        data[count] = row
        count = count + 1

final_array = np.zeros((102963, 284))
#shitData = np.zeros(((23, 16434, 20)))
for i in range(0, 2368149, 23):
    final_array[math.ceil(i/23)][0:8] = data[i][0:8]
    for x in range(0, 23):
        final_array[math.ceil(i/23)][8+12*x:20+12*x] = data[i][8:20]
#        shitData [x] [math.ceil(i/23)] [0:20] = data[i+2][0:20]

non_duped = np.unique(final_array, axis=0)
poo = 2
transposed_knee_data = open("transposed_knee_data_1_to_8.csv", "w")
np.savetxt(transposed_knee_data, non_duped, newline="\n", delimiter=",", fmt="%13f")
