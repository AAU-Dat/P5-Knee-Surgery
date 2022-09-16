import csv
import numpy as np
import math

data = np.zeros((377982, 20))

with open("./temp_knee_data" + ".csv") as csvfile:
    print(csvfile.name + " is opened.")
    csvreader = csv.reader(csvfile)

    #populate matrix with temp_knee_data
    count = 0
    for row in csvreader:
        data[count] = row
        count = count + 1

final_array = np.zeros((16434, 284))
shitData = np.zeros(((23, 16434, 20)))
for i in range(0, 377982, 23):
    final_array[math.ceil(i/23)][0:8] = data[i][0:8]
    for x in range(0, 23):
        final_array[math.ceil(i/23)][8+12*x:20+12*x] = data[i][8:20]
        shitData [x] [math.ceil(i/23)] [0:20] = data[i+2][0:20]
print("hi")
