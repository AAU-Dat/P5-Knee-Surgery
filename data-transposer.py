import csv
import numpy as np

data = np.zeros((377982, 20))

with open("./temp_knee_data" + ".csv") as csvfile:
    print(csvfile.name + " is opened.")
    csvreader = csv.reader(csvfile)

    #populate matrix with temp_knee_data
    count = 0
    for row in csvreader:
        data[count] = row;
        count = count + 1

    i = 2

