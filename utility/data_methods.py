import pandas
import numpy
import csv
from pathlib import Path

'''
Given a project absolute path to a CSV file, outputs a dataframe with floating point elements of
certain precision. Precision parameter is OPTIONAL, and defaults to 64.
'''
def load_final_data_into_dataframe(path_to_data, precision=numpy.float64):
    return pandas.read_csv(path_to_data).astype(precision)


'''
Given a project absolute path to a directory with original knee data, loads a number of files and
returns a numpy array. Number of files is OPTIONAL, and defaults to 8. Element precision is OPTIONAL
and defaults to 64. Expects a trailing slash at the end of the directory path. Default (and max) precision
is 16 digits (float64).
'''
def load_multiple_files_into_numpy_array(path_to_directory, number_of_files=8, precision=numpy.float64):
    # 6561328 is the size of all datafiles.
    data = numpy.zeros((6561328, 31), dtype=precision)
    count = 0
    for x in range(0, number_of_files):
        with open(f"{path_to_directory}data" + str(x + 1) + ".csv") as csvfile:
            print("Loading original data file " + csvfile.name)
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                data[count] = row
                count = count + 1
    print("Finished loading " + str(number_of_files) + "knee data files.")
    return data
