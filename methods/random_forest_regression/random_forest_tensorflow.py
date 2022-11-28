import keras.callbacks
import math
import pandas as pd
import keras_tuner as kt
from keras import layers
from sklearn import metrics
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.regularizers import L2
from keras_tuner import Hyperband
from datetime import datetime
import tensorflow_decision_forests as tfdf

now = datetime.now().strftime("%d-%b-%Y at %H:%M")

ACL_k, ACL_epsr, PCL_k, PCL_epsr = 'ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr'
MCL_k, MCL_epsr, LCL_k, LCL_epsr = 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr'
df = pd.read_csv('../data_processing/final_final_final.csv', index_col=0)
result_columns = [ACL_k, ACL_epsr, PCL_k, PCL_epsr, MCL_k, MCL_epsr, LCL_k, LCL_epsr]

seed = 69                                       # to get reproducible data splits and hyperparameters
train_ratio = 0.80                              # percentage of the data set allocated to train and tune
validation_ratio = 0.10                         # percentage of the data set allocated to validation
test_ratio = 0.10                               # percentage of the data set allocated to evaluation

