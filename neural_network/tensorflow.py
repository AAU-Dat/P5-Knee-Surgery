import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

train_df = pd.read_csv("../data_processing/final_final_final.csv", index_col=0)

print(train_df.head())

model = keras.Sequential([keras.layers.Dense()])
