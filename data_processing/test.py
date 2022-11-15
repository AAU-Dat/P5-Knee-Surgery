import pandas as pd


old = pd.read_csv('../data_processing/final_final_final.csv')
new = pd.read_csv('../data_processing/processed_data.csv')


print(old.equals(new))