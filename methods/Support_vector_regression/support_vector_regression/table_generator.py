import pandas as pd
import numpy as np


def gives_header_array():
    return ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']


def table_gen(target):
    df = pd.read_csv(f'models\\{target}\\{target}_GridsearchCV_Results.csv')
    table = df[["param_svc__C", "mean_test_score"]]
    print(table.to_latex(index=False))


headers = gives_header_array()
for i in range(1):
    table_gen(headers[i])

