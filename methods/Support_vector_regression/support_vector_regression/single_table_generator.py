import pandas as pd
import numpy as np


def gives_header_array():
    return ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']


def left_side_columns():
    columns = 'l'
    for i in range(len(hyperparameters_renaming)):
        columns += 'l'
    return columns


def change_df_column_names(dataframe):
    for i in range(len(hyperparameters_renaming)):
        dataframe.rename(columns={hyperparameters_renaming[i][0]: hyperparameters_renaming[i][1]}, inplace=True)


def table_gen(target):
    df = pd.read_csv(f'models\\{target}\\{target}_GridsearchCV_Results.csv')
    change_df_column_names(df)
    df.rename(columns={'mean_test_score': 'rmse'}, inplace=True)
    table = df[["C", "rmse"]]
    str = table.style.hide(axis="index").to_latex(column_format=left_side_columns(),
                                                  hrules=True,
                                                  caption=f"results from {method_name} {search_method_name} on {caption_headers[target]}",
                                                  label=f"table:{method_name}_{search_method_name}_results_{target}")
    file = open(f"{method_name}_table_{target}.csv", 'w')
    file.write(str)
    file.close()
    print()


headers = gives_header_array()
caption_headers = {'ACL_k': 'ACL\_k',
         'ACL_epsr': 'ACL\_epsr',
         'PCL_k': 'PCL\_k',
         'PCL_epsr': 'PCL\_epsr',
         'MCL_k': 'MCL\_k',
         'MCL_epsr': 'MCL\_epsr',
         'LCL_k': 'LCL\_k',
         'LCL_epsr': 'LCL\_epsr'}

# fit these variables to your method
method_name = 'SVR'
search_method_name = 'gridsearch'
hyperparameters_renaming = [['param_svc__C', 'C']]






for i in range(8):
    table_gen(headers[i])

