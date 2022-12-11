import pandas as pd
import numpy as np


def gives_header_array():
    return ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']


def gives_caption_header_array():
    return ['ACL\_k', 'ACL\_epsr', 'PCL\_k', 'PCL\_epsr', 'MCL\_k', 'MCL\_epsr', 'LCL\_k', 'LCL\_epsr']


def left_side_columns():
    columns = 'l'
    for i in range(columns_in_table):
        columns += 'l'
    return columns


def change_df_column_names(dataframe):
    for i in range(len(hyperparameters_renaming)):
        dataframe.rename(columns={hyperparameters_renaming[i][0]: hyperparameters_renaming[i][1]}, inplace=True)


def df_gen(target, ):
    df = pd.read_csv(f'models\\{target}\\{target}_GridsearchCV_Results.csv')
    change_df_column_names(df)
    df.rename(columns={'mean_test_score': f'{caption_headers[target]}'}, inplace=True)
    return df[["C", f"{caption_headers[target]}"]]


def table_gen(df, values):
    str = df.style.hide(axis="index").to_latex(column_format=left_side_columns(),
                                                  hrules=True,
                                                  caption=f"results from {method_name} {search_method_name} on {values}",
                                                  label=f"table:{method_name}_{search_method_name}_results_{values}",
                                               position_float='centering')
    file = open(f"{method_name}_table_{values}.csv", 'w')
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
hyperparameters_renaming = [['param_svc__C', 'C']]
columns_in_table = 5
method_name = 'SVR'
search_method_name = 'gridsearch'

acl_epsr = df_gen(headers[1])
lcl_epsr = df_gen(headers[7])
mcl_epsr = df_gen(headers[5])
pcl_epsr = df_gen(headers[3])
epsr_table = pd.merge(pd.merge(pd.merge(acl_epsr, lcl_epsr, on='C'), mcl_epsr, on='C'), pcl_epsr, on='C')
table_gen(epsr_table, 'epsr')

acl_k = df_gen(headers[0])
lcl_k = df_gen(headers[6])
mcl_k = df_gen(headers[4])
pcl_k = df_gen(headers[2])
k_table = pd.merge(pd.merge(pd.merge(acl_k, lcl_k, on='C'), mcl_k, on='C'), pcl_k, on='C')
table_gen(k_table, 'k')



