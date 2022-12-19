import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


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


def column_gen(target):
    columns = []
    for i in range(len(hyperparameters_renaming)):
        columns.append(hyperparameters_renaming[i][1])
    columns.append(caption_headers[target])
    return columns


def invert(df):
    for i in range(len(df['mean_test_score'])):
        df['mean_test_score'][i] *= -1


def df_gen(target):
    df = pd.read_csv(f'models\\{target}\\{target}_GridsearchCV_Results.csv')
    change_df_column_names(df)
    invert(df)
    df.rename(columns={'mean_test_score': f'{caption_headers[target]}'}, inplace=True)
    return df[column_gen(target)]


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


def create_and_save_graph(values, rmse, acl, pcl, mcl, lcl):
    fig, ax = plt.subplots()
    ax.scatter(rmse, acl, s=5, c='b', alpha=alpha, label=f'acl')
    ax.scatter(rmse, pcl, s=5, c='r', alpha=alpha, label='pcl')
    ax.scatter(rmse, mcl, s=5, c='g', alpha=alpha, label='mcl')
    ax.scatter(rmse, lcl, s=5, c='orange', alpha=alpha, label='lcl')

    # Add title, axes labels and a legend to the plot
    plt.title(f"{method_name} accuarcy")
    plt.xlabel(f'{hyperparameters_renaming[0][1]}')
    plt.ylabel('rmse')
    plt.legend()

    # Save and close out the graph, to avoid them cumulating
    #plt.show()
    plt.savefig(f"{method_name}_{values}_graph.png", dpi=300)
    plt.close()


headers = gives_header_array()
caption_headers = {'ACL_k': 'ACL\_k',
         'ACL_epsr': 'ACL\_epsr',
         'PCL_k': 'PCL\_k',
         'PCL_epsr': 'PCL\_epsr',
         'MCL_k': 'MCL\_k',
         'MCL_epsr': 'MCL\_epsr',
         'LCL_k': 'LCL\_k',
         'LCL_epsr': 'LCL\_epsr'}


style.use('seaborn-v0_8')
alpha = 0.5

hyperparameters_renaming = [['param_svc__C', 'C']]
columns_in_table = 5
method_name = 'SVR'
search_method_name = 'gridsearch'


acl_epsr = df_gen(headers[1])
lcl_epsr = df_gen(headers[7])
mcl_epsr = df_gen(headers[5])
pcl_epsr = df_gen(headers[3])
epsr_table = pd.merge(pd.merge(pd.merge(acl_epsr, lcl_epsr, on=hyperparameters_renaming[0][1]), mcl_epsr, on=hyperparameters_renaming[0][1]), pcl_epsr, on=hyperparameters_renaming[0][1])
table_gen(epsr_table, 'epsr')
create_and_save_graph('epsr', epsr_table[hyperparameters_renaming[0][1]], epsr_table[caption_headers['ACL_epsr']], epsr_table[caption_headers['PCL_epsr']], epsr_table[caption_headers['MCL_epsr']], epsr_table[caption_headers['LCL_epsr']])

acl_k = df_gen(headers[0])
lcl_k = df_gen(headers[6])
mcl_k = df_gen(headers[4])
pcl_k = df_gen(headers[2])
k_table = pd.merge(pd.merge(pd.merge(acl_k, lcl_k, on=hyperparameters_renaming[0][1]), mcl_k, on=hyperparameters_renaming[0][1]), pcl_k, on=hyperparameters_renaming[0][1])
table_gen(k_table, 'k')
create_and_save_graph('k', k_table[hyperparameters_renaming[0][1]], k_table[caption_headers['ACL_k']], k_table[caption_headers['PCL_k']], k_table[caption_headers['MCL_k']], k_table[caption_headers['LCL_k']])



