import pandas as pd
import numpy as np


def gives_header_array():
    return ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']


def gives_caption_header_array():
    return ['ACL\_k', 'ACL\_epsr', 'PCL\_k', 'PCL\_epsr', 'MCL\_k', 'MCL\_epsr', 'LCL\_k', 'LCL\_epsr']



def table_gen(target, caption_target):
    df = pd.read_csv(f'models\\{target}\\{target}_GridsearchCV_Results.csv')
    df.rename(columns={'param_svc__C': 'C'}, inplace=True)
    df.rename(columns={'mean_test_score': 'mean test score'}, inplace=True)
    table = df[["C", "mean test score"]]
    table.style.set_properties(**{'text-align': 'left'})
    str = table.style.hide(axis="index").to_latex(hrules=True,
                                                  caption=f"results from SVR gridsearch on {caption_target}",
                                                  label="table:svr_grid_results",
                                                  position_float="centering")
    file = open(f"C:\\Users\\joens\\source\\repos\\p5\\P5-Knee-Surgery\\methods\\Support_vector_regression\\svr_table_{target}.csv", 'a')
    file.write(str)
    file.close()
    print()


headers = gives_header_array()
caption_headers = gives_caption_header_array()
for i in range(8):
    table_gen(headers[i], caption_headers[i])

