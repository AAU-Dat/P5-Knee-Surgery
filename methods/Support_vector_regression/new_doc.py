import pandas as pd
import numpy as np
import sklearn.svm as sk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import sys
project_path = 'C:\\Users\\houga\\Desktop\\uni\\P5-Knee-Surgery\\'
sys.path.append(f'{project_path}p5_package')
import p5_standardizations as p5


# data
df = pd.read_csv('../data_processing/final_final_final.csv').astype(np.float32)
header = p5.gives_header_array()
path = f'{project_path}Support_vector_regression\\'


def run_svr(target_index):
    x_all, y_all, x_test, y_test = p5.train_test_split(df, target_index)

    # Setup for grid search
    pipe = Pipeline([('scaler', StandardScaler()), ('svc', sk.LinearSVR(max_iter=10000))])

    list = []
    for i in range(100, 200, 100):
        list.append(i)
    parameter_grid = {'svc__C': list}

    # gridsearch
    gridsearch = GridSearchCV(estimator=pipe, param_grid=parameter_grid, scoring="r2", cv=5, verbose=3, n_jobs=3)
    results = gridsearch.fit(x_all, y_all)
    print(f"{results.best_params_}\t{results.best_score_}")

    # making best model
    final_model = Pipeline([('scaler', StandardScaler()), ('svc', sk.LinearSVR(max_iter=10000, C=results.best_params_["svc__C"]))])
    final_model.fit(x_all, y_all)

    #predicting with best model
    prediction = final_model.predict(x_test)

    #saving results
    p5.print_predicted_data(y_test, prediction, f'{path}predictions_{header[target_index]}.csv')
    p5.print_a_graph(y_test, prediction, target_index, f'{path}svr_figures\\{header[target_index]}.png')
    p5.save_results_from_search(results, f'{path}gridsearch_results.csv')
    p5.save_results(target_index, y_test, prediction, f'{path}svr_results.csv')


def main():
    for x in range(1, 9):
        run_svr(x)


main()
