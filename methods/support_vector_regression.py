import pandas as pd
import numpy as np
import sklearn.svm as sk
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import sys
from lib.standards import *

# Gets int from
ran_seed = get_seed()

# Path constants
LOG_DIR = f"results/support_vector_regression"  # the main path for your method output
MODEL_DIR = f"{LOG_DIR}/models/"  # the path for the specific models
RESULT_DIR = f"{LOG_DIR}/"  # the path for the result csv file

def gives_x_param_header():
    x = []
    for i in range(1, 24):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x

def handle_model(target):
    # Set data up
    x = df[gives_x_param_header()].values
    y = df[target].values

    # Train gets the train_ratio of the data set (train = 80%, test = 20% - af det fulde datasæt)
    x_train, x_test, y_train, y_test = get_train_test_split(x, y)

    # Setup for grid search
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('pca', PCA()),
                     ('svc', sk.LinearSVR(max_iter=35000))])

    # Making list with all the steps
    pca_list = []
    for i in range(97, 277, 45):
        pca_list.append(i)

    c_list = [50]
    for i in range(100, 1000, 100):
        c_list.append(i)

    parameter_grid = {'pca__n_components': pca_list,
                      'svc__C': c_list}

    #parameter_grid = {'svc__C': c_list}

    # Gridsearch
    gridsearch = GridSearchCV(estimator=pipe, param_grid=parameter_grid, scoring="neg_root_mean_squared_error", cv=5, verbose=3, n_jobs=4)

    # Fit the model
    results = gridsearch.fit(x_train, y_train)

    print('Best score: ', results.best_score_)
    print('Best parameters: ', results.best_params_)

    # Fitting the model
    final_model = Pipeline([('scaler', StandardScaler()),
                            ('pca', PCA(n_components=results.best_params_['pca__n_components'])),
                            ('svc', sk.LinearSVR(max_iter=35000, C=results.best_params_["svc__C"]))])

    # Fitting the model
    final_model.fit(x_train, y_train)

    # Making predictions
    predictions_train = final_model.predict(x_train)
    predictions_test = final_model.predict(x_test)

    train_evaluation = evaluate_model(y_train, predictions_train)
    test_evaluation = evaluate_model(y_test, predictions_test)

    create_and_save_graph(target, y_test, predictions_test, f'{MODEL_DIR}{target}/{target}-plot.png')

    return get_evaluation_results(train_evaluation, test_evaluation)


# importing data and converting it to float32
df = pd.read_csv('./data.csv', index_col=0).astype(np.float32)
result_columns = get_result_columns()

# Create, train and evaluate all eight models
#acl_epsr = pd.DataFrame(handle_model("ACL_epsr"), index=["ACL_epsr"])
#lcl_epsr = pd.DataFrame(handle_model("LCL_epsr"), index=["LCL_epsr"])
#mcl_epsr = pd.DataFrame(handle_model("MCL_epsr"), index=["MCL_epsr"])
#pcl_epsr = pd.DataFrame(handle_model("PCL_epsr"), index=["PCL_epsr"])
#acl_k = pd.DataFrame(handle_model("ACL_k"), index=["ACL_k"])
#lcl_k = pd.DataFrame(handle_model("LCL_k"), index=["LCL_k"])
mcl_k = pd.DataFrame(handle_model("MCL_k"), index=["MCL_k"])
#pcl_k = pd.DataFrame(handle_model("PCL_k"), index=["PCL_k"])

# Concatenate intermediate results
result = pd.concat([mcl_k])
#acl_epsr , lcl_epsr, mcl_epsr, pcl_epsr, acl_k, lcl_k, mcl_k, pcl_k
# Print and save results
print(result.to_string())
save_csv(result, f"{RESULT_DIR}result.csv")