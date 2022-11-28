import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from p5_package.standards import *
matplotlib.use('Agg')

# Constants t ochange random seed
ran_seed = get_seed()

# Constants to change train test split
train_ratio = 0.8
test_ratio = 0.1
validation_ratio = 0.1

# Constants to change style in graph
MarkerSize = 0.1
DotColor = 'Blue'

# Constants to change x & y axises in the saved graphs
xleft_k     = -2500
xright_k    = 30000
ybottom_k   = -2500
ytop_k      = 30000

xleft_epsr  = -0.10
xright_epsr = 0.30
ybottom_epsr= -0.10
ytop_epsr   = 0.30

# Read the data
ACL_k, ACL_epsr, PCL_k, PCL_epsr = 'ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr'
MCL_k, MCL_epsr, LCL_k, LCL_epsr = 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr'

# Path constants
LOG_DIR = f"results/your_method_name"  # the main path for your method output
MODEL_DIR = f"{LOG_DIR}/models/"  # the path for the specific models
RESULT_DIR = f"{LOG_DIR}/"  # the path for the result csv file

def handle_model(target):

    # Read the data
    predictors = list(set(list(df.columns)) - set(result_columns))
    df[predictors] = df[predictors] / df[predictors].max()

    # Set data up
    x = df[predictors].values
    y = df[target].values

    # Train gets the train_ratio of the data set (train = 80%, test = 20% - af det fulde datasæt)
    x_train, x_test, y_train, y_test = get_train_test_split(x, y)

    # creating polynomial features
    poly_reg_model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('pca_poly_reg', PolynomialFeatures(degree=2, include_bias=False)),
        ('lin_reg', LinearRegression(n_jobs=-1))
    ])

    # Making list with all the steps
    list_param = []
    for i in range(50, 130, 5):
        list_param.append(i)

    param_grid = {'pca__n_components': list_param}

    # Grid search
    grid_search = GridSearchCV(poly_reg_model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=3)

    # Fit the model
    results = grid_search.fit(x_train, y_train)

    print('Best score: ', results.best_score_)
    print('Best parameters: ', results.best_params_)

    # Train the model
    best_poly_reg_model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=results.best_params_['pca__n_components'])),
        ('pca_poly_reg', PolynomialFeatures(degree=2, include_bias=False)),
        ('lin_reg', LinearRegression(n_jobs=-1))
    ])

    # fitting the model
    best_poly_reg_model.fit(x_train, y_train)

    # making predictions
    predictions_train = best_poly_reg_model.predict(x_train)
    predictions_test = best_poly_reg_model.predict(x_test)

    train_evaluation = evaluate_model(y_train, predictions_train)
    test_evaluation = evaluate_model(y_test, predictions_test)

    create_and_save_graph(target, y_test, predictions_test, f'{MODEL_DIR}{target}/{target}-plot.png')
    get_evaluation_results(train_evaluation, test_evaluation)


# importing data and converting it to float32
df = pd.read_csv('./data.csv', index_col=0).astype(np.float32)
result_columns = get_result_columns()

# Create, train and evaluate all eight models
acl_epsr = pd.DataFrame(handle_model("ACL_epsr"), index=["ACL_epsr"])
lcl_epsr = pd.DataFrame(handle_model("LCL_epsr"), index=["LCL_epsr"])
mcl_epsr = pd.DataFrame(handle_model("MCL_epsr"), index=["MCL_epsr"])
pcl_epsr = pd.DataFrame(handle_model("PCL_epsr"), index=["PCL_epsr"])
acl_k = pd.DataFrame(handle_model("ACL_k"), index=["ACL_k"])
lcl_k = pd.DataFrame(handle_model("LCL_k"), index=["LCL_k"])
mcl_k = pd.DataFrame(handle_model("MCL_k"), index=["MCL_k"])
pcl_k = pd.DataFrame(handle_model("PCL_k"), index=["PCL_k"])

# Concatenate intermediate results
result = pd.concat([acl_epsr, lcl_epsr, mcl_epsr, pcl_epsr, acl_k, lcl_k, mcl_k, pcl_k])

# Print and save results
print(result.to_string())
standards.save_csv(result, f"{RESULT_DIR}result.csv")