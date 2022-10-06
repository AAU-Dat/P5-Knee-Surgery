import numpy as np


def nn(iter, layers):
    # Step 1: Loading the Required Libraries and Modules

    import pandas as pd
    from sklearn import metrics
    from sklearn.neural_network import MLPRegressor

    from sklearn.model_selection import train_test_split

    # Step 2: Reading the Data and Performing Basic Data Checks

    df = pd.read_csv('../data_processing/final_final_final.csv', index_col=0)
    df.shape
    df.describe().transpose()

    # Step 3: Creating Arrays for the Features and the Response Variable

    result_columns = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
    target_column = ['ACL_k']
    predictors = list(set(list(df.columns)) - set(result_columns))
    df[predictors] = df[predictors] / df[predictors].max()
    df.describe().transpose()

    # Step 4: Creating the Training and Test Datasets

    X = df[predictors].values
    y = df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)

    # Step 5: Building, Predicting, and Evaluating the Neural Network Model

    # Setting up the regressor
    mlp = MLPRegressor(hidden_layer_sizes=layers, activation='relu', solver='adam', max_iter=iter)
    mlp.fit(X_train, y_train.ravel())

    # Predicting the regressor
    expected_y = y_test
    predicted_y = mlp.predict(X_test)

    # mean squared log error commented out until normalized (negative value fix)
    # return metrics.r2_score(expected_y, predicted_y), metrics.mean_squared_log_error(expected_y, predicted_y)
    return metrics.r2_score(expected_y, predicted_y)



def runnn(attempts,iter,layers):
    arrx = np.zeros((attempts))
    arry = np.zeros((attempts))

    for x in range(attempts):
        #arrx[x],arry[x] = nn(iter,layers)
        arrx[x] = nn(iter, layers)

    retx = 0
    rety = 0
    for x in range(attempts):
        retx += arrx[x]
        rety += arry[x]

    #return retx/attempts,rety/attempts
    return retx / attempts

nnaccs = np.zeros((100))

nnaccs[0] = runnn(10,400,(8,8,8))
nnaccs[1] = runnn(10,450,(8,8,8))
nnaccs[2] = runnn(10,500,(8,8,8))
nnaccs[3] = runnn(10,550,(8,8,8))
nnaccs[4] = runnn(10,600,(8,8,8))
nnaccs[5] = runnn(10,650,(8,8,8))
nnaccs[6] = runnn(10,700,(8,8,8))

nnaccs[7] = runnn(10,400,(16,16,16))
nnaccs[8] = runnn(10,450,(16,16,16))
nnaccs[9] = runnn(10,500,(16,16,16))
nnaccs[10] = runnn(10,550,(16,16,16))
nnaccs[11] = runnn(10,600,(16,16,16))
nnaccs[12] = runnn(10,650,(16,16,16))
nnaccs[13] = runnn(10,700,(16,16,16))

nnaccs[14] = runnn(10,400,(40,40))
nnaccs[15] = runnn(10,450,(40,40))
nnaccs[16] = runnn(10,500,(40,40))
nnaccs[17] = runnn(10,550,(40,40))
nnaccs[18] = runnn(10,600,(40,40))
nnaccs[19] = runnn(10,650,(40,40))
nnaccs[20] = runnn(10,700,(40,40))




for x in nnaccs:
    print(str(x)+": " + x)