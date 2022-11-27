import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from numpy.random import RandomState, SeedSequence
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.constants._codata import val
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')

# Constant to change the random seed for the train_test_split
ran_seed = 69

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

def graph_information_k_value(title, xlable, ylable):
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.ylim(-2500, 27500)

def graph_information_epsr_value(title, xlable, ylable):
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.ylim(-0.25, 0.25)

def plt_graph_test(y_test, predictions_test, header_name):
    plt.figure()  # This makes a new figure
    plt.scatter(y_test, predictions_test, color=DotColor, s=MarkerSize)
    plt.plot(y_test, y_test, color='red')

    if 'k' in header_name:
        graph_information_k_value(f'{header_name} test model', 'Actual ' f'{header_name} value', 'Predicted ' f'{header_name} value')
    else:
        graph_information_epsr_value(f'{header_name} test model', 'Actual ' f'{header_name} value', 'Predicted ' f'{header_name} value')

    plt.savefig('./multiple-linear-regression-figures/prediction_test_' f'{header_name}.png')

def plt_graph_train(y_train, predictions_train, header_name):
    plt.figure()  # This makes a new figure
    plt.scatter(y_train, predictions_train, color=DotColor, s=MarkerSize)
    plt.plot(y_train, y_train, color='red')

    if 'k' in header_name:
        graph_information_k_value(f'{header_name} train model', 'Actual ' f'{header_name} value', 'Predicted ' f'{header_name} value')
    else:
        graph_information_epsr_value(f'{header_name} train model', 'Actual ' f'{header_name} value', 'Predicted ' f'{header_name} value')

    plt.savefig('./multiple-linear-regression-figures/prediction_train_' f'{header_name}.png')

def train_test_model(target, make_graph=True, calculate=True):

    # Read the data
    predictors = list(set(list(df.columns)) - set(result_columns))
    df[predictors] = df[predictors] / df[predictors].max()

    x = df[predictors].values
    y = df[target].values

    # Train gets the train_ratio of the data set (train = 80%, test = 20% - af det fulde datasæt)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio, random_state=ran_seed)

    # Both Validation and Test get 50% each of the remainder (val = 10%, test = 10% - af det fulde datasæt)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio), random_state=ran_seed)

    x_all, y_all = np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val))


    # creating a regression model
    # model = LinearRegression()
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mul_linear_reg', linear_model.LinearRegression())
    ])

    # fitting the model
    model.fit(x_all, y_all)

    # making predictions
    predictions_test = model.predict(x_test)
    predictions_train = model.predict(x_all)

    if make_graph:
        plt_graph_train(y_all, predictions_train, header)
        plt_graph_test(y_test, predictions_test, header)

    if calculate:
        r2_train = r2_score(y_all, predictions_train)
        rmse_train = mean_squared_error(y_all, predictions_train, squared=False)
        mae_train = mean_absolute_error(y_all, predictions_train)

        r2_test = r2_score(y_test, predictions_test)
        rmse_test = mean_squared_error(y_test, predictions_test, squared=False)
        mae_test = mean_absolute_error(y_test, predictions_test)

        return [r2_train, rmse_train, mae_train, r2_test, rmse_test, mae_test]

def update_dict(dict, header, list_data):
    l_train = dict[header + '_train']
    l_test = dict[header + '_test']
    l_train.append({'R2': list_data[0], 'RMSE': list_data[1], 'MAE': list_data[2]})
    l_test.append({'R2': list_data[3], 'RMSE': list_data[4], 'MAE': list_data[5]})

def get_stats(header, list):
    return f'{header};{np.max(list)};{np.min(list)};{np.mean(list)}\n'

def find_stats(dict, header):
    r2_train = [x['R2'] for x in dict[header + '_train']]
    rmse_train = [x['RMSE'] for x in dict[header + '_train']]
    mae_train = [x['MAE'] for x in dict[header + '_train']]
    r2_test = [x['R2'] for x in dict[header + '_test']]
    rmse_test = [x['RMSE'] for x in dict[header + '_test']]
    mae_test = [x['MAE'] for x in dict[header + '_test']]

    res = []
    res.append(get_stats(header + '_train_R2', r2_train))
    res.append(get_stats(header + '_train_RMSE', rmse_train))
    res.append(get_stats(header + '_train_MAE', mae_train))
    res.append(get_stats(header + '_test_R2', r2_test))
    res.append(get_stats(header + '_test_RMSE', rmse_test))
    res.append(get_stats(header + '_test_MAE', mae_test))
    return res


# importing data
df = pd.read_csv('../data_processing/raw_data/final_final_final.csv')
result_columns = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
results = dict()
#rounds = 1
file = open('./multiple-linear-regression-figures-results.csv', 'w')
file.write('ID;Max;Min;Avg\n')

for header in result_columns:
    print('\n' + header + ':')
    results[header + '_train'] = []
    results[header + '_test'] = []
    list_data = train_test_model(header, make_graph=True, calculate=True)
    update_dict(results, header, list_data)

    #for j in range(rounds):
    #    print("\r" f'{j + 1} / {rounds}', end='')
    #    list_data = dynamic_train_test_model(header)
    #    update_dict(results, header, list_data)

    res = find_stats(results, header)
    file.writelines(res)

file.close()