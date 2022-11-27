import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

"""
To use any of the methods in this file, insert ONE of the following lines in your imports or simply select the relevant
methods to import:

1) from p5_package.standards import *
   (imports everything in standards - how to use: get_result_columns())
   
2) from p5_package import standards
   (imports standards library - how to use: standards.get_result_columns())

------------------------------------------------------------------------------------------------------------------------

Implementation instructions (temporary):
----------------------------------------
1. Clean up and collect everything together in ONE script (if you haven't already), and name it "your_method_name.py".

2. Move the script to the methods folder (this is a necessary evil, in order to use this library without any elaborate
   hacks - the problem is that the library folder needs to be in the same file execution path).
   
3. Import the following line in your script:
   from p5_package import standards

4. import the data file like this:
   data = pd.read_csv('./data.csv', index_col=0)

5. Use the following file paths as main outputs and adapt to your use case:
   LOG_DIR = f"results/your_method_name"           # the main path for your method output
   MODEL_DIR = f"{LOG_DIR}/models/"                # the path for the specific models
   RESULT_DIR = f"{LOG_DIR}/"                      # the path for the result csv file
   
6. If in doubt, have a look at "neural_network.py" - it is fully implemented. Alternatively just ask Jamie.

NB! These instructions (below and including the dashes) will be removed, once all methods has been successfully
    implemented.
"""

matplotlib.use('Agg')                           # backend for pyplot, makes it possible to save graphs to a png format
style.use('seaborn-v0_8')                       # style handles the themes for the graphs

seed = 69                                       # to get reproducible data splits and hyperparameters
train_ratio = 0.8                               # represents the percentage of the data set allocated to train and tune
validation_ratio = 0.1                          # represents the percentage of the data set allocated to validation
test_ratio = 0.1                                # represents the percentage of the data set allocated to evaluation

# <editor-fold desc="Getters">


def get_seed():
    """ Returns the seed as an integer, used for reproducibility. """
    return seed


def get_result_columns():
    """ Returns a list of the result column names in string format. """
    return ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']


def get_train_test_split(x, y):
    """ Returns a train (90%) and test (10%) data split. """
    return train_test_split(x, y, test_size=test_ratio, random_state=seed)


def get_train_validation_test_split(x, y):
    """ Returns a train (80%), validation (10%) and test (10%) data split. """
    # get remainder (90%) and test (10%) split
    x_rem, x_test, y_rem, y_test = get_train_test_split(x, y)
    # get train (approx 89% of the remainder) and validation (approx 11% of the remainder) split
    x_train, x_val, y_train, y_val = train_test_split(x_rem, y_rem, random_state=seed,
                                                      train_size=train_ratio / (train_ratio + validation_ratio))
    return x_train, y_train, x_val, y_val, x_test, y_test


def evaluate_model(expected_y, predicted_y):
    """ Returns an evaluation of the model, with the following scores: R^2, MAE and RMSE. """
    r2 = r2_score(expected_y, predicted_y)
    mae = mean_absolute_error(expected_y, predicted_y)
    rmse = mean_squared_error(expected_y, predicted_y, squared=False)
    return r2, mae, rmse


def get_prefix(train, test):
    """ Returns a positive (+) prefix if train is greater than test, else it already has a negative (-) prefix. """
    return '+' if (train - test) > 0 else ''


def get_evaluation_results(train_evaluation, test_evaluation):
    """ Returns a row of results to a model - test, train, difference - R^2, MAE, RMSE. """
    train_r2, train_mae, train_rmse = train_evaluation
    test_r2, test_mae, test_rmse = test_evaluation

    # Calculate differences between the three results
    r2_difference = f"({get_prefix(train_r2, test_r2)}{((train_r2 - test_r2) * 100):.4f}%)"
    mae_difference = f"({get_prefix(train_mae, test_mae)}{(train_mae - test_mae):.4f})"
    rmse_difference = f"({get_prefix(train_rmse, test_rmse)}{(train_rmse - test_rmse):.4f})"

    # Create table row with results
    intermediate_results = {'Test_R2': [f"{(test_r2 * 100):.4f}%"],
                            'Train_R2': [f"{(train_r2 * 100):.4f}%"],
                            'Difference R2': [f"{r2_difference}"],
                            'Test_MAE': [f"{test_mae:.4f}"],
                            'Train_MAE': [f"{train_mae:.4f}"],
                            'Difference MAE': [f"{mae_difference}"],
                            'Test_RMSE': [f"{test_rmse:.4f}"],
                            'Train_RMSE': [f"{train_rmse:.4f}"],
                            'Difference RMSE': [f"{rmse_difference}"]}

    return intermediate_results


# </editor-fold>

# <editor-fold desc="Graph">


def create_and_save_graph(target, actual_values, predicted_values, path):
    """ Creates a scatter plot graph with the expected data and predicted data. """
    fig, ax = plt.subplots()
    ax.scatter(actual_values, predicted_values, s=5, c='b', alpha=0.25, label='Data Points')

    min_lim, max_lim = min(actual_values), max(actual_values)
    five_percent = (max_lim - min_lim) * .05
    min_lim, max_lim = min_lim - five_percent, max_lim + five_percent

    plt.ylim(min_lim, max_lim)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, linewidth=2, c='r', label='Best Fit Line')

    plt.title(f"Predicted and Actual {target} Values")
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    plt.savefig(path, dpi=300)
    plt.close()


# </editor-fold>

# <editor-fold desc="Saving Data">


def save_csv(data, path):
    """ This saves a given Pandas DataFrame to a given file path in a CSV format. """
    data.to_csv(f"./{path}")


# </editor-fold>