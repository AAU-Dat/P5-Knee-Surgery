import keras.callbacks
import matplotlib
import numpy as np
import pandas as pd
import keras_tuner as kt
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import linear_model
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.regularizers import L2
from keras_tuner import Hyperband
from datetime import datetime

# Backend and style for pyplot
matplotlib.use('Agg')
style.use('seaborn')

# Date and Time for when running this file
now = datetime.now().strftime("%d-%b-%Y at %H:%M")

# Relevant path directories
LOG_DIR = f"Test ran on {now}"
MODEL_DIR = f"{LOG_DIR}/models/"
RESULT_DIR = f"{LOG_DIR}/"

# Preparing the Data Set and relevant constants
ACL_k, ACL_epsr, PCL_k, PCL_epsr = 'ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr'
MCL_k, MCL_epsr, LCL_k, LCL_epsr = 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr'
df = pd.read_csv('../data_processing/final_final_final.csv', index_col=0)
result_columns = [ACL_k, ACL_epsr, PCL_k, PCL_epsr, MCL_k, MCL_epsr, LCL_k, LCL_epsr]
input_shape = (276,)                            # input is a row of entries (284 [total cols] - 8 [result cols] = 276)
seed = 69                                       # to get reproducible data splits and hyperparameters
train_ratio = 0.8                               # percentage of the data set allocated to train and tune
validation_ratio = 0.1                          # percentage of the data set allocated to validation
test_ratio = 0.1                                # percentage of the data set allocated to evaluation

# Configure the Search Space
max_layers = 10                                 # the maximum amount of hidden layers in any model
max_epochs = 100                                # the maximum amount of epochs allowed in any model
HP = kt.HyperParameters()
HP.Int('number_of_layers', min_value=1, max_value=max_layers)
HP.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
HP.Choice('l2', values=[0.01, 0.001, 0.1, 0.005, 0.05])
HP.Choice('activation', values=['relu', 'tanh', 'sigmoid', 'softplus'])
HP.Choice('loss', values=['mean_squared_error', 'mean_absolute_error', 'mean_squared_logarithmic_error'])
HP.Boolean('use_dropout')
HP.Fixed('metrics', value='mean_absolute_error')
for i in range(max_layers):
    HP.Int(name=f"units_layer_{i + 1}", min_value=32, max_value=2048, step=32)
    HP.Float(name=f"dropout_{i + 1}", min_value=0.0, max_value=0.5, default=0.25, step=0.05)


# <editor-fold desc="Building the Model architecture">

def build_model(hp):
    # Create Sequential Model and set up Input Layer
    model = Sequential([Flatten(name='input_layer', input_shape=input_shape)])

    # Hidden Layers
    for j in range(hp.get('number_of_layers')):
        number = f"{j + 1}"

        model.add(Dense(units=hp.get(f"units_layer_{number}"), name=f"hidden_layer_{number}",
                        activation=hp.get('activation'), kernel_regularizer=L2(hp.get('l2'))))

        if hp.get('use_dropout'):
            model.add(Dropout(rate=hp.get(f"dropout_{number}")))

    # Output Layer
    model.add(Dense(1, activation='linear', kernel_initializer='random_normal', name='output_layer'))

    # Set up Compiler
    model.compile(optimizer=Adam(learning_rate=hp.get('learning_rate')), loss=hp.get('loss'), metrics=[hp.get('metrics')])
    return model

# </editor-fold>


# <editor-fold desc="Building the Hyperparameter tuner">

def build_tuner(x_train, y_train, x_val, y_val):
    # Create the Hyperband tuner with the objective to minimise the error on the validation data
    tuner = Hyperband(
        build_model, objective='val_mean_absolute_error', max_epochs=max_epochs, hyperparameters=HP,
        factor=3, hyperband_iterations=3, directory=LOG_DIR, project_name='P5-Knee-Surgery', seed=seed
    )

    # Set up early stopping
    stop_early = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=3)

    # Search for the best model and its hyperparameters
    tuner.search(x_train, y_train, epochs=max_epochs, validation_data=(x_val, y_val),
                 callbacks=[stop_early], use_multiprocessing=True, workers=8)

    return tuner

# </editor-fold>


# <editor-fold desc="Training the best Model">

def train_hypermodel(target, tuner, x_train, y_train, x_val, y_val):
    best_hp = tuner.get_best_hyperparameters()[0]

    # Build model with the best hyperparameters and train it for 100 epochs
    best_model = tuner.hypermodel.build(best_hp)
    history = best_model.fit(x_train, y_train.ravel(), epochs=max_epochs, validation_data=(x_val, y_val))

    # Find the best epoch from the first training
    val_mae_per_epoch = history.history['val_mean_absolute_error']
    best_epoch = val_mae_per_epoch.index(min(val_mae_per_epoch)) + 1

    # Build the best hypermodel from the best hyperparameters
    hypermodel = tuner.hypermodel.build(best_hp)

    # Retrain the model with the best found epoch
    x, y = np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val))
    hypermodel.fit(x, y.ravel(), epochs=best_epoch)

    # Save the model to the log directory for future reference
    hypermodel.save(f"{MODEL_DIR}{target}")

    return hypermodel

# </editor-fold>


# <editor-fold desc="Evaluating the Model">


def evaluate(expected_y, predicted_y):
    # Evaluate the model
    r2 = r2_score(expected_y, predicted_y)
    mae = mean_absolute_error(expected_y, predicted_y)
    rmse = mean_squared_error(expected_y, predicted_y, squared=False)

    return [r2, mae, rmse]


def save_test_data(target, expected_y, predicted_y):
    test_data = {'Expected': expected_y.tolist(), 'Predicted': predicted_y.tolist()}
    data = pd.DataFrame(test_data)
    data.to_csv(f"{MODEL_DIR}{target}/test-data.csv")


def get_prefix(train, test):
    # If value is greater than 0 it is positive, else it is negative
    return '+' if (train - test) > 0 else ''


def round_result(number):
    # If number is less than 1 or -1, greater accuracy is wanted and four decimals, else 2 decimals
    return f"{number:.4f}" if number < 1 or number < -1 else f"{number:.2f}"


def evaluate_model(target, x_test, y_test, x_train, y_train, hypermodel):
    # Prepare the expected and predicted data set
    expected_y_train, predicted_y_train = y_train, hypermodel.predict(x_train)
    expected_y_test, predicted_y_test = y_test, hypermodel.predict(x_test)

    # Save the test data for the graph plotting
    save_test_data(target, expected_y_test, predicted_y_test)

    # Evaluate both the test and train data set
    test_r2, test_mae, test_rmse = evaluate(expected_y_test, predicted_y_test)
    train_r2, train_mae, train_rmse = evaluate(expected_y_train, predicted_y_train)

    # Calculate differences between the three results
    r2_difference = f"({get_prefix(train_r2, test_r2)}{((train_r2 - test_r2) * 100):.4f}%)"
    mae_difference = f"({get_prefix(train_mae, test_mae)}{(train_mae - test_mae):.4f})"
    rmse_difference = f"({get_prefix(train_rmse, test_rmse)}{(train_rmse - test_rmse):.4f})"

    # Create table row with results
    intermediate_results = {'Test_R2': [f"{round_result(test_r2 * 100)}%"],
                            'Train_R2': [f"{round_result(train_r2 * 100)}%"],
                            'Difference R2': [f"{r2_difference} |"],
                            'Test_MAE': [f"{round_result(test_mae)}"],
                            'Train_MAE': [f"{round_result(train_mae)}"],
                            'Difference MAE': [f"{mae_difference} |"],
                            'Test_RMSE': [f"{round_result(test_rmse)}"],
                            'Train_RMSE': [f"{round_result(train_rmse)}"],
                            'Difference RMSE': [f"{rmse_difference} |"]}

    return [intermediate_results, [expected_y_test, predicted_y_test]]

# </editor-fold>


# <editor-fold desc="Make the Graph">

def make_graph(target, expected_data, predicted_data):
    # Prepare regression line
    values = linear_model.LinearRegression()
    values.fit(expected_data.reshape(-1, 1), predicted_data)
    regression_line = values.predict(expected_data.reshape(-1, 1))

    # Set up graph
    plt.scatter(expected_data, predicted_data, label='Data Points', c='r', alpha=0.6, s=5)
    plt.plot(expected_data, regression_line, label='Best Fit Line', c='b', linewidth=2)
    plt.title(f"Expected and Actual {target} Values")
    plt.xlabel('Actual Values')
    plt.ylabel('Expected Values')
    plt.legend()

    # Save and close figure
    plt.savefig(f"{MODEL_DIR}{target}/{target}-plot.png")
    plt.close()

# </editor-fold>


# <editor-fold desc="Handling a Model">

def handle_model(target):
    # Set up and normalise the input
    predictors = list(set(list(df.columns)) - set(result_columns))
    df[predictors] = df[predictors] / df[predictors].max()

    # Set up the data set
    x = df[predictors].values
    y = df[target].values

    # Train gets the train_ratio of the data set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio, random_state=seed)

    # Both Validation and Test get 50% each of the remainder
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=seed,
                                                    test_size=test_ratio / (test_ratio + validation_ratio))

    # Build a Tuner and search for Hyperparameters on the Train data, test on the Validate data
    tuner = build_tuner(x_train, y_train, x_val, y_val)

    # Build and train the best Model on the Train data, test on the Validate data
    best_model = train_hypermodel(target, tuner, x_train, y_train, x_val, y_val)

    # Evaluate the model using cross-validation on the Test data and return the result
    results, test_data = evaluate_model(target, x_test, y_test, x_train, y_train, best_model)

    # Prints and saves the graph to the target model folder
    expected_y, predicted_y = test_data
    make_graph(target, expected_y, predicted_y)

    return results

# </editor-fold>


# Create, train and evaluate all eight models
acl_epsr = pd.DataFrame(handle_model(ACL_epsr), index=[f"{ACL_epsr} |"])
pcl_epsr = pd.DataFrame(handle_model(PCL_epsr), index=[f"{PCL_epsr} |"])
mcl_epsr = pd.DataFrame(handle_model(MCL_epsr), index=[f"{MCL_epsr} |"])
lcl_epsr = pd.DataFrame(handle_model(LCL_epsr), index=[f"{LCL_epsr} |"])
acl_k = pd.DataFrame(handle_model(ACL_k), index=[f"{ACL_k}    |"])
pcl_k = pd.DataFrame(handle_model(PCL_k), index=[f"{PCL_k}    |"])
mcl_k = pd.DataFrame(handle_model(MCL_k), index=[f"{MCL_k}    |"])
lcl_k = pd.DataFrame(handle_model(LCL_k), index=[f"{LCL_k}    |"])

# Concatenate intermediate results
result = pd.concat([acl_epsr, pcl_epsr, mcl_epsr, lcl_epsr, acl_k, pcl_k, mcl_k, lcl_k])

# Print and save results
print(result.to_string())
result.to_csv(f"{RESULT_DIR}result.csv")
