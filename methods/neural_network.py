import keras.callbacks
import numpy as np
import pandas as pd
import keras_tuner as kt
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import L2
from keras_tuner import Hyperband
from lib.standards import *

# Relevant path directories
LOG_DIR = f"results/neural_network"             # the main path for neural network output
MODEL_DIR = f"{LOG_DIR}/models/"                # the path for the specific models
RESULT_DIR = f"{LOG_DIR}/"                      # the path for the result csv file

# Preparing the Data Set and relevant constants
data = pd.read_csv('./data.csv', index_col=0)   # data set without the index column
result_columns = get_result_columns()           # a list of the label column names, in string format
input_shape = (276,)                            # input is a row of entries (284 [total cols] - 8 [result cols] = 276)
seed = get_seed()                               # to get reproducible data splits and hyperparameters

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
    model.compile(optimizer=Adam(learning_rate=hp.get('learning_rate')),
                  loss=hp.get('loss'), metrics=[hp.get('metrics')])
    return model


# </editor-fold>

# <editor-fold desc="Hyperparameter tuning">


def hyperparameter_tuning(x_train, y_train, x_val, y_val):
    # Create the Hyperband tuner with the objective to minimise the error on the validation data
    tuner = Hyperband(
        build_model, objective='val_mean_absolute_error', max_epochs=max_epochs, hyperparameters=HP,
        factor=3, hyperband_iterations=1, directory=LOG_DIR, project_name='P5-Knee-Surgery', seed=seed
    )

    # TODO: change to RMSE if possible

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
    # TODO: change to RMSE if possible
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


def save_test_data(target, expected_y, predicted_y):
    test_data = {'Expected': expected_y, 'Predicted': predicted_y}
    data_frame = pd.DataFrame(test_data)
    save_csv(data_frame, f"{MODEL_DIR}{target}/test-data.csv")


def handle_evaluation(x_train, y_train, x_test, y_test, hypermodel):
    # Prepare the expected and predicted data set
    expected_y_train, predicted_y_train = y_train, hypermodel.predict(x_train)
    expected_y_test, predicted_y_test = y_test, hypermodel.predict(x_test)

    # Reshape all sets to make sure that they are saved in a one dimensional list and save the test data
    expected_y_train, predicted_y_train = expected_y_train.reshape(-1, 1), predicted_y_train.reshape(-1, 1)
    expected_y_test, predicted_y_test = expected_y_test.reshape(-1, 1), predicted_y_test.reshape(-1, 1)
    save_test_data(target, expected_y_test, predicted_y_test)

    # Run evaluation on both the Train data and Test data
    train_evaluation = evaluate_model(expected_y_train, predicted_y_train)
    test_evaluation = evaluate_model(expected_y_test, predicted_y_test)

    return train_evaluation, test_evaluation, [expected_y_test, predicted_y_test]


# </editor-fold>

# <editor-fold desc="Handling a Model">


def handle_model(target):
    # Set up and normalise the input
    predictors = list(set(list(data.columns)) - set(result_columns))
    data[predictors] = data[predictors] / data[predictors].max()

    # Set up the domain and label of the data set
    x, y = data[predictors].values, data[target].values

    # Train, Test and Validation data split
    train, validation, test = get_train_validation_test_split(x, y)
    x_train, y_train = train
    x_validation, y_validation = validation
    x_test, y_test = test

    # Build a Tuner and search for Hyperparameters on the Train data, test on the Validation data
    tuner = hyperparameter_tuning(x_train, y_train, x_validation, y_validation)

    # Build and train the best Model on the Train data, test on the Validation data
    best_model = train_hypermodel(target, tuner, x_train, y_train, x_validation, y_validation)

    # Handle the evaluation of the model on both train and test data
    train_evaluation, test_evaluation, test_data = handle_evaluation(x_train, y_train, x_test, y_test, best_model)
    actual_y, predicted_y = test_data

    # Make a scatter plot graph with the actual and predicted values
    create_and_save_graph(target, actual_y, predicted_y, f"{MODEL_DIR}{target}/{target}-plot.png")

    results = get_evaluation_results(train_evaluation, test_evaluation)
    return results


# </editor-fold>

# Create, train and evaluate all eight models
# acl_epsr = pd.DataFrame(handle_model(ACL_epsr), index=["ACL_epsr"])
# lcl_epsr = pd.DataFrame(handle_model(LCL_epsr), index=["LCL_epsr"])
# mcl_epsr = pd.DataFrame(handle_model(MCL_epsr), index=["MCL_epsr"])
# pcl_epsr = pd.DataFrame(handle_model(PCL_epsr), index=["PCL_epsr"])
acl_k = pd.DataFrame(handle_model("ACL_k"), index=["ACL_k"])
# lcl_k = pd.DataFrame(handle_model("LCL_k"), index=["LCL_k"])
# mcl_k = pd.DataFrame(handle_model("MCL_k"), index=["MCL_k}])
# pcl_k = pd.DataFrame(handle_model("PCL_k"), index=["PCL_k"])

# Concatenate intermediate results
# result = pd.concat([acl_epsr, lcl_epsr, mcl_epsr, pcl_epsr, acl_k, lcl_k, mcl_k, pcl_k])
result = acl_k

# Print and save results
print(result.to_string())
save_csv(result, f"{RESULT_DIR}result.csv")
