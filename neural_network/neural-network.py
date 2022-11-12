import keras.callbacks
import math
import pandas as pd
import keras_tuner as kt
from keras import layers
from sklearn import metrics
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.regularizers import L2
from keras_tuner import Hyperband
from datetime import datetime

# Date and Time for when running this file
now = datetime.now().strftime("%d-%b-%Y at %H:%M")

# Relevant path directories
LOG_DIR = f"Test ran on {now}"
MODEL_DIR = f"{LOG_DIR}/models/"

# Data Set preparation
ACL_k, ACL_epsr, PCL_k, PCL_epsr = 'ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr'
MCL_k, MCL_epsr, LCL_k, LCL_epsr = 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr'
df = pd.read_csv('../data_processing/final_final_final.csv', index_col=0)
result_columns = [ACL_k, ACL_epsr, PCL_k, PCL_epsr, MCL_k, MCL_epsr, LCL_k, LCL_epsr]
input_shape = (276,)                            # input is a row of entries (284 - [8 result columns])
seed = 69                                       # to get reproducible data splits and hyperparameters
train_ratio = 0.80                              # percentage of the data set allocated to train and tune
validation_ratio = 0.10                         # percentage of the data set allocated to validation
test_ratio = 0.10                               # percentage of the data set allocated to evaluation

# Configure the Search Space
HP = kt.HyperParameters()
HP.Int('number_of_layers', min_value=2, max_value=8)
HP.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
HP.Boolean('use_dropout')
HP.Fixed('loss', value='mean_squared_error')
HP.Fixed('metrics', value='mean_absolute_error')
for i in range(8):
    HP.Int(name=f"units_layer_{i + 1}", min_value=32, max_value=512, step=32)
    HP.Float(name=f"dropout_{i + 1}", min_value=0.0, max_value=0.5, default=0.25, step=0.05)


# <editor-fold desc="Building the Model architecture">

def build_model(hp):
    # Create Sequential Model and set up Input Layer
    model = Sequential([layers.Flatten(name='input_layer', input_shape=input_shape)])

    # Hidden Layers
    for j in range(hp.get('number_of_layers')):
        number = f"{j + 1}"

        model.add(layers.Dense(units=hp.get(f"units_layer_{number}"), name=f"hidden_layer_{number}",
                               activation='relu', kernel_regularizer=L2(0.001)))

        if hp.get('use_dropout'):
            model.add(layers.Dropout(rate=hp.get(f"dropout_{number}")))

    # Output Layer
    model.add(layers.Dense(1, activation='linear', kernel_initializer='normal', name='output_layer'))

    # Set up Compiler
    model.compile(optimizer=Adam(learning_rate=hp.get('learning_rate')), loss=hp.get('loss'), metrics=[hp.get('metrics')])
    return model

# </editor-fold>


# <editor-fold desc="Building the Hyperparameter tuner">

def build_tuner(x_train, y_train, x_val, y_val):
    # Create the Hyperband tuner with the objective to minimise the error on the validation data
    tuner = Hyperband(
        build_model, objective='val_mean_absolute_error', max_epochs=20, hyperparameters=HP,
        factor=3, hyperband_iterations=1, directory=LOG_DIR, project_name='P5-Knee-Surgery', seed=seed
    )

    # Split the data in search and validation (80/20 split) and set up early stopping
    stop_early = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=3)

    # Search for the best model and its hyperparameters
    tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val),
                 callbacks=[stop_early], use_multiprocessing=True, workers=8)

    return tuner

# </editor-fold>


# <editor-fold desc="Training the best Model">

def train_hypermodel(target, tuner, x_train, y_train, x_val, y_val):
    best_hp = tuner.get_best_hyperparameters()[0]

    # Build model with the best hyperparameters and train it for 50 epochs
    best_model = tuner.hypermodel.build(best_hp)
    history = best_model.fit(x_train, y_train.ravel(), epochs=50, validation_data=(x_val, y_val))

    # Find the best epoch from the first training
    val_mae_per_epoch = history.history['val_mean_absolute_error']
    best_epoch = val_mae_per_epoch.index(min(val_mae_per_epoch)) + 1

    # Build the best hypermodel from the best hyperparameters
    hypermodel = tuner.hypermodel.build(best_hp)

    # Retrain the model with the best found epoch
    hypermodel.fit(x_train, y_train.ravel(), epochs=best_epoch, validation_data=(x_val, y_val))

    # Save the model to the log directory for future reference
    hypermodel.save(f"{MODEL_DIR}{target}")

    return hypermodel

# </editor-fold>


# <editor-fold desc="Evaluating the Model">

def evaluate_model(x_test, y_test, hypermodel, target):
    expected_y = y_test
    predicted_y = hypermodel.predict(x_test)

    current_r2_score = metrics.r2_score(expected_y, predicted_y)
    current_mse = metrics.mean_squared_error(expected_y, predicted_y)
    current_mae = metrics.mean_absolute_error(expected_y, predicted_y)
    current_rmse = math.sqrt(current_mse)

    # TODO : Return strings or similar, so that the evaluation can be seen for all of the eight models at once
    print(f"This is the evaluation of the {target}")
    print(f"The coefficient of determination (R^2): {current_r2_score}")
    print(f"Mean Absolute Error (MAE): {current_mae}")
    print(f"Root Mean Squared Error (RMSE): {current_rmse}")

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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)

    # Both Validation and Test get 50% each of the remainder
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio))

    # Build a Tuner and search for Hyperparameters on the Train data, test on the Validate data
    tuner = build_tuner(x_train, y_train, x_val, y_val)

    # Build and train the best Model on the Train data, test on the Validate data
    best_model = train_hypermodel(target, tuner, x_train, y_train, x_val, y_val)

    # Evaluate the model using cross-validation on the Test data
    evaluate_model(x_test, y_test, best_model, target)

# </editor-fold>


handle_model(ACL_k)
# TODO : Run for the other models too; but try to utilise the GPU
# handle_model(['ACL_epsr'])
# handle_model(['PCL_k'])
# handle_model(['PCL_epsr'])
# handle_model(['MCL_k'])
# handle_model(['MCL_epsr'])
# handle_model(['LCL_k'])
# handle_model(['LCL_epsr'])
