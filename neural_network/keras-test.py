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
df = pd.read_csv('../data_processing/final_final_final.csv', index_col=0)
result_columns = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
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
        number = str(j + 1)

        model.add(layers.Dense(units=hp.get('units_layer_' + number), name='hidden_layer_' + number,
                               activation='relu', kernel_regularizer=L2(0.001)))

        if hp.get('use_dropout'):
            model.add(layers.Dropout(rate=hp.get('dropout_' + number)))

    # Output Layer
    model.add(layers.Dense(1, activation='linear', kernel_initializer='normal', name='output_layer'))

    # Set up Compiler
    model.compile(optimizer=Adam(learning_rate=hp.get('learning_rate')), loss=hp.get('loss'), metrics=[hp.get('metrics')])
    return model

# </editor-fold>


# <editor-fold desc="Building the Hyperparameter tuner">

def build_tuner(x_train, y_train, x_test, y_test):
    # Create the Hyperband tuner with the objective to minimise the error on the validation data
    tuner = Hyperband(
        build_model, objective='val_mean_absolute_error', max_epochs=20, hyperparameters=HP,
        factor=3, hyperband_iterations=1, directory=LOG_DIR, project_name='P5-Knee-Surgery', seed=seed
    )

    # Split the data in search and validation (80/20 split) and set up early stopping
    stop_early = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=3)

    # Search for the best model and its hyperparameters
    tuner.search(x_train, y_train, epochs=50, batch_size=64,
                 validation_data=(x_test, y_test), callbacks=[stop_early],
                 shuffle=True, use_multiprocessing=True, workers=8)

    return tuner

# </editor-fold>


# <editor-fold desc="Training the best Model">

def train_hypermodel(target, tuner, x_train, y_train):
    best_hp = tuner.get_best_hyperparameters()[0]

    # Build model with the best hyperparameters and train it (with 20% of the train data) for 50 epochs
    best_model = tuner.hypermodel.build(best_hp)
    history = best_model.fit(x_train, y_train.ravel(), epochs=50, validation_split=0.20, batch_size=64)

    # Find the best epoch
    val_mae_per_epoch = history.history['val_mean_absolute_error']
    best_epoch = val_mae_per_epoch.index(min(val_mae_per_epoch)) + 1

    # Build the best hypermodel from the best hyperparameters
    hypermodel = tuner.hypermodel.build(best_hp)

    # Retrain the model with the best found epoch
    hypermodel.fit(x_train, y_train.ravel(), epochs=best_epoch, validation_split=0.20, batch_size=64)

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

    print(f"This is the evaluation of the {target}")
    print(f"The coefficient of determination (R^2): {current_r2_score}")
    print(f"Mean Absolute Error (MAE): {current_mae}")
    print(f"Root Mean Squared Error (RMSE): {current_rmse}")

# </editor-fold>


def manage_model(target):
    # Set up the input
    predictors = list(set(list(df.columns)) - set(result_columns))
    df[predictors] = df[predictors] / df[predictors].max()

    # Set up the data set
    x = df[predictors].values
    y = df[target].values

    # Split the data set (80% train, 20% test) for cross-validation using the seed
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=seed)

    # Get the best model and hyperparameters from the tuner
    tuner = build_tuner(x_train, y_train, x_test, y_test)

    # Get the best model and hyperparameters from the tuner
    best_model = train_hypermodel(target, tuner, x_train, y_train)

    # Evaluate the model
    evaluate_model(x_test, y_test, best_model, target)


manage_model(['ACL_k'])
# TODO : Run for the other models too; but try to utilise the GPU
# manage_model(['ACL_espr'])
# manage_model(['PCL_k'])
# manage_model(['PCL_espr'])
# manage_model(['MCL_k'])
# manage_model(['MCL_espr'])
# manage_model(['LCL_k'])
# manage_model(['LCL_espr'])
