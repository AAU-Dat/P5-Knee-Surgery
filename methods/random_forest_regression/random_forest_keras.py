import math
from datetime import datetime
import keras.callbacks
import keras_tuner as kt
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import keras_tuner as kt
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import L2
from keras_tuner import Hyperband
from sklearn import metrics
from sklearn.model_selection import train_test_split

#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

now = datetime.now().strftime("%d-%b-%Y at %H:%M")

ACL_k, ACL_epsr, PCL_k, PCL_epsr = 'ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr'
MCL_k, MCL_epsr, LCL_k, LCL_epsr = 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr'
df = pd.read_csv('../data_processing/final_final_final.csv', index_col=0)
result_columns = [ACL_k, ACL_epsr, PCL_k, PCL_epsr, MCL_k, MCL_epsr, LCL_k, LCL_epsr]

seed_value = 69                                 # to get reproducible data splits and hyperparameters
train_ratio = 0.80                              # percentage of the data set allocated to train and tune
validation_ratio = 0.10                         # percentage of the data set allocated to validation
test_ratio = 0.10                               # percentage of the data set allocated to evaluation

LOG_DIR = f"Test ran on {now}"
MODEL_DIR = f"{LOG_DIR}/models/"

HP = kt.HyperParameters()

#obj = kt.Objective('root_mean_Squared_error', direction=min)

def build_tuner(x_train, y_train, x_val, y_val):
    # Create the Hyperband tuner with the objective to minimise the error on the validation data
    keras_tuner = kt.Hyperband(
        hypermodel=build_model,
        objective='val_mean_absolute_error',
        seed=seed_value,
        hyperband_iterations=2,
        factor=3,
        max_epochs=1, # changed from 20 to 1
        directory=LOG_DIR,
        project_name='P5-Knee-Surgery'
    )
    # Split the data in search and validation (80/20 split) and set up early stopping
    stop_early = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=3)
    # Search for the best model and its hyperparameters
    keras_tuner.search(x_train, y_train, epochs=3, validation_data=(x_val, y_val), callbacks=[stop_early], use_multiprocessing=True, workers=7) #epochs changed to 1

    return keras_tuner

# specify regression task
def build_model(hp):
    model = tfdf.keras.RandomForestModel(
        task=tfdf.keras.Task.REGRESSION,
        #min_examples=hp.Choice("min_examples", [2, 5, 7, 10]),
        #categorical_algorithm=hp.Choice("categorical_algorithm", ["CART", "RANDOM"]),
        max_depth=hp.Choice("max_depth", [4, 5, 6, 7]),
        num_trees=hp.Choice("num_trees", [1,2,3,4,5]),
        # The keras tuner convert automaticall boolean parameters to integers.
        #shrinkage=hp.Choice("shrinkage", [0.02, 0.05, 0.10, 0.15]),
        #num_candidate_attributes_ratio=hp.Choice("num_candidate_attributes_ratio", [0.2, 0.5, 0.9, 1.0])
    )
    # Optimize the model accuracy as computed on the validation dataset.
    model.compile(metrics=["mean_absolute_error"])

    return model


def train_hypermodel(target, tuner, x_train, y_train, x_val, y_val):
    best_hp = tuner.get_best_hyperparameters()[0]

    # Build model with the best hyperparameters and train it for 50 epochs
    best_model = tuner.hypermodel.build(best_hp)
    history = best_model.fit(x_train, y_train.ravel(), epochs=1, validation_data=(x_val, y_val)) # changed from 3 to 1

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

handle_model(ACL_k)
