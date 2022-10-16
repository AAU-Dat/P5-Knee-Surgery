import keras.callbacks
from keras import layers
from keras import metrics
import keras as kt
import pandas as pd
import time
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.regularizers import L2
from keras_tuner import RandomSearch
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

LOG_DIR = f"{int(time.time())}"

# Import Dataset and prepare for Training
df = pd.read_csv('../data_processing/final_final_final.csv', index_col=0)
df.describe().transpose()

result_columns = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
target_column = ['ACL_k']
predictors = list(set(list(df.columns)) - set(result_columns))
df[predictors] = df[predictors] / df[predictors].max()
df.describe().transpose()

X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)


def build_model(hp):
    """
    Build model for hyperparameters tuning

    hp: HyperParameters class instance
    """

    _input_shape = X_train.shape[1:]
    _regularizer = L2(0.001)

    hp_number_of_layers = hp.Int('n_layers', min_value=2, max_value=8)

    # Create Sequential Model and setup Input Layer
    model = Sequential([layers.Flatten(name='input_layer', input_shape=_input_shape)])

    # Hidden Layers
    for i in range(hp_number_of_layers):
        number = str(i + 1)
        layer_name = 'hidden_layer_' + number
        units_name = 'units_layer_' + number

        hp_units = hp.Int(name=units_name, min_value=16, max_value=512, step=16)

        model.add(layers.Dense(units=hp_units,
                               name=layer_name,
                               activation='relu',
                               kernel_regularizer=_regularizer))

        model.add(layers.Dropout(0.05))

    # Output Layer
    model.add(layers.Dense(1, activation='linear', name='output_layer'))

    _loss = 'mean_absolute_error'
    _metrics = ['mean_absolute_error']

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Setup Compiler
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss=_loss,
                  metrics=_metrics)
    return model


tuner = RandomSearch(
    build_model,
    objective='val_mean_absolute_error',
    max_trials=100,
    executions_per_trial=3,
    directory=LOG_DIR,
    project_name='P5-Knee-Surgery',
    seed=40,
)

stop_early = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=3)
tuner.search(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), callbacks=[stop_early])
best_model = tuner.get_best_models()[0]

best_model.build()

print(tuner.results_summary())
print(best_model.summary())
