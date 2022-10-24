import keras.callbacks
from keras import layers
from sklearn import metrics
import pandas as pd
import time
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.regularizers import L2
from keras_tuner import RandomSearch, BayesianOptimization, Hyperband
from keras.losses import MeanSquaredLogarithmicError
msle = MeanSquaredLogarithmicError()

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
    _loss = msle
    _metrics = [msle]

    hp_number_of_layers = hp.Int('n_layers', min_value=2, max_value=8)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Create Sequential Model and setup Input Layer
    model = Sequential([layers.Flatten(name='input_layer', input_shape=_input_shape)])

    # Hidden Layers
    for i in range(hp_number_of_layers):
        number = str(i + 1)
        layer_name = 'hidden_layer_' + number
        units_name = 'units_layer_' + number

        hp_units = hp.Int(name=units_name, min_value=32, max_value=512, step=32)

        model.add(layers.Dense(units=hp_units,
                               name=layer_name,
                               activation='relu',
                               kernel_regularizer=_regularizer))

        model.add(layers.Dropout(0.05))

    # Output Layer
    model.add(layers.Dense(1, activation='linear', kernel_initializer='normal', name='output_layer'))

    # Setup Compiler
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss=_loss,
                  metrics=_metrics)
    return model


tuner = BayesianOptimization(
    build_model,
    objective='val_mean_squared_logarithmic_error',
    max_trials=10,
    # executions_per_trial=3,
    directory=LOG_DIR,
    project_name='P5-Knee-Surgery',
    seed=40,
)

stop_early = keras.callbacks.EarlyStopping(monitor='val_mean_squared_logarithmic_error', patience=3)
tuner.search(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), callbacks=[stop_early])
best_model = tuner.get_best_models()[0]

best_model.build()

print("The best model has now been build!")

best_model.fit(
    X_train,
    y_train.ravel(),
    epochs=10,
    batch_size=64
)

expected_y = y_test
predicted_y = best_model.predict(X_test)

current_r2_score = metrics.r2_score(expected_y, predicted_y)
current_msle = metrics.mean_squared_log_error(expected_y, predicted_y)

print("R2 Score: ", current_r2_score)
print("MSLE Score: ", current_msle)

# mean squared logarithmic error
# maybe print this?
# msle(y_test, best_model.predict(X_test)).numpy()
