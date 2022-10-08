import keras.callbacks
from keras import layers
from keras import metrics
import keras_tuner as kt
import pandas as pd
import time
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.regularizers import L2

LOG_DIR = f"{int(time.time())}"

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

    # Hyperparameters
    number_of_layers = hp.Int("n_layers", 2, 6)
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    units = [hp.Int(name='layer_1_units', min_value=16, max_value=256, step=16),
             hp.Int(name='layer_2_units', min_value=16, max_value=256, step=16),
             hp.Int(name='layer_3_units', min_value=16, max_value=256, step=16),
             hp.Int(name='layer_4_units', min_value=16, max_value=256, step=16),
             hp.Int(name='layer_5_units', min_value=16, max_value=256, step=16),
             hp.Int(name='layer_6_units', min_value=16, max_value=256, step=16)]

    # Parameters
    input_shape = X_train.shape[1:]
    regularizer = L2(0.001)

    model = Sequential([
        layers.Flatten(name='input_layer', input_shape=input_shape)
    ])

    # Hidden Layers
    for i in range(number_of_layers):
        model.add(layers.Dense(
            units=units[i],
            name='hidden_layer_' + str(i + 1),
            activation='relu',
            kernel_regularizer=regularizer)
        )
        # TODO : reduce over fitting the model with -> model.add(layers.Dropout(0.05))

    # Output Layer
    model.add(layers.Dense(1, activation='softmax', name='output_layer'))

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=[metrics.mean_squared_error,
                        metrics.mean_absolute_error],
                  metrics=[metrics.RootMeanSquaredError(),
                           metrics.MeanAbsoluteError()],)
    return model


tuner = kt.RandomSearch(
    build_model,
    objective=kt.Objective("val_root_mean_squared_error", direction='min'),
    max_trials=10,
    executions_per_trial=2,
    directory=LOG_DIR,
    project_name='P5-Knee-Surgery',
    seed=40
)

stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
tuner.search(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_test, y_test), callbacks=[stop_early])
best_model = tuner.get_best_models()[0]

best_model.build()

print(tuner.results_summary())
print(best_model.summary())
