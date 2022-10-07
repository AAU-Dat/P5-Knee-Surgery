from keras import layers
from keras import metrics
import keras_tuner
import pandas as pd
import time

from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

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
    model = Sequential([
        layers.Flatten(name='Input-Layer', input_shape=X_train.shape[1:])
    ])

    # Hidden Layers
    for i in range(hp.Int("n_layers", 1, 5)):
        model.add(layers.Dense(
            hp.Int('units', min_value=32, max_value=512, step=32),
            name='Hidden-Layer-' + str(i + 1),
            activation='relu')
        )

    # Output Layer
    model.add(layers.Dense(1, activation='softmax', name='Output-Layer'))

    learning_rate = hp.Float("Learning-Rate", min_value=1e-4, max_value=1e-2, sampling="log")

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=[metrics.RootMeanSquaredError()])
    return model


tuner = keras_tuner.RandomSearch(
    build_model,
    objective=keras_tuner.Objective("val_root_mean_squared_error", direction='min'),
    max_trials=10,
    executions_per_trial=2,
    directory=LOG_DIR,
    project_name='p5_neural_network'
)

tuner.search(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_test, y_test))
best_model = tuner.get_best_models()[0]

best_model.build(input_shape=X_train.shape[1:])

print(tuner.results_summary())
print(best_model.summary())
