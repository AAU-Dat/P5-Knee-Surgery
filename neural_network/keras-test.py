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
from keras.losses import MeanSquaredLogarithmicError
from datetime import datetime

msle = MeanSquaredLogarithmicError()
now = datetime.now()

LOG_DIR = 'Test ran on ' + now.strftime("%d-%b-%Y at %H:%M")

# <editor-fold desc="Preparing the Data Set">

df = pd.read_csv('../data_processing/final_final_final.csv', index_col=0)

result_columns = ['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
target_column = ['ACL_k']
predictors = list(set(list(df.columns)) - set(result_columns))
df[predictors] = df[predictors] / df[predictors].max()

X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=69)

# </editor-fold>

# <editor-fold desc="Configure the search space">

HP = kt.HyperParameters()

HP.Int('n_layers', min_value=2, max_value=8)
HP.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
for i in range(8):
    HP.Int(name='units_layer_' + str(i + 1), min_value=32, max_value=512, step=32)
    HP.Float(name='dropout_' + str(i + 1), min_value=0.0, max_value=0.5, default=0.25, step=0.05)

# </editor-fold>

# <editor-fold desc="Building the model for hyperparameters tuning">


def build_model(hp):
    # Create Sequential Model and setup Input Layer
    model = Sequential([layers.Flatten(name='input_layer', input_shape=X_train.shape[1:])])

    # Hidden Layers
    for j in range(hp.get('n_layers')):
        number = str(j + 1)

        model.add(layers.Dense(units=hp.get('units_layer_' + number), name='hidden_layer_' + number,
                               activation='relu', kernel_regularizer=L2(0.001)))

        model.add(layers.Dropout(rate=hp.get('dropout_' + number)))

    # Output Layer
    model.add(layers.Dense(1, activation='linear', kernel_initializer='normal', name='output_layer'))

    # Setup Compiler
    model.compile(optimizer=Adam(learning_rate=hp.get('learning_rate')), loss=msle, metrics=[msle])
    return model

# </editor-fold>

# <editor-fold desc="Optimising the model">


tuner = Hyperband(
    build_model, objective='val_mean_squared_logarithmic_error', max_epochs=20, hyperparameters=HP,
    factor=3, hyperband_iterations=1, directory=LOG_DIR, project_name='P5-Knee-Surgery', seed=69
)

# </editor-fold>

# <editor-fold desc="Finding the best model">

stop_early = keras.callbacks.EarlyStopping(monitor='val_mean_squared_logarithmic_error', patience=3)
tuner.search(X_train, y_train, epochs=50, batch_size=64,
             validation_data=(X_test, y_test), callbacks=[stop_early],
             shuffle=True, use_multiprocessing=True, workers=8)

best_model = tuner.get_best_models()[0]
best_model.build()

print(best_model.summary())

# </editor-fold>

# <editor-fold desc="Training the model">

best_model.fit(X_train, y_train.ravel(), epochs=10, batch_size=64)

# </editor-fold>

# <editor-fold desc="Evaluating the model">

expected_y = y_test
predicted_y = best_model.predict(X_test)

current_r2_score = metrics.r2_score(expected_y, predicted_y)
current_mse = metrics.mean_squared_error(expected_y, predicted_y)
current_mae = metrics.mean_absolute_error(expected_y, predicted_y)
current_msle = metrics.mean_squared_log_error(expected_y, predicted_y)
current_rmse = math.sqrt(current_mse)

print("The coefficient of determination (R^2): ", current_r2_score)
print("Mean Squared Logarithmic Error (MSLE): ", current_msle)
print("Mean Absolute Error (MAE): ", current_mae)
print("Root Mean Squared Error (RMSE): ", current_rmse)

# </editor-fold>
