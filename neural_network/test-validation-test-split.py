import pandas as pd
from sklearn.model_selection import train_test_split

ACL_k, ACL_epsr, PCL_k, PCL_epsr = 'ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr'
MCL_k, MCL_epsr, LCL_k, LCL_epsr = 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr'

# Data Set preparation
df = pd.read_csv('../data_processing/final_final_final.csv', index_col=0)
result_columns = [ACL_k, ACL_epsr, PCL_k, PCL_epsr, MCL_k, MCL_epsr, LCL_k, LCL_epsr]
input_shape = (276,)                            # input is a row of entries (284 - [8 result columns])
seed = 69                                       # to get reproducible data splits and hyperparameters
train_ratio = 0.80                              # percentage of the data set allocated to train and tune
validation_ratio = 0.10                         # percentage of the data set allocated to validation
test_ratio = 0.10                               # percentage of the data set allocated to evaluation

# Set up the input
predictors = list(set(list(df.columns)) - set(result_columns))
df[predictors] = df[predictors] / df[predictors].max()

# Set up the data set
x = df[predictors].values
y = df[ACL_k].values

print(x.shape)

num_rows, num_cols = x.shape
full_size = num_rows

# Train is now 80% of the entire data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)

# Both test and validation is now 10% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                test_size=test_ratio / (test_ratio + validation_ratio))

# <editor-fold desc="Check that we get the correct percentages">

train_size, train_cols = x_train.shape
validate_size, validate_cols = x_val.shape
test_size, test_cols = x_test.shape

train_percentage = f"{train_size / full_size}"
validate_percentage = f"{validate_size / full_size}"
test_percentage = f"{test_size / full_size}"

train = f"Train: {train_size} (actual: {train_percentage} %, target: {train_ratio * 100} %)"
validate = f"Validate: {validate_size} (actual: {validate_percentage} %, target: {validation_ratio * 100} %)"
test = f"Test: {test_size} (actual: {test_percentage} %, target: {test_ratio * 100} %)"

new_line = '\n'

print(f"{train} {new_line}{validate} {new_line}{test}")

# </editor-fold>
