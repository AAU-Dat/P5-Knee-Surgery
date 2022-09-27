import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


def gives_x_all_param_header():
    x = []
    for i in range(1, 24):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                       'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                       'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x


# importing data
df = pd.read_csv('../data_processing/final_final_final.csv')

# print(df)


x = df[gives_x_all_param_header()]
y = df[['ACL_k']]

# print(x)
# print(y)

regr = linear_model.LinearRegression()
regr.fit(x, y)

# creating train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6)

# creating a regression model
model = LinearRegression()

# fitting the model
model.fit(x_train, y_train)

# making predictions
predictions = model.predict(x_test)

# model evaluation
print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))
