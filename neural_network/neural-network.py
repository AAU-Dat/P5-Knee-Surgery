'''import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("../data_processing/final_final_final.csv")

data = np.array(data)
amount_of_rows, amount_of_columns = data.shape
test_data_size = int(np.ceil(amount_of_rows * .2))
np.random.shuffle(data)

data_dev = data[0:test_data_size].T
Y_dev = data_dev[0]
X_dev = data_dev[8:amount_of_columns]

data_train = data[test_data_size:amount_of_rows]
Y_train = data_train[0]
X_train = data_train[8:amount_of_columns]


def init_params():
    w1 = np.random.rand(8, 277) - 0.5
    b1 = np.random.rand(8, 1) - 0.5
    w2 = np.random.rand(8, 8) - 0.5
    b2 = np.random.rand(8, 1) - 0.5
    return w1, b1, w2, b2


def relu(z):
    return np.maximum(z, 0)


def softmax(z):
    a = np.exp(z) / sum(np.exp(z))
    return a


def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def relu_derive(z):
    return z > 0


# todo : might need to change the size of the table here
def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def backward_prop(z1, a1, z2, a2, w1, w2, x, y):
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / amount_of_rows * dz2.dot(a1.T)
    db2 = 1 / amount_of_rows * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * relu_derive(z1)
    dw1 = 1 / amount_of_rows * dz1.dot(x.T)
    db1 = 1 / amount_of_rows * np.sum(dz1)
    return dw1, db1, dw2, db2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2


def get_predictions(a2):
    return np.argmax(a2, 0)


def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size


def gradient_descent(x, y, alpha, iterations):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backward_prop(z1, a1, z2, a2, w1, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(a2)
            print(get_accuracy(predictions, y))
    return w1, b1, w2, b2


w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
'''