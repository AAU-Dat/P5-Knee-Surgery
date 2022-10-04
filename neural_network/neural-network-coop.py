import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

#wine = pd.read_csv('wine_data.csv', names=["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"])
wine = pd.read_csv("../data_processing/final_final_final.csv", index_col=0)

print(wine.head())

print(wine.describe().transpose())

# 178 data points with 13 features and 1 label column
print(wine.shape)

X = wine.drop(['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr'], axis=1)
y = wine['ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']

'''
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)

mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

print(len(mlp.coefs_))

print(len(mlp.coefs_[0]))

print(len(mlp.intercepts_[0]))
'''