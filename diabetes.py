# Import required libraries
import pandas as p
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
d = p.read_csv('diabetes.csv') 
print(d.shape)
d.describe().transpose()
target_column = ['diabetes'] 
predictors = list(set(list(d.columns))-set(target_column))
d[predictors] = df[predictors]/df[predictors].max()
d.describe().transpose()
a = d[predictors].values
b = d[target_column].values

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.30, random_state=40)
print(a_train.shape); print(a_test.shape)
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(a_train,b_train)

predict_train = mlp.predict(a_train)
predict_test = mlp.predict(a_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(b_train,predict_train))
print(classification_report(b_train,predict_train))
print(confusion_matrix(b_test,predict_test))
print(classification_report(b_test,predict_test))
