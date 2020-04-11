# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, PReLU, ELU, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.activations import relu, sigmoid
from sklearn.metrics import confusion_matrix, accuracy_score


# Import the .csv file
dataset = pd.read_csv("data/Churn_Modelling.csv")
x = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]
# print(dataset.info())

# Create dummy variables
geography = pd.get_dummies(x['Geography'], drop_first=True)
gender = pd.get_dummies(x['Gender'], drop_first=True)
# print(gender.head(20))

# Concatenate the dummies and to the original dataset and drop irrelevant columns
x = pd.concat([x, geography, gender], axis=1)
x = x.drop(['Geography', 'Gender'], axis=1)

# Split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim=x_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))

    model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, verbose=0)

layers = [[20], [40, 20], [45, 30, 15]]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size=[128, 256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_result = grid.fit(x_train, y_train)

# Model test results
print(grid_result.best_score_, grid_result.best_params_)

pred_y = grid.predict(x_test)
y_pred = (pred_y > 0.5)

cm = confusion_matrix(y_pred, y_test)
score = accuracy_score(y_pred, y_test)