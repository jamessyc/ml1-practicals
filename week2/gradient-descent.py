# regression.py
# parsons/2017-2-05
#
# A simple example using regression.
#
# This illustrates both using the linear regression implmentation that is
# built into scikit-learn and the function to create a regression problem.
#
# Code is based on:
#
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

#
# Generate a regression problem:
#
NUM_FEATURES = 10

# The main parameters of make-regression are the number of samples, the number
# of features (how many dimensions the problem has), and the amount of noise.
X, y = make_regression(n_samples=100, n_features=NUM_FEATURES, noise = 10)

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

learning_rate = 0.05
weight_0 = 5.0
weight_0 = 1.0
weights = [1.0] * NUM_FEATURES
iteration = 0
error = 900000

learning_curve_x = []
learning_curve_y = []
for i in range(3):
    for i in range(len(X_train)):
        iteration += 1
        x_val = X_train[i]
        y_val = y_train[i]
        
        sum_of_weights = 0.0
        for n in range(NUM_FEATURES):
            sum_of_weights += x_val[n] * weights[n]
        predicted_value = weight_0 + sum_of_weights

        error = y_val - predicted_value
        weight_0 = weight_0 + learning_rate * error
        for n in range(NUM_FEATURES):
           weights[n] =  weights[n] + learning_rate * error * x_val[n]

        learning_curve_x.append(iteration)
        learning_curve_y.append(error ** 2)

mse_sum = 0.0
for i in range(len(X_test)):
    x_val = X_test[i]
    y_val = y_test[i]

    sum_of_weights = 0.0
    for n in range(NUM_FEATURES):
        sum_of_weights += x_val[n] * weights[n]
    predicted_value = weight_0 + sum_of_weights

    error = y_val - (predicted_value)
    mse_sum += error ** 2
mse = mse_sum / len(X_test)
print("---Hand-made model results---")
print("Mean squared error: %.2f" % mse)
print()

#
# Solve the problem using the built-in regresson model
#

regr = linear_model.LinearRegression() # A regression model object
regr.fit(X_train, y_train)             # Train the regression model

#
# Evaluate the model
#

# Data on how good the model is:
print("---Built-in model results---")
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))


plt.plot(learning_curve_x, learning_curve_y)
plt.show()

# # Plotting training data, test data, and results.
# plt.scatter(X_train, y_train, color="black")
# plt.scatter(X_test, y_test, color="red")
# plt.scatter(X_test, regr.predict(X_test), color="blue")

# plt.show()




