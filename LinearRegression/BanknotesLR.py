import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Batch Gradient Descent
    def batch_gradient_descent(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y

            # Update weights and bias
            self.weights -= (self.learning_rate / m) * np.dot(X.T, error)
            self.bias -= (self.learning_rate / m) * np.sum(error)

        return self.weights, self.bias

    # Stochastic Gradient Descent
    def stochastic_gradient_descent(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for epoch in range(self.epochs):
            for i in range(m):
                y_pred = np.dot(X[i], self.weights) + self.bias
                error = y_pred - y[i]

                # Update weights and bias
                self.weights -= self.learning_rate * error * X[i]
                self.bias -= self.learning_rate * error

        return self.weights, self.bias

    # Prediction
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Load Data
train_data = pd.read_csv('train.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Instantiate and train Linear Regression model
linear_model = LinearRegression(learning_rate=0.01, epochs=100)

# Batch Gradient Descent
weights_batch, bias_batch = linear_model.batch_gradient_descent(X_train, y_train)
y_pred_batch = linear_model.predict(X_test)
mse_batch = mean_squared_error(y_test, y_pred_batch)
print(f'Batch Gradient Descent Weights: {weights_batch}')
print(f'Batch Gradient Descent Bias: {bias_batch}')
print(f'Batch Gradient Descent Test MSE: {mse_batch}')

# Stochastic Gradient Descent
weights_stochastic, bias_stochastic = linear_model.stochastic_gradient_descent(X_train, y_train)
y_pred_stochastic = linear_model.predict(X_test)
mse_stochastic = mean_squared_error(y_test, y_pred_stochastic)
print(f'Stochastic Gradient Descent Weights: {weights_stochastic}')
print(f'Stochastic Gradient Descent Bias: {bias_stochastic}')
print(f'Stochastic Gradient Descent Test MSE: {mse_stochastic}')