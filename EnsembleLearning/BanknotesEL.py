import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, epochs=10):
        self.epochs = epochs

    # Standard Perceptron
    def standard_perceptron(self, X, y):
        w = np.zeros(X.shape[1])
        for epoch in range(self.epochs):
            for i in range(len(X)):
                if y[i] * np.dot(w, X[i]) <= 0:
                    w += y[i] * X[i]
        return w

    # Voted Perceptron
    def voted_perceptron(self, X, y):
        w = np.zeros(X.shape[1])
        weight_vectors = []
        counts = []
        count = 1

        for epoch in range(self.epochs):
            for i in range(len(X)):
                if y[i] * np.dot(w, X[i]) <= 0:
                    weight_vectors.append(w.copy())
                    counts.append(count)
                    w += y[i] * X[i]
                    count = 1
                else:
                    count += 1

        weight_vectors.append(w.copy())
        counts.append(count)
        return weight_vectors, counts

    # Averaged Perceptron
    def averaged_perceptron(self, X, y):
        w = np.zeros(X.shape[1])
        w_avg = np.zeros(X.shape[1])
        for epoch in range(self.epochs):
            for i in range(len(X)):
                if y[i] * np.dot(w, X[i]) <= 0:
                    w += y[i] * X[i]
                w_avg += w
        return w_avg / (self.epochs * len(X))

    # Prediction for Standard and Averaged Perceptron
    def predict(self, w, X):
        return np.sign(np.dot(X, w))

    # Prediction for Voted Perceptron
    def voted_prediction(self, weight_vectors, counts, X):
        predictions = np.zeros(len(X))
        for w, c in zip(weight_vectors, counts):
            predictions += c * np.sign(np.dot(X, w))
        return np.sign(predictions)

# Load Data
train_data = pd.read_csv('train.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)

X_train = train_data.iloc[:, :-1].values
y_train = np.where(train_data.iloc[:, -1].values == 0, -1, 1)
X_test = test_data.iloc[:, :-1].values
y_test = np.where(test_data.iloc[:, -1].values == 0, -1, 1)

# Instantiate Perceptron
perceptron = Perceptron(epochs=10)

# Standard Perceptron
w_standard = perceptron.standard_perceptron(X_train, y_train)
y_pred_standard = perceptron.predict(w_standard, X_test)
error_standard = np.mean(y_pred_standard != y_test)
print(f'Standard Perceptron - Learned Weights: {w_standard}')
print(f'Standard Perceptron - Test Error: {error_standard}')

# Voted Perceptron
weight_vectors, counts = perceptron.voted_perceptron(X_train, y_train)
y_pred_voted = perceptron.voted_prediction(weight_vectors, counts, X_test)
error_voted = np.mean(y_pred_voted != y_test)
print(f'Voted Perceptron - Test Error: {error_voted}')

# Averaged Perceptron
w_avg = perceptron.averaged_perceptron(X_train, y_train)
y_pred_avg = perceptron.predict(w_avg, X_test)
error_avg = np.mean(y_pred_avg != y_test)
print(f'Averaged Perceptron - Learned Weights: {w_avg}')
print(f'Averaged Perceptron - Test Error: {error_avg}')