import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Concrete Dataset (replace 'concrete_train.csv' and 'concrete_test.csv' with actual filenames)
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Separate features and labels
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# (a) Batch Gradient Descent for Linear Regression with Convergence Check
def batch_gradient_descent(X, y, learning_rate=0.01, tolerance=1e-6, n_iterations=1000):
    m, n = X.shape
    X = np.c_[np.ones(m), X]  # Add intercept term
    w = np.zeros(n + 1)
    cost_history = []
    prev_w = np.zeros_like(w)
    
    for i in range(n_iterations):
        predictions = X.dot(w)
        errors = predictions - y
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        cost_history.append(cost)
        
        # Update weights
        prev_w = w.copy()
        w -= (learning_rate / m) * X.T.dot(errors)
        
        # Convergence check
        if np.linalg.norm(w - prev_w) < tolerance:
            print(f'Convergence achieved at iteration {i}')
            break

    return w, cost_history

# Perform Batch Gradient Descent
learning_rate_bgd = 0.01
w_batch, cost_history_bgd = batch_gradient_descent(X_train.to_numpy(), y_train.to_numpy(), learning_rate=learning_rate_bgd)

# Plot cost over iterations for Batch Gradient Descent
plt.plot(cost_history_bgd)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title(f'Batch Gradient Descent: Cost over Iterations (Learning rate = {learning_rate_bgd})')
plt.show()

# Calculate test set cost for Batch Gradient Descent
X_test_with_intercept = np.c_[np.ones(len(X_test)), X_test]
y_test_pred_bgd = X_test_with_intercept.dot(w_batch)
mse_bgd = np.mean((y_test_pred_bgd - y_test) ** 2)
print(f"Mean Squared Error (Batch Gradient Descent) on Test Set: {mse_bgd}")

# (b) Stochastic Gradient Descent for Linear Regression with Convergence Check
def stochastic_gradient_descent(X, y, learning_rate=0.01, tolerance=1e-6, n_iterations=1000):
    m, n = X.shape
    X = np.c_[np.ones(m), X]  # Add intercept term
    w = np.zeros(n + 1)
    cost_history = []
    prev_w = np.zeros_like(w)

    for i in range(n_iterations):
        for j in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]
            prediction = xi.dot(w)
            error = prediction - yi
            prev_w = w.copy()
            w -= learning_rate * xi.T.dot(error)

            # Check for convergence
            if np.linalg.norm(w - prev_w) < tolerance:
                print(f'SGD Converged at iteration {i * m + j}')
                break

        # Track cost at each iteration
        cost = (1 / (2 * m)) * np.sum((X.dot(w) - y) ** 2)
        cost_history.append(cost)
    
    return w, cost_history

# Perform Stochastic Gradient Descent
learning_rate_sgd = 0.01
w_stochastic, cost_history_sgd = stochastic_gradient_descent(X_train.to_numpy(), y_train.to_numpy(), learning_rate=learning_rate_sgd)

# Plot cost over iterations for Stochastic Gradient Descent
plt.plot(cost_history_sgd)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title(f'Stochastic Gradient Descent: Cost over Iterations (Learning rate = {learning_rate_sgd})')
plt.show()

# Calculate test set cost for Stochastic Gradient Descent
y_test_pred_sgd = X_test_with_intercept.dot(w_stochastic)
mse_sgd = np.mean((y_test_pred_sgd - y_test) ** 2)
print(f"Mean Squared Error (Stochastic Gradient Descent) on Test Set: {mse_sgd}")

# (c) Analytical Solution for Linear Regression
X_train_with_intercept = np.c_[np.ones(len(X_train)), X_train]
w_analytical = np.linalg.inv(X_train_with_intercept.T.dot(X_train_with_intercept)).dot(X_train_with_intercept.T).dot(y_train)

print("Analytical Solution Weights:", w_analytical)

# Evaluate on Test Set for Analytical Solution
y_test_pred_analytical = X_test_with_intercept.dot(w_analytical)
mse_analytical = np.mean((y_test_pred_analytical - y_test) ** 2)
print("Mean Squared Error (Analytical Solution) on Test Set:", mse_analytical)

# Compare the learned weight vectors
print("Comparison of Weight Vectors:")
print(f"Batch Gradient Descent Weights:\n{w_batch}")
print(f"Stochastic Gradient Descent Weights:\n{w_stochastic}")
print(f"Analytical Solution Weights:\n{w_analytical}")
