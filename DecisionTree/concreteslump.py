import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the Slump Test dataset
column_names = [
    "No", "Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr.", 
    "Fine Aggr.", "SLUMP(cm)", "FLOW(cm)", "Compressive Strength (28-day)(Mpa)"
]
data = pd.read_csv('slump_test.data', names=column_names, skiprows=1)

# Remove 'No' column
data = data.drop(columns=['No'])

# Separate features (X) and target (y)
X = data.drop(columns=["Compressive Strength (28-day)(Mpa)"])  # Features
y = data["Compressive Strength (28-day)(Mpa)"]  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AdaBoost with Decision Trees (500 iterations)
train_errors_ada = []
test_errors_ada = []
T_values = range(1, 501) 
for T in T_values:
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=T, algorithm='SAMME', random_state=42)
    ada.fit(X_train, y_train.astype('int'))

    # Calculate errors
    train_pred = ada.predict(X_train)
    test_pred = ada.predict(X_test)
    train_errors_ada.append(mean_squared_error(y_train, train_pred))
    test_errors_ada.append(mean_squared_error(y_test, test_pred))

# Plot errors for AdaBoost
plt.plot(T_values, train_errors_ada, label='Training Error')
plt.plot(T_values, test_errors_ada, label='Test Error')
plt.xlabel('Number of Iterations (T)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('AdaBoost Training and Test Errors (Slump Test) - 500 Iterations')
plt.show()


# Bagged Trees with 500 trees
train_errors_bagging = []
test_errors_bagging = []
tree_values = range(1, 501)

for n_trees in tree_values:
    bagging = BaggingClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=n_trees, random_state=42)
    bagging.fit(X_train, y_train.astype('int'))

    # Calculate errors
    train_pred = bagging.predict(X_train)
    test_pred = bagging.predict(X_test)
    train_errors_bagging.append(mean_squared_error(y_train, train_pred))
    test_errors_bagging.append(mean_squared_error(y_test, test_pred))

# Plot errors for Bagging
plt.plot(tree_values, train_errors_bagging, label='Training Error')
plt.plot(tree_values, test_errors_bagging, label='Test Error')
plt.xlabel('Number of Trees')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Bagging Training and Test Errors (Slump Test) - 500 Trees')
plt.show()


# Random Forest with 500 trees
train_errors_rf = []
test_errors_rf = []
tree_values = range(1, 501)

for n_trees in tree_values:
    rf = RandomForestClassifier(n_estimators=n_trees, max_features='sqrt', max_depth=5, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train.astype('int'))

    # Calculate errors
    test_pred = rf.predict(X_test)
    test_errors_rf.append(mean_squared_error(y_test, test_pred))

# Plot errors for Random Forest
plt.plot(tree_values, test_errors_rf, label='Test Error')
plt.xlabel('Number of Trees')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Random Forest Test Errors (Slump Test) - 500 Trees')
plt.show()

# Fully Expanded Single Decision Tree for Comparison (use DecisionTreeRegressor for regression)
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)

# Calculate errors
train_pred_tree = tree.predict(X_train)
test_pred_tree = tree.predict(X_test)

train_error_tree = mean_squared_error(y_train, train_pred_tree)
test_error_tree = mean_squared_error(y_test, test_pred_tree)

print(f'Fully Expanded Decision Tree -> Training Error: {train_error_tree}, Test Error: {test_error_tree}')
