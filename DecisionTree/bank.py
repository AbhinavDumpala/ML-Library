import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Load the Bank Dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocessing: Encode the target labels and apply one-hot encoding for categorical variables
X_train = pd.get_dummies(train_data.iloc[:, :-1])
y_train = train_data.iloc[:, -1]
X_test = pd.get_dummies(test_data.iloc[:, :-1])
y_test = test_data.iloc[:, -1]

# Align columns in train and test sets
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Encode target labels to numerical values
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# 2(a) AdaBoost with Decision Stumps (batching iterations for faster execution)
train_errors_ada = []
test_errors_ada = []

# Instead of running 500 iterations at once, run in batches of 100 for faster results analysis
for batch in range(1, 6):
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=batch * 100, algorithm='SAMME', random_state=42)
    ada.fit(X_train, y_train)

    # Calculate errors after each batch
    train_pred = ada.predict(X_train)
    test_pred = ada.predict(X_test)
    train_errors_ada.append(1 - accuracy_score(y_train, train_pred))
    test_errors_ada.append(1 - accuracy_score(y_test, test_pred))

    print(f"Batch {batch * 100} iterations completed")

# Plot errors for AdaBoost
plt.plot(range(100, 501, 100), train_errors_ada, label='Training Error') 
plt.plot(range(100, 501, 100), test_errors_ada, label='Test Error')
plt.xlabel('Number of Iterations (T)')
plt.ylabel('Error')
plt.legend()
plt.title('AdaBoost Training and Test Errors (Batched)')
plt.show()


# 2(b) Bagged Trees (500 trees, fully expanded without max depth)
train_errors_bagging = []
test_errors_bagging = []
bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, random_state=42, n_jobs=-1)  # No max_depth restriction

for n_trees in range(1, 501, 100):
    bagging.n_estimators = n_trees
    bagging.fit(X_train, y_train)
    train_pred = bagging.predict(X_train)
    test_pred = bagging.predict(X_test)
    train_errors_bagging.append(1 - accuracy_score(y_train, train_pred))
    test_errors_bagging.append(1 - accuracy_score(y_test, test_pred))

# Plot errors for Bagging
plt.plot(range(1, 501, 100), train_errors_bagging, label='Training Error')
plt.plot(range(1, 501, 100), test_errors_bagging, label='Test Error')
plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.legend()
plt.title('Bagging Training and Test Errors (Fully Expanded Trees)')
plt.show()

# 2(c) Bias-Variance Decomposition for Bagged Trees (1000 samples, 100 repeats)
def bias_variance_decomposition_single_vs_ensemble(model, X_train, y_train, X_test, y_test, n_repeats=100):
    predictions_single = np.zeros((n_repeats, len(X_test)))  # For single tree
    predictions_ensemble = np.zeros((n_repeats, len(X_test)))  # For bagged ensemble

    def train_and_predict(i):
        model.fit(X_train, y_train)
        return model.estimators_[0].predict(X_test), model.predict(X_test)  # First tree and the ensemble predictions

    results = Parallel(n_jobs=-1)(delayed(train_and_predict)(i) for i in range(n_repeats))
    
    for i, (single_tree_pred, ensemble_pred) in enumerate(results):
        predictions_single[i, :] = single_tree_pred
        predictions_ensemble[i, :] = ensemble_pred

    mean_pred_single = np.mean(predictions_single, axis=0)
    mean_pred_ensemble = np.mean(predictions_ensemble, axis=0)
    
    bias_single = np.mean((mean_pred_single - y_test) ** 2)
    variance_single = np.mean(np.var(predictions_single, axis=0))
    
    bias_ensemble = np.mean((mean_pred_ensemble - y_test) ** 2)
    variance_ensemble = np.mean(np.var(predictions_ensemble, axis=0))
    
    error_single = bias_single + variance_single
    error_ensemble = bias_ensemble + variance_ensemble

    return (bias_single, variance_single, error_single), (bias_ensemble, variance_ensemble, error_ensemble)

n_samples = 1000  # Sample size is 1000, as required
X_train_sampled = X_train.sample(n_samples, replace=True)
y_train_sampled = pd.Series(y_train).sample(n_samples, replace=True)

# Bias-Variance Decomposition for Bagging
bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, random_state=42, n_jobs=-1)
(single_tree_stats, bagged_stats) = bias_variance_decomposition_single_vs_ensemble(bagging, X_train_sampled, y_train_sampled, X_test, y_test)

print(f'Single Tree -> Bias: {single_tree_stats[0]}, Variance: {single_tree_stats[1]}, Error: {single_tree_stats[2]}')
print(f'Bagged Trees -> Bias: {bagged_stats[0]}, Variance: {bagged_stats[1]}, Error: {bagged_stats[2]}')


# 2(d) Random Forest (500 trees, varying feature subset size)
train_errors_rf = []
test_errors_rf = []

# Looping through different feature subset sizes (2, 4, 6)
for max_features in [2, 4, 6]:
    train_errors = []
    test_errors = []
    rf = RandomForestClassifier(n_estimators=500, max_features=max_features, random_state=42, n_jobs=-1)
    
    for n_trees in range(1, 501, 100):
        rf.n_estimators = n_trees
        rf.fit(X_train, y_train)
        train_pred = rf.predict(X_train)
        test_pred = rf.predict(X_test)
        train_errors.append(1 - accuracy_score(y_train, train_pred))
        test_errors.append(1 - accuracy_score(y_test, test_pred))

    train_errors_rf.append(train_errors)
    test_errors_rf.append(test_errors)

    # Plot errors for Random Forest with varying feature subset size
    plt.plot(range(1, 501, 100), train_errors, label=f'Training Error (max_features={max_features})')
    plt.plot(range(1, 501, 100), test_errors, label=f'Test Error (max_features={max_features})')

plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.legend()
plt.title('Random Forest Training and Test Errors (Varying Feature Subset Size)')
plt.show()

# 2(e) Bias-Variance Decomposition for Random Forest (1000 samples, 100 repeats)
rf = RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=42, n_jobs=-1)
(single_tree_stats_rf, random_forest_stats) = bias_variance_decomposition_single_vs_ensemble(rf, X_train_sampled, y_train_sampled, X_test, y_test)

print(f'Single Random Tree -> Bias: {single_tree_stats_rf[0]}, Variance: {single_tree_stats_rf[1]}, Error: {single_tree_stats_rf[2]}')
print(f'Random Forest -> Bias: {random_forest_stats[0]}, Variance: {random_forest_stats[1]}, Error: {random_forest_stats[2]}')
