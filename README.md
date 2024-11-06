# Machine Learning Library by Abhinav Dumpala
This is a machine learning library developed by **Abhinav Dumpala** for **CS5350/6350** at the **University of Utah**.

## Machine Learning on Slump Test, Concrete, and Bank Datasets
This repository contains implementations of machine learning algorithms like **AdaBoost**, **Bagging**, **Random Forests**, and **Linear Regression**. The datasets used include:

- **Bank Dataset**: For classification tasks.
- **Concrete Dataset**: For linear regression tasks.
- **Slump Test Dataset**: For regression tasks predicting compressive strength.

## How to Run the Code

### 1. Clone the repository:
```bash
git clone https://github.com/YourUsername/YourRepositoryName.git
cd YourRepositoryName
```
### 2. Install dependencies:
Install the required libraries by running:
```bash
pip install -r requirements.txt
```
### 3. Change the Dataset Paths:
Before running the code, make sure to update the file paths in the code to point to where the datasets are located on your machine or Colab environment. For example:

```python
pd.read_csv('path/to/your/dataset.csv')
```
### 4. Run the scripts:
To run AdaBoost, Bagging, or Random Forest on the **Slump Test Dataset**:
```bash
python concreteslump.py
```
### To run Bagging and Random Forest on the Bank Dataset:
```bash
python bank.py
```
### To run Linear Regression on the Concrete Dataset:
```bash
python concretes.py
```
### How to Use the Decision Tree Code
The **Decision Trees** are used in multiple models:

- **AdaBoost** uses decision stumps (`max_depth=1`).
- **Bagging** uses decision trees with customizable depth.
- **Random Forest** uses a random selection of features with decision trees.

### Changeable Parameters:
- **`n_estimators`**: Number of trees/iterations.
- **`max_depth`**: Maximum depth of the decision trees.
### Example:
```python
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=500)
ada.fit(X_train, y_train)
```
### Datasets:
- **train.csv** and **test.csv**: For Bank and Concrete datasets.
- **slump_test.data**: For Slump Test dataset.

### Outputs:
- The models will output **Mean Squared Error (MSE)** for both training and testing sets.
- **Error plots** will be generated for various models to visualize performance.


# Ensemble Learning Implementation

This implementation includes several ensemble learning algorithms based on Perceptron and decision trees:
- **Standard Perceptron**
- **Voted Perceptron**
- **Averaged Perceptron**
- **AdaBoost**
- **Bagging**
- **Random Forest**

## Requirements
Ensure the following Python libraries are installed:
- `numpy`
- `pandas`
- `sklearn`

Install with:
```bash
pip install numpy pandas scikit-learn
```

# Usage

To execute the various models, modify and run `EnsembleLearning.py`. Below is a brief guide on each model’s usage.

### Standard Perceptron
```python
from EnsembleLearning import standard_perceptron, perceptron_prediction
w_standard = standard_perceptron(X_train, y_train, epochs=10)
y_pred_standard = perceptron_prediction(w_standard, X_test)
```

# Voted Perceptron

```python
from EnsembleLearning import voted_perceptron, voted_prediction
weight_vectors, counts = voted_perceptron(X_train, y_train, epochs=10)
y_pred_voted = voted_prediction(weight_vectors, counts, X_test)
```

# Averaged Perceptron

```python
from EnsembleLearning import averaged_perceptron, perceptron_prediction
w_avg = averaged_perceptron(X_train, y_train, epochs=10)
y_pred_avg = perceptron_prediction(w_avg, X_test)
```

# AdaBoost

```python
from EnsembleLearning import adaboost
adaboost_model = adaboost(X_train, y_train, n_estimators=50)
y_pred_ada = adaboost_model.predict(X_test)
```

# Bagging

```python
from EnsembleLearning import bagging
bagging_model = bagging(X_train, y_train, n_estimators=50)
y_pred_bagging = bagging_model.predict(X_test)
```

# Random Forest

```python
from EnsembleLearning import random_forest
rf_model = random_forest(X_train, y_train, n_estimators=50, max_features='sqrt')
y_pred_rf = rf_model.predict(X_test)
```

# Parameters

- **X_train, y_train, X_test, y_test**: Training and test data loaded from .csv files.
- **epochs**: Number of passes over the training data (10 for perceptron methods).
- **n_estimators**: Number of estimators in AdaBoost, Bagging, and Random Forest (default: 50).
- **max_features**: Number of features to consider in Random Forest (default: 'sqrt').

Experiment by adjusting parameters and observe changes in the performance.

# Linear Regression Implementation

This implementation covers Linear Regression using two optimization methods:

- **Batch Gradient Descent**
- **Stochastic Gradient Descent**

## Requirements

Ensure the following Python libraries are installed:

- `numpy`
- `pandas`

Install with:

```bash
pip install numpy pandas
```

## Usage

To execute the models, edit and run `LinearRegression.py`. The following examples guide you through each method.

### Batch Gradient Descent

```python
from LinearRegression import LinearRegression
linear_model = LinearRegression(learning_rate=0.01, epochs=100)
weights_batch, bias_batch = linear_model.batch_gradient_descent(X_train, y_train)
y_pred_batch = linear_model.predict(X_test)
```

### Stochastic Gradient Descent

```python
from LinearRegression import LinearRegression
linear_model = LinearRegression(learning_rate=0.01, epochs=100)
weights_stochastic, bias_stochastic = linear_model.stochastic_gradient_descent(X_train, y_train)
y_pred_stochastic = linear_model.predict(X_test)
```

## Parameters

- **learning_rate**: Controls the step size for gradient updates (default: 0.01).
- **epochs**: Number of iterations (100 by default) for both batch and stochastic gradient descent.

## Prediction

To make predictions after training, use:

```python
y_pred = linear_model.predict(X_test)
```
By adjusting parameters such as `learning_rate` and `epochs`, you can experiment with the model’s convergence behavior and performance.
