This is a machine learning library developed by Abhinav Dumpala for CS5350/6350 at the University of Utah.
Machine Learning on Slump Test, Concrete, and Bank Datasets
This repository contains implementations of machine learning algorithms like AdaBoost, Bagging, Random Forests, and Linear Regression. The datasets used include:

Bank Dataset: For classification tasks.
Concrete Dataset: For linear regression tasks.
Slump Test Dataset: For regression tasks predicting compressive strength.
How to Run the Code
Clone the repository:

bash
git clone https://github.com/YourUsername/YourRepositoryName.git
cd YourRepositoryName
Install dependencies: Install the required libraries by running:

bash
pip install -r requirements.txt
Change the Dataset Paths:

Before running the code, make sure to update the file paths in the code to point to where the datasets are located on your machine or Colab environment. For example:
python
pd.read_csv('path/to/your/dataset.csv')
Run the scripts:

To run AdaBoost, Bagging, or Random Forest on the Slump Test Dataset:
bash
python concreteslump.py
To run Bagging and Random Forest on the Bank Dataset:
bash
python bank.py
To run Linear Regression on the Concrete Dataset:
bash
python concretes.py


How to Use the Decision Tree Code
The Decision Trees are used in multiple models:

AdaBoost uses decision stumps (max_depth=1).
Bagging uses decision trees with customizable depth.
Random Forest uses a random selection of features with decision trees.
You can change the following parameters in the scripts:

n_estimators: Number of trees/iterations.
max_depth: Maximum depth of the decision trees.
Example:

python
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=500)
ada.fit(X_train, y_train)

Datasets
train.csv and test.csv: For Bank and Concrete datasets.
slump_test.data: For Slump Test dataset.

Outputs
The models will output Mean Squared Error (MSE) for training and testing sets.
Error plots will be generated for various models to visualize performance.
