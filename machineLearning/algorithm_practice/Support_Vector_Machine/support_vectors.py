import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV

cancer = load_breast_cancer()
print('\n-------------------------------------------------------------------\n')
print(cancer.keys())
print('\n-------------------------------------------------------------------\n')
print(cancer['DESCR'])

df_feat = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
print(df_feat.head())
print('\n-------------------------------------------------------------------\n')
print(df_feat.info())
print('\n-------------------------------------------------------------------\n')
print(cancer['target_names'])

X = df_feat
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

model = SVC()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n-------------------------------------------------------------------\n")
print("Classification report for support vector classification")
print(classification_report(y_test, predictions))
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix for support vector classification")
print(confusion_matrix(y_test, predictions))
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
# UndefinedMetricWarning: Precision and F-score are ill-defined and being
# set to 0.0 in labels with no predicted samples.
#
# may help t0 normalize data when passing into a SVM
# search for best parameters using a grid search with sklearn
#-------------------------------------------------------------------------------

param_grid = {'C' : [0.1, 1, 10, 100, 1000], 'gamma' : [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose = 3)
grid.fit(X_train, y_train)
print("\n-------------------------------------------------------------------\n")
print("Best Cross-Validation params:" , grid.best_params_)
print("\n-------------------------------------------------------------------\n")
print("Best Cross-Validation estimator:" , grid.best_estimator_)
print("\n-------------------------------------------------------------------\n")
print("Best Cross-Validation score:" , grid.best_score_)

grid_predictions = grid.predict(X_test)

print("\n-------------------------------------------------------------------\n")
print("Classification report for support vector classification with a grid search")
print(classification_report(y_test, grid_predictions))
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix for support vector classification with a grid search")
print(confusion_matrix(y_test, grid_predictions))
print("\n-------------------------------------------------------------------\n")
print("Note model is now much better up to 95% from 61%")
