import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV

iris = sns.load_dataset("iris")
print(iris.head())

"""

sns.pairplot(data = iris, hue = "species")
plt.show()

setosa = iris[iris['species'] == 'setosa']
sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'], cmap = 'plasma', shade = True, shade_lowest = False)
plt.show()

"""

#-------------------------------------------------------------------------------

X = iris.drop('species', axis = 1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

model = SVC()
model.fit(X_train, y_train)

svc_predictions = model.predict(X_test)

print("\n-------------------------------------------------------------------\n")
print("Classification report for support vector classification")
print(classification_report(y_test, svc_predictions))
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix for support vector classification")
print(confusion_matrix(y_test, svc_predictions))
print("\n-------------------------------------------------------------------\n")

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
