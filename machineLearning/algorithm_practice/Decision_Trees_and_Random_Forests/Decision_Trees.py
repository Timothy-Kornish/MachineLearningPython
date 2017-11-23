import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


df = pd.read_csv("kyphosis.csv")
print('\n-------------------------------------------------------------------\n')
print(df.head())
print('\n-------------------------------------------------------------------\n')
print(df.info())


"""
sns.pairplot(df, hue = 'Kyphosis')
plt.show()
"""

#-------------------------------------------------------------------------------
#                           Tree Model
#-------------------------------------------------------------------------------

X = df.drop("Kyphosis", axis = 1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

tree_predictions = dtree.predict(X_test)

print("\n-------------------------------------------------------------------\n")
print("Classification report for a single tree")
print(classification_report(y_test, tree_predictions))
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix for a single tree")
print(confusion_matrix(y_test, tree_predictions))

#-------------------------------------------------------------------------------
#                           Random Forest Model
#-------------------------------------------------------------------------------

rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)

rfc_predictions = rfc.predict(X_test)

print("\n-------------------------------------------------------------------\n")
print("Classification report for a random forest of 200 trees")
print(classification_report(y_test, rfc_predictions))
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix for a random forest of 200 trees")
print(confusion_matrix(y_test, rfc_predictions))
print("\n-------------------------------------------------------------------\n")
print("Notice value_counts is diproportionate towards cases absent of kyphosis:\n", df['Kyphosis'].value_counts())

#-------------------------------------------------------------------------------
#      Viewing Tree Model, graphviz will not run, part of whole different software, not python
#-------------------------------------------------------------------------------


from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot

features = list(df.columns[1:])
features

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())
