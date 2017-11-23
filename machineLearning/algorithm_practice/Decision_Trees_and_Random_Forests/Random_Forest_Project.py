import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


loans = pd.read_csv("load_data.csv")

print('\n-------------------------------------------------------------------\n')
print(loans.head())
print('\n-------------------------------------------------------------------\n')
print(loans.info())
print('\n-------------------------------------------------------------------\n')
print(loans.describe())

#-------------------------------------------------------------------------------

plt.figure(figsize = (10,6))
loans[loans['credit.policy'] == 1]['fico'].hist(alpha = 0.5, color = "blue",
                                                bins = 30, label = "credit.policy = 1")
loans[loans['credit.policy'] == 0]['fico'].hist(alpha = 0.5, color = "red",
                                                bins = 30, label = "credit.policy = 1")
plt.xlabel("FICO")
plt.legend(loc = "best")
plt.show()

#-------------------------------------------------------------------------------

plt.figure(figsize = (10,6))
loans[loans['not.fully.paid'] == 1]['fico'].hist(alpha = 0.5, color = "blue",
                                                 bins = 30, label = "not.fully.paid = 1")
loans[loans['not.fully.paid'] == 0]['fico'].hist(alpha = 0.5, color = "red",
                                                 bins = 30, label = "not.fully.paid = 1")
plt.xlabel("FICO")
plt.legend(loc = "best")
plt.show()

#-------------------------------------------------------------------------------

sns.countplot(x = "purpose", data = loans, hue = "not.fully.paid")
plt.show()

#-------------------------------------------------------------------------------

sns.jointplot(x = 'fico', y = 'int.rate', data = loans)
plt.show()

#-------------------------------------------------------------------------------

plt.figure(figsize=(11,7))
sns.lmplot(y = 'int.rate', x = 'fico', data = loans, hue = 'credit.policy',
           col = 'not.fully.paid', palette = 'Set1')
plt.show()

#-------------------------------------------------------------------------------
#           Categorical features for a Decision Tree
#-------------------------------------------------------------------------------

cat_feats = ['purpose']
final_data = pd.get_dummies(loans, columns = cat_feats, drop_first = True)
print(final_data.info())


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']

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

rfc = RandomForestClassifier(n_estimators = 600)
rfc.fit(X_train, y_train)

rfc_predictions = rfc.predict(X_test)

print("\n-------------------------------------------------------------------\n")
print("Classification report for a random forest of 600 trees")
print(classification_report(y_test, rfc_predictions))
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix for a random forest of 600 trees")
print(confusion_matrix(y_test, rfc_predictions))
print("\n-------------------------------------------------------------------\n")
