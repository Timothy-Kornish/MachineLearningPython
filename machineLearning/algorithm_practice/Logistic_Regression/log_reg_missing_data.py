import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#-------------------------------------------------------------------------------
#               Cleaning Data set, filling in null data
#-------------------------------------------------------------------------------


train = pd.read_csv("titanic_train.csv")
print(train.head())

#-------------------------------------------------------------------------------
#               box plot of passengers in each class
#
#  used to find mean age of passengers in each class
#-------------------------------------------------------------------------------

"""

sns.boxplot(x = "Pclass", y = "Age", data = train)
plt.show()

"""

#-------------------------------------------------------------------------------
#               def function to fill in null data
#-------------------------------------------------------------------------------

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train["Age"] = train[["Age", "Pclass"]].apply(impute_age, axis = 1)

#-------------------------------------------------------------------------------
#          Looking at new data set with filled in data on a heatmap
#-------------------------------------------------------------------------------

"""

heat = sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
heat.set_xticklabels(heat.get_xticklabels(), rotation = 45)
plt.title("null values heatmap")
plt.show()

"""

#-------------------------------------------------------------------------------
#          Dropping Cabin column which is around 90% null values
#
#           also dropping null embarked values
#
# use inplace to make the change take affect permenantly
#-------------------------------------------------------------------------------

train.drop("Cabin", axis = 1, inplace = True)
train.dropna(inplace = True)

"""

heat = sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
heat.set_xticklabels(heat.get_xticklabels(), rotation = 45)
plt.title("null values heatmap, no null values now")
plt.show()

"""

#-------------------------------------------------------------------------------
#    converting categorical variables (strings) to dummy variables (numbers)
#
# drop_first removes first column because first column perfectly predicts value
# in second column.
#
# this is known as multiple colinearity
#-------------------------------------------------------------------------------

sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'], drop_first = True)

train = pd.concat([train, sex, embark], axis = 1)
#added new male, Q, S columns and can remove Sex and Embarked columns

#-------------------------------------------------------------------------------
#    dropping irrelevant columns for machine learning algorithm
#-------------------------------------------------------------------------------

train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)
train.drop("PassengerId", axis = 1, inplace = True) # Id is index + 1, therefore irrelevant
print("\n-------------------------------------------------------------------\n")
print("New training data set for machine learning, only numeric values")
print(train.head())

#-------------------------------------------------------------------------------
#    predicting values
#-------------------------------------------------------------------------------

X = train.drop("Survived", axis = 1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 101)

logModel = LogisticRegression()
logModel.fit(X_train, y_train)

predictions = logModel.predict(X_test)

print("\n-------------------------------------------------------------------\n")
print("Classification report")
print(classification_report(y_test, predictions))
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix")
print(confusion_matrix(y_test, predictions))
