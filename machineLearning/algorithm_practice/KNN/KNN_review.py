import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#-------------------------------------------------------------------------------
#                       K-Nearest-Neighbors
#-------------------------------------------------------------------------------

df = pd.read_csv("Classified_Data.csv", index_col = 0)
print("\n-------------------------------------------------------------------\n")
print(df.head())

#-------------------------------------------------------------------------------
#         Scale classified columns to predict a target class
#
# scale the data so all columns have a mean of 0 and standard deviation of 1
# this makes it so KNN treats distances between data points equally
#-------------------------------------------------------------------------------

scaler = StandardScaler()
scaler.fit(df.drop("TARGET CLASS", axis = 1))

scaled_features = scaler.transform(df.drop("TARGET CLASS", axis = 1))

df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1]) # all columns but the last

print("\n-------------------------------------------------------------------\n")
print(df_feat.head())

X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 101)

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

print("\n-------------------------------------------------------------------\n")
print("Classification report: k = 1")
print(classification_report(y_test, predictions))
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix: k = 1")
print(confusion_matrix(y_test, predictions))

#-------------------------------------------------------------------------------
#            Elbow method to choose the best k-value for data set
#
# np.mean(pred_i != y_test) is the average error rate for each i
#-------------------------------------------------------------------------------

error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize = (10, 6))
plt.plot(range(1,40), error_rate, color = 'b', linestyle = '--', marker = 'o', markerfacecolor = 'r', markersize = 10)
plt.title("Error Rate vs K Value")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.show()

#-------------------------------------------------------------------------------
#            new confustion matrix with a a better fitted k-value
#-------------------------------------------------------------------------------


knn = KNeighborsClassifier(n_neighbors = 12)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

print("\n-------------------------------------------------------------------\n")
print("Classification report: k = 12")
print(classification_report(y_test, predictions))
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix: k = 12")
print(confusion_matrix(y_test, predictions))
