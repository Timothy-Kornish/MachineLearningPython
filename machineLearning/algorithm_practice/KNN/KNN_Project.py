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
#                    K-Nearest-Neighbors Project
#-------------------------------------------------------------------------------

knn_df = pd.read_csv('KNN_Project_Data.csv')
print(knn_df.head())

#-------------------------------------------------------------------------------
#                    pairplot of dataframe
#-------------------------------------------------------------------------------

"""

sns.pairplot(knn_df, hue = 'TARGET CLASS', palette = 'coolwarm')
plt.show()

"""

#-------------------------------------------------------------------------------
#                    scaling the data
#-------------------------------------------------------------------------------

scaler = StandardScaler()
scaler.fit(knn_df.drop("TARGET CLASS", axis = 1))

scaled_features = scaler.transform(knn_df.drop("TARGET CLASS", axis = 1))

knn_df_feat = pd.DataFrame(scaled_features, columns = knn_df.columns[:-1])
print("\n-------------------------------------------------------------------\n")
print("Scaled data frame")
print(knn_df_feat.head())

X = knn_df_feat
y = knn_df['TARGET CLASS']

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
#-------------------------------------------------------------------------------

error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize = (10, 6))
plt.plot(range(1, 40), error_rate, color = 'b', linestyle = '--', marker = 'o', markerfacecolor = 'r', markersize = 10)
plt.title("Error Rate vs K Value")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.show()

#-------------------------------------------------------------------------------
#    using optimal k-value (17 is decent, 31 is better but pretty large for minimal gain)
#-------------------------------------------------------------------------------

knn = KNeighborsClassifier(n_neighbors = 17)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print("\n-------------------------------------------------------------------\n")
print("Classification report: k = 17")
print(classification_report(y_test, predictions))
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix: k = 17")
print(confusion_matrix(y_test, predictions))
