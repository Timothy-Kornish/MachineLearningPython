import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#-------------------------------------------------------------------------------
#                   Bank Note Data Tensor Flow Project
#-------------------------------------------------------------------------------

bank = pd.read_csv('bank_note_data.csv')
print("\n-------------------------------------------------------------------\n")
print(bank.head())
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#                        Data Set Visualization
#-------------------------------------------------------------------------------
"""
sns.countplot(x = 'Class', data = bank)
plt.show()

sns.pairplot(bank, hue = 'Class')
plt.show()
"""
#-------------------------------------------------------------------------------
#                   Data Preparation - Standard Scaling
#-------------------------------------------------------------------------------

scaler = StandardScaler()

scaler.fit(bank.drop('Class', axis = 1))
scaled_features = scaler.transform(bank.drop("Class", axis = 1))

bank_features = pd.DataFrame(scaled_features, columns = bank.columns[:-1])
print(bank_features.head())
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#                   Training Model Data
#-------------------------------------------------------------------------------

X = bank_features
y = bank['Class']

X = X.as_matrix()
y = y.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#-------------------------------------------------------------------------------
#                   fitting a DNN Classifier for predictions
#-------------------------------------------------------------------------------

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]
classifier = learn.DNNClassifier(hidden_units = [10, 20, 10], n_classes = 2, feature_columns = feature_columns)

classifier.fit(X_train, y_train, steps = 200, batch_size = 32)
bank_predictions = classifier.predict(X_test, as_iterable = False)

print("\n-------------------------------------------------------------------\n")
print("Classification report for Deep Neural Network:\n", classification_report(y_test, bank_predictions))
print("\n-------------------------------------------------------------------\n")
print("Confusion matrix for Deep Neural Network:\n", confusion_matrix(y_test, bank_predictions))
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#                   fitting a Random Forest Classifier for predictions
#-------------------------------------------------------------------------------

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)

print("Classification report for Random Forest:\n", classification_report(y_test, rfc_pred))
print("\n-------------------------------------------------------------------\n")
print("Confusion matrix for Random Forest:\n", confusion_matrix(y_test, rfc_pred))
print("\n-------------------------------------------------------------------\n")
