import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#-------------------------------------------------------------------------------
#           Logistic Regression Project on simulated Advertising Data
#-------------------------------------------------------------------------------

ad_data = pd.read_csv("advertising.csv")

print(ad_data.head())
print("\n-------------------------------------------------------------------\n")
print(ad_data.info())
print("\n-------------------------------------------------------------------\n")
print(ad_data.describe())


#-------------------------------------------------------------------------------
#           histogram of ages of people looking at ads
#-------------------------------------------------------------------------------

sns.set_style("whitegrid")

ad_data['Age'].hist(bins = 30)
plt.title("Ages of people looking at ads")
plt.xlabel("Age")
plt.show()

#-------------------------------------------------------------------------------
#           joint plot of area income vs age
#-------------------------------------------------------------------------------

sns.jointplot(x = 'Age', y = 'Area Income', data = ad_data)
plt.show()

#-------------------------------------------------------------------------------
#           joint plot showing kde of Daily Time spent on site vs age
#-------------------------------------------------------------------------------

sns.jointplot(x = 'Age', y = 'Daily Time Spent on Site', data = ad_data, kind = "kde", color = "red")
plt.show()

#-------------------------------------------------------------------------------
#           joint plot of Daily Time Spent on Site vs Daily Internet Usage
#-------------------------------------------------------------------------------

sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = ad_data, color = 'green')
plt.show()

#-------------------------------------------------------------------------------
#           pairplot with the hue defined by the 'Clicked on Ad' column feature
#-------------------------------------------------------------------------------

sns.pairplot(ad_data, hue = "Clicked on Ad")
plt.show()

#-------------------------------------------------------------------------------
#           training the model
#-------------------------------------------------------------------------------

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

logModel = LogisticRegression()
logModel.fit(X_train, y_train)

predictions = logModel.predict(X_test)

print("\n-------------------------------------------------------------------\n")
print("Classification report")
print(classification_report(y_test, predictions))
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix")
print(confusion_matrix(y_test, predictions))
