import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

customers = pd.read_csv('Ecommerce_Customers.csv')

print("\n----------=========================----------\n")
print(customers.head())
print("\n----------=========================----------\n")
print(customers.info())
print("\n----------=========================----------\n")
print(customers.describe())
print("\n----------=========================----------\n")

#-------------------------------------------------------------------------------
#                Exploratory Data Analysis
#-------------------------------------------------------------------------------

"""
sns.jointplot(x = "Time on Website", y = "Yearly Amount Spent", data = customers)
# p is 0.95 suggesting no relation
plt.show()

sns.jointplot(x = "Time on App", y = "Yearly Amount Spent", data = customers)
# p is 6.9e-33 suggesting probable relation
plt.show()

sns.jointplot(x = "Time on App", y = "Length of Membership", data = customers, kind = "hex")
# p is 0.52 suggesting probably little to no relation
plt.show()

sns.pairplot(customers)
#Length of Membership and Yearly Amount Spent seem to be the only columns related to each others
plt.show()

sns.lmplot(x = "Yearly Amount Spent", y = "Length of Membership", data = customers)
#lm -> linear model, good for linear regression plots
plt.show()
"""

#-------------------------------------------------------------------------------
#                Exploratory Data Analysis
#-------------------------------------------------------------------------------

print(customers.columns)
print("\n----------=========================----------\n")

X = customers[['Avg. Session Length', 'Time on App',
               'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

#-------------------------------------------------------------------------------
#                Training the Model
#-------------------------------------------------------------------------------

lm = LinearRegression()
lm.fit(X_train, y_train)

print("training model coefficients: ", lm.coef_)
print("\n----------=========================----------\n")

#-------------------------------------------------------------------------------
#                Predicting Test Data
#-------------------------------------------------------------------------------

predictions = lm.predict(X_test) # y-values produced from known x-values

plt.scatter(y_test, predictions)
plt.title("Y-Values vs. Predicted Y-Values")
plt.xlabel("Y Test")
plt.ylabel("Predicted Y")
plt.show()

#-------------------------------------------------------------------------------
#                Evaluate Error on Data using sklearn.metrics
#-------------------------------------------------------------------------------

mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("MAE: ", mae)
print("\n----------=========================----------\n")
print("MSE: ", mse)
print("\n----------=========================----------\n")
print("RMSE: ", rmse)
print("\n----------=========================----------\n")

#-------------------------------------------------------------------------------
#                Residuals
#-------------------------------------------------------------------------------

sns.distplot((y_test - predictions), bins = 50)
plt.title("Residuals")
plt.show()

#-------------------------------------------------------------------------------
#                create DataFrame of Coefficients
#-------------------------------------------------------------------------------

cdf = pd.DataFrame(lm.coef_, X.columns, columns = ['Coefficient'])
print(cdf.head())
#mobile app has a greater coef than the website, more time should be invested there
