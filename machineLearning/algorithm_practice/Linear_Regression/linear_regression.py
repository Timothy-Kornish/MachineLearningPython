import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

df = pd.read_csv("USA_Housing.csv") # this is artificial data

print(df.head())
print("\n----------=========================----------\n")
print(df.describe())

print("\n----------=========================----------\n")
print("column names:\n", df.columns)

"""
sns.pairplot(df)
plt.show()

sns.distplot(df['Price'])
plt.show()

sns.heatmap(df.corr())
plt.show()
"""

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y = df['Price']

# test_size is a percentage of data being tested, 0.4 = 40%, 30-40% is usually good

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 101)

# Now with test and training data, we create the model

lm = LinearRegression()

lm.fit(X_train, y_train)

print("\n----------=========================----------\n")
print("Linear model intercept:", lm.intercept_)
print("Linear model coefficient:", lm.coef_)

print("\n----------=========================----------\n")
print("X_train columns:\n", X_train.columns)

print("\n----------=========================----------\n")
cdf = pd.DataFrame(lm.coef_, X.columns, columns = ["Coeff"])
print(cdf.head())

boston = load_boston()

print("\n----------=========================----------\n")
print("Keys of boston dataset: ", boston.keys())

print("\n----------=========================----------\n")
print("Description of Boston dataset:\n",boston['DESCR'])

print("\n----------=========================----------\n")
print("targets of Boston dataset:\n",boston['target'])

#-------------------------------------------------------------------------------
#               Making Predictions from test set
#-------------------------------------------------------------------------------

predictions = lm.predict(X_test)

print("\n----------=========================----------\n")
print("Predictions in Boston: ", predictions)

plt.scatter(y_test, predictions)
plt.title("Y_test values vs. predicted values")
plt.ylabel("prediction values")
plt.xlabel("Y_test values")
plt.show()

sns.distplot((y_test - predictions))
plt.title("Residuals: normalized gaussian means good prediction for data")
plt.show()


#-------------------------------------------------------------------------------
#              Regression Evaluation Metrics
#
# Mean Absolute Error (MAE): average of sum of residuals (difference between actual value and predicted value)
""" Easy to understand, average error """
# Mean Squared Error (MSE): average of sum of squared residuals
""" better than MAE, MSE punishes large errors, tends to be useful in real world """
# Root Mean Squared Error (RMSE): square root of average of sum of squared residuals
""" better than MSE, RMSE is interpretable in y-units """
#-------------------------------------------------------------------------------

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, predictions)
print("\n----------=========================----------\n")
print("Mean Absolute Error: ", mae)

mse = metrics.mean_squared_error(y_test, predictions)
print("\n----------=========================----------\n")
print("Mean Squared Error: ", mse)

rmse = np.sqrt(mse)
print("\n----------=========================----------\n")
print("Root Mean Squared Error: ", rmse)
