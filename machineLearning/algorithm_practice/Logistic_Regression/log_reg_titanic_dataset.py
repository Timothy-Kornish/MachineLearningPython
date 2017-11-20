import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("titanic_train.csv")
print(train.head())

#-------------------------------------------------------------------------------
#               Show columns with null values in data set
#
#  age column only missing about 20% of data, low enough we can use other columns
#  to fill in approximate values for missing data.
#
#  cabin is missing far too much data however
#-------------------------------------------------------------------------------

heat = sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
heat.set_xticklabels(heat.get_xticklabels(), rotation = 45)
plt.title("null values heatmap")
plt.show()

#-------------------------------------------------------------------------------
#               show ratio of survivors to non-survivors with count plot
#-------------------------------------------------------------------------------

sns.set_style("whitegrid")

sns.countplot(x = "Survived", data = train, hue = 'Sex', palette = "RdBu_r")
plt.title("Survivors of Titanic by gender")
plt.show()

sns.countplot(x = "Survived", data = train, hue = 'Pclass')
plt.title("Survivors of Titanic by class")
plt.show()

#-------------------------------------------------------------------------------
#               show histogram of ages of passengers on Titanic
#-------------------------------------------------------------------------------

sns.distplot(train['Age'].dropna(), kde = False, bins = 30)
plt.show()

train['Age'].hist(bins = 50)
plt.show()

#-------------------------------------------------------------------------------
#               show number of siblings and spouses with count plot
#-------------------------------------------------------------------------------

sns.countplot(x= 'SibSp', data = train)
plt.title("Number of passengers with siblings or spouse on board")
plt.show()

#-------------------------------------------------------------------------------
#               histogram of fare cost of passengers
#-------------------------------------------------------------------------------

train['Fare'].hist(bins = 40, figsize = (10, 4))
plt.xlabel("Cost of fare")
plt.ylabel("Number of tickets sold")
plt.title("Distribution of tickets")
plt.show()
