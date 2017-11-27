import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report

college = pd.read_csv("College_Data.csv", index_col = 0)
print("\n-------------------------------------------------------------------\n")
print(college.head())
print("\n-------------------------------------------------------------------\n")
print(college.info())
print("\n-------------------------------------------------------------------\n")
print(college.describe())
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#           Exploratory Data Analysis
#-------------------------------------------------------------------------------

"""

sns.lmplot(data = college, x = "Room.Board",  y = "Grad.Rate", fit_reg = False, hue = "Private")
plt.show()

sns.lmplot(data = college, x = "Outstate", y = "F.Undergrad", fit_reg = False, hue = "Private")
plt.show()

facet = sns.FacetGrid(data = college, palette = "coolwarm", hue = "Private")
facet.map(plt.hist, "Outstate", bins = 20, alpha = 0.5)
plt.show(facet)

facet = sns.FacetGrid(data = college, palette = "coolwarm", hue = "Private")
facet.map(plt.hist, "Grad.Rate", bins = 20, alpha = 0.5)
plt.show(facet)

"""

print(college[college["Grad.Rate"] > 100])

college["Grad.Rate"]["Cazenovia College"] = 100

"""
facet = sns.FacetGrid(data = college, palette = "coolwarm", hue = "Private")
facet.map(plt.hist, "Grad.Rate", bins = 20, alpha = 0.5)
plt.show(facet)
"""

kmeans = KMeans(n_clusters = 2)
kmeans.fit(college.drop("Private", axis = 1))

print("cluster center vector coordinates:\n", kmeans.cluster_centers_)
print("\n-------------------------------------------------------------------\n")

college["Cluster"] = college["Private"].apply(lambda x: 1 if x == "Yes" else 0)
print(college.head())

print("\n-------------------------------------------------------------------\n")
print("Classification report for a 2 cluster K-Means-Cluster")
print(classification_report(college['Cluster'], kmeans.labels_))
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix for a 2 cluster K-Means-Cluster")
print(confusion_matrix(college['Cluster'], kmeans.labels_))
