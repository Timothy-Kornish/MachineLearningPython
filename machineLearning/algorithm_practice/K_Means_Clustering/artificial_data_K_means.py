import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#-------------------------------------------------------------------------------
#               K Means Clustering Method on artificial data
#-------------------------------------------------------------------------------

data = make_blobs(n_samples = 200, n_features = 2, centers = 4,
                  cluster_std = 1.8, random_state = 101)

print(data[0].shape)
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#               plotting two seperate blobs of data with 4 groups
#-------------------------------------------------------------------------------

plt.scatter(data[0][:, 0], data[0][:, 1], c = data[1], cmap = 'rainbow')
plt.show()

#-------------------------------------------------------------------------------
#               unsupervised clustering
#-------------------------------------------------------------------------------

kmeans = KMeans(n_clusters = 4)
kmeans.fit(data[0]) # data[0] are the features in the data set, there are 2 features
kmeans3 = KMeans(n_clusters = 3)
kmeans3.fit(data[0])
kmeans2 = KMeans(n_clusters = 2)
kmeans2.fit(data[0])
print("cluster center coordinates:\n", kmeans.cluster_centers_)
print("\n-------------------------------------------------------------------\n")
print("group label name for each point:\n",kmeans.labels_)
print("\n-------------------------------------------------------------------\n")

fig , (ax1, ax2, ax3, ax) = plt.subplots(1, 4, sharey = True, figsize = (10, 6))

ax1.set_title("K Means 4")
ax1.scatter(data[0][:, 0], data[0][:, 1], c = kmeans.labels_)

ax2.set_title("K Means 3")
ax2.scatter(data[0][:, 0], data[0][:, 1], c = kmeans3.labels_)

ax3.set_title("K Means 2")
ax3.scatter(data[0][:, 0], data[0][:, 1], c = kmeans2.labels_)

ax.set_title("Original")
ax.scatter(data[0][:, 0], data[0][:, 1], c = data[1], cmap = 'rainbow')
plt.show()
