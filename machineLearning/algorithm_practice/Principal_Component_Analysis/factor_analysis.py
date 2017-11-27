#-------------------------------------------------------------------------------
#                 Principal Component Analysis
#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cancer = load_breast_cancer()
print("\n-------------------------------------------------------------------\n")
print("keys of cancer data set:\n", cancer.keys())
print("\n-------------------------------------------------------------------\n")
print("Describe cancer data set:\n", cancer['DESCR'])

df = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
print("\n-------------------------------------------------------------------\n")
print("cancer data frame head:\n", df.head())

scaler = StandardScaler()
scaler.fit(df)

scaler_data = scaler.transform(df)

pca = PCA(n_components = 2)
pca.fit(scaler_data)

x_pca = pca.transform(scaler_data)
print("\n-------------------------------------------------------------------\n")
print("dimension of new scaled data:\n", scaler_data.shape)
print("\n-------------------------------------------------------------------\n")
print("dimension of new scaled pca data:\n", x_pca.shape)


plt.figure(figsize = (8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c = cancer['target'], cmap = 'plasma')
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()

df_comp = pd.DataFrame(pca.components_, columns = cancer['feature_names'])
print("\n-------------------------------------------------------------------\n")
print("pca components head:\n", df.head())

plt.figure(figsize = (12, 8))
heat = sns.heatmap(df_comp, cmap = 'plasma')
heat.set_xticklabels(heat.get_xticklabels(), rotation = 90)
plt.show(heat)
