import seaborn as sns
import matplotlib.pyplot as plt


tips = sns.load_dataset('tips')
#print(tips.head()) # meal and tip afterward

#-------------------------------------------------------------------------------
#                               Distribution Plots
#                   Good for analyzing one column
#-------------------------------------------------------------------------------
"""
sns.distplot(tips['total_bill'], kde = True, bins = 30)
#kde is the line on the histogram
#kde: kernal density estimation
"""
#-------------------------------------------------------------------------------
#                               Joint Plots
#                   Good for comparing relation of two columns
#-------------------------------------------------------------------------------
"""
sns.jointplot(x = 'total_bill', y = 'tip', data = tips, kind = 'kde')
# kind: type of plot in center, defaut = scatter, others: hex, reg (regression), kde (density plot)
"""
#-------------------------------------------------------------------------------
#                               Pair Plots
#                    Good for comparing relation of multiple columns
#-------------------------------------------------------------------------------

"""
sns.pairplot(tips, hue = 'sex', palette = 'coolwarm')
#comparison of different columns in data frame produce scatter Plots
#comparison of the same column in data frame produce histograms
#hue: used to show values in a discrete number of categories in same column (hue = 'sex') only male or femail
"""

#-------------------------------------------------------------------------------
#                               Rug Plots
#                    draws a dash mark for every point on distribution line
#-------------------------------------------------------------------------------

"""
sns.rugplot(tips['total_bill'])
"""

plt.tight_layout()
plt.show()
