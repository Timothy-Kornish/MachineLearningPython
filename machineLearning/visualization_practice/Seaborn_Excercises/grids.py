import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


iris = sns.load_dataset('iris')
tips = sns.load_dataset('tips')

#-------------------------------------------------------------------------------
#                         Grid Plots
#               Multiple plots arranged in a matrix
#               default: pairplot()

#   pair = sns.pairplot(iris)
#   grid.map(plt.scatter)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#                        Custom Grid Plots
#                           PairGrid
"""
map_diag():  set plots along diagonal to plot type A
map_upper(): set plots above diagonal to plot type B
map_lower(): set plots below diagonal to plot type C
"""
#-------------------------------------------------------------------------------

plt.figure(0)
grid = sns.PairGrid(iris)

grid.map_diag(sns.distplot)
grid.map_upper(plt.scatter)
grid.map_lower(sns.kdeplot)
plt.close(0)

#-------------------------------------------------------------------------------
#                        Custom Grid Plots
#                           FacetGrid
"""
'facet' variable sets up empty grid of plots
"""
#-------------------------------------------------------------------------------

facet = sns.FacetGrid(data = tips, col = 'time', row = 'smoker')
#facet.map(sns.distplot, 'total_bill')
facet.map(plt.scatter, 'total_bill', 'tip')
plt.show(facet)
