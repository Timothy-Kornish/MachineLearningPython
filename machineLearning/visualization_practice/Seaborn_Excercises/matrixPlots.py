import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#-------------------------------------------------------------------------------
#                         Matrix Plots
# must show each plot independently or they will overlap on same plot,
# this feature can be useful for combining certain plots
#-------------------------------------------------------------------------------

tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
print("tips table:\n-------------\n", tips.head(),
 '\n============\nflights table:\n------------\n',
  flights.head(), '\n============\n')

#-------------------------------------------------------------------------------
#                         Heat Map
#               Data must be in matrix form
#           can use correlation data, pivot tables, etc
"""
annot: annotation puts value in individual cell block on heat map
cmap: coolwarm, warm, cool, etc.

set_yticklabels(): first parameter requires plot labels, then set rotation

pivot_table:
        index: choose a column to replace the index, basically the rows
        columns: choose a column who's values become individual columns in the head rows
        values: choose a column who's values become correlating values between new rows and columns

linewidth: width of line between cells for seperation and to make more readable
linecolor: color of line between cells

"""
#-------------------------------------------------------------------------------

tc = tips.corr()
print(tc)

heat = sns.heatmap(tc, annot = True, cmap = 'coolwarm')
plt.show(heat)

fpt = flights.pivot_table(index = 'month', columns = 'year', values = 'passengers')

flights_heat = sns.heatmap(fpt, cmap = 'magma', linecolor = 'white', linewidths = 1)
flights_heat.set_yticklabels(flights_heat.get_yticklabels(), rotation = 45)
flights_heat.set_xticklabels(flights_heat.get_xticklabels(), rotation = 45)

plt.show(flights_heat)

#-------------------------------------------------------------------------------
#                         Cluster Map
#               Data must be in matrix form
#           tries to showcase correlation of columns and rows
#           puts similar groups close to each other
"""
metric: 'correlation' will change heirarchy view
        cluster = sns.clustermap(fpt, metric = 'correlation')
stadard_scale: 1 will normalize the values

set axis name angle for cluster map:
plt.setp():
    labels: cluster.ax_heatmap.yaxis.get_majorticklabels() #do not put labels = ...
    rotation = 0, makes words horizontal
"""
#-------------------------------------------------------------------------------

cluster = sns.clustermap(fpt, cmap = 'coolwarm', standard_scale = 1)
plt.setp(cluster.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0)

plt.show(cluster)
