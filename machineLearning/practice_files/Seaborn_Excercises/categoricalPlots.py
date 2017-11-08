import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#-------------------------------------------------------------------------------
#                         Categorical Plots
# must show each plot independently or they will overlap on same plot,
# this feature can be useful for combining certain plots
#-------------------------------------------------------------------------------

tips = sns.load_dataset('tips')
# print(tips.head()) show columns and data of dataset

#-------------------------------------------------------------------------------
#                         Bar Plot
"""
x is Categorical
y is numeric
data is data set used
estimator is an aggregator function
for other options in plot line in jupyter notebook, press shift + tab
"""
#-------------------------------------------------------------------------------
ax = plt.axes()
bar = sns.barplot(x = 'sex', y = 'total_bill', data = tips, ax = ax, estimator = np.std)
ax.set_title('average bill by gender')
plt.show(bar)
#-------------------------------------------------------------------------------
#                         Count  Plots
"""
 only choose x and dataset
 y is the count for each x and is set automatically
"""
#-------------------------------------------------------------------------------

count = sns.countplot(x = 'sex', data = tips)
plt.show(count)
#-------------------------------------------------------------------------------
#                         Box  Plots  (Box and wisker plot)
"""
x: Categorical (finite possibilities)
y: numeric     (infinite possibilities)
data: dataset
hue: categorical seperation of x axis for further level of detail
"""
#-------------------------------------------------------------------------------

box = sns.boxplot(x = 'day', y = 'total_bill', data = tips, hue = 'smoker')

plt.show(box)

#-------------------------------------------------------------------------------
#                         Violin plots
# basically a more detailed box plot
# draw back, requires more time to understand visulization of data

# Shows Kernal Density Estimation of underlying distribution
# width shows density at that each y-value (not like every cent, but in small groups)
"""
x: Categorical (finite possibilities)
y: numeric     (infinite possibilities)
data: dataset
hue: categorical seperation of x axis for further level of detail
split: instead of two violins next to each other from hue, split each violen in half
"""
#-------------------------------------------------------------------------------


violin = sns.violinplot(x = 'day', y = 'total_bill', data = tips, hue = 'sex', split = True)
plt.show(violin)

#-------------------------------------------------------------------------------
#                         Strip plot
#       Basically verticle scatter plot for each category
"""
x: Categorical (finite possibilities)
y: numeric     (infinite possibilities)
data: dataset
hue: categorical seperation of x axis for further level of detail
jitter: random width to help visualize number of dots, avoids dots stacking on top of eachother
dodge: split is renamed as dodge in newer version of seaborn, split will still work
"""
#-------------------------------------------------------------------------------

strip = sns.stripplot(x = 'day', y = 'total_bill', data = tips, jitter = True, hue = 'sex', dodge = True)

plt.show(strip)

#-------------------------------------------------------------------------------
#                         Swarm plot
#       Basically a strip/scatter plot and violin plot combined
#       Sometimes won't scale well with large ammounts of data
#       don't use for large data sets, don't show alone
"""
x: Categorical (finite possibilities)
y: numeric     (infinite possibilities)
data: dataset
hue: categorical seperation of x axis for further level of detail
jitter: random width to help visualize number of dots, avoids dots stacking on top of eachother
dodge: split is renamed as dodge in newer version of seaborn, split will still work
"""
#-------------------------------------------------------------------------------

swarm = sns.swarmplot(x = 'day', y = 'total_bill', data = tips)
plt.show(swarm)

#-------------------------------------------------------------------------------
#                         Swarm plot on top of Violin plot
#-------------------------------------------------------------------------------

sns.violinplot(x = 'day', y = 'total_bill', data = tips)
sns.swarmplot(x = 'day', y = 'total_bill', data = tips, color = 'black')
plt.show()

#-------------------------------------------------------------------------------
#                         Factor plot
#                 can plot any categorical plot,
#           recommend calling plot required plot itself over this one
"""
kind: bar, violin, swarm, etc.
"""
#-------------------------------------------------------------------------------

factor = sns.factorplot(x = 'day', y = 'total_bill', data = tips, kind = 'bar')
