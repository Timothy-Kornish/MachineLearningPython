import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#-------------------------------------------------------------------------------
#                       seaborn review excercises
#-------------------------------------------------------------------------------

sns.set_style('whitegrid')
titanic = sns.load_dataset('titanic')

print(titanic.head())

#-------------------------------------------------------------------------------
#                       Excercise 1
#                       joint plot: fare vs age
#-------------------------------------------------------------------------------

joint = sns.jointplot(x = 'fare', y = 'age', data = titanic)
plt.show(joint)

#-------------------------------------------------------------------------------
#                       Excercise 2
#                       dist plot: fare
#-------------------------------------------------------------------------------

dist = sns.distplot(titanic['fare'], kde = False, color = 'red', bins = 30)
plt.show(dist)


#-------------------------------------------------------------------------------
#                       Excercise 3
#                       box plot: class vs. age
#-------------------------------------------------------------------------------

box = sns.boxplot(x = 'class', y ='age', data = titanic, palette = 'rainbow')
plt.show(box)

#-------------------------------------------------------------------------------
#                       Excercise 4
#                       swarm plot: class vs. age
#-------------------------------------------------------------------------------

swarm = sns.swarmplot(x = 'class', y = 'age', data = titanic, palette = 'Set2')
plt.show(swarm)

#-------------------------------------------------------------------------------
#                       Excercise 5
#                       count plot: sex
#-------------------------------------------------------------------------------

count = sns.countplot(x = 'sex', data = titanic)
plt.show(count)

#-------------------------------------------------------------------------------
#                       Excercise 6
#                       heat map plot:   color: coolwarm
#                       correlation plot
#-------------------------------------------------------------------------------

heat = sns.heatmap(titanic.corr(), cmap = 'coolwarm')
plt.title('titanic.corr')
plt.show(heat)

#-------------------------------------------------------------------------------
#                       Excercise 7
#                       facet grid plot:   color: coolwarm
#                       histograms
#-------------------------------------------------------------------------------

facet = sns.FacetGrid(data = titanic, col = 'sex')
facet.map(plt.hist, 'age')
plt.show(facet)
