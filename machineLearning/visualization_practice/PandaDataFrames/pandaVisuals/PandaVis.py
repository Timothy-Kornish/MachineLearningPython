import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#-------------------------------------------------------------------------------
#                    Pandas Data Visualization
"""
importing seaborn will make plots look more like seaborn plot even though
seaborn is not used directly

pandas calls will then look like seaborn
"""
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#   Load in Data Frames
#-------------------------------------------------------------------------------

df1 = pd.read_csv('df1.csv', index_col = 0)
df2 = pd.read_csv('df2.csv')
df3 = pd.read_csv('df3.csv')

#-------------------------------------------------------------------------------
#   Multiple ways of ploting a histogram with pandas
#-------------------------------------------------------------------------------

hist1 = df1['A'].hist(bins = 30)
plt.show(hist1)

hist2 = df1['A'].plot(kind = 'hist', bins = 20)
plt.show(hist2)

hist3 = df1['A'].plot.hist()
plt.show(hist3)

#-------------------------------------------------------------------------------
#   Area plot
#-------------------------------------------------------------------------------

area = df2.plot.area(alpha = 0.4)
plt.show(area)

#-------------------------------------------------------------------------------
#   bar plot
"""
takes index as a category for x axis
"""
#-------------------------------------------------------------------------------

bar = df2.plot.bar(stacked = True)
plt.show(bar)

#-------------------------------------------------------------------------------
#   line plot
#-------------------------------------------------------------------------------

line = df1.plot.line(x = df1.index, y = 'B', figsize = (12, 3), lw = 2)
plt.show(line)

#-------------------------------------------------------------------------------
#   scatter plot
#-------------------------------------------------------------------------------

scatter = df1.plot.scatter(x = 'A', y = 'B', c = 'C', cmap = 'magma', s = (df1['C']* 50))
plt.show(scatter)

#-------------------------------------------------------------------------------
#   box plot
#-------------------------------------------------------------------------------

box = df2.plot.box()
plt.show(box)

#-------------------------------------------------------------------------------
#   Hex bin  plot
#-------------------------------------------------------------------------------

df = pd.DataFrame(np.random.randn(1000, 2), columns = ['A', 'B'])

hexbin = df.plot.hexbin(x = 'A', y = 'B', gridsize = 25)
plt.show(hexbin)

#-------------------------------------------------------------------------------
#   KDE  plot
#-------------------------------------------------------------------------------

kde = df2.plot.kde()
plt.show(kde)
