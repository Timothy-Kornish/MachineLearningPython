import matplotlib.pyplot as plt
import numpy as np

"""
Learning about ways to apply color, line width and other customization options
graph types: line plots, scatter plots, histograms, box-plots, etc.
"""

x = np.linspace(0, 5, 11)
y = x ** 2

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

#color Options:
#           basic full strings:         green, blue, red, orange, purple, etc
#           Single letters:             r, g, b, etc
#           RGB Hexcode:                #FF8C00

#Line Width and Style Options:
#           linewidth: (lw works too)   default = 1, any number includeing
#                                       floating point numberas allowed, no negatives
#           alpha:                      controls opacity
#           linestyle: (ls works too)   -, --, -., +, *, :, steps, o

#Markers
#           marker:                     +, *, o, 1,
#           markersize:                 default = 1
#           markerfacecolor:            defaults to color of linew
#           markeredgewidth:            default 1
#           markeredgecolor:            default to color of line

ax.plot(x, y, color = '#FF8C00', linewidth = 2, alpha = 1, linestyle = '--',
        marker = 'o', markersize = 15, markerfacecolor = 'yellow', markeredgewidth = 3, markeredgecolor = 'green')

ax.plot(x, y, color = '#248765', linewidth = 20, alpha = 0.1)
plt.close(fig)

fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(x, y, color = 'purple', lw = 2, ls = '--')
ax2.set_xlim([0, 1]) # [lower bound, upper bound]
ax2.set_ylim([0, 1])
plt.show()
