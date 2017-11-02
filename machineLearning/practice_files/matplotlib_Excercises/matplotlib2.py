import matplotlib.pyplot as plt
import numpy as np


"""
more object oriented plots, using subplots, playing with size, and legend in plot
"""

x = np.linspace(0, 5, 11)
y = x ** 2

"""
using columns and rows, do not plot()

fig, axes = plt.subplots(nrows = 1, ncols = 2)
# axes.plot(x, y)

or can plot this way for single large one:

fig, axes = plt.subplots()
axes.plot(x, y)
"""

fig, axes = plt.subplots(nrows = 1, ncols = 2) # subplots automatically does add_axes method
                                               # automatically chooses graph location and size
                                               # base on number of subplots

plt.tight_layout() # fixes overlapping of graphs in same window, highly recommended
#print("axes variable:\n", axes)
"""
for current_ax in axes:
    current_ax.plot(x, y)
"""


axes[0].plot(x, y)
axes[0].set_title('first plot')
axes[1].plot(y, x)
axes[1].set_title('second plot')
plt.close(fig)

#dpi = dots per inch, also known as pixels per inch
#figsize = (length, height) not full inches but close
fig2 = plt.figure(figsize = (3, 2), dpi = 100)
ax = fig2.add_axes([0, 0, 1, 1])
ax.plot(x, y)
plt.close(fig2)

fig3, axes3 = plt.subplots(figsize = (8, 2))
axes3.plot(x, y)
plt.close(fig3)

fig4, axes4 = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 2))
axes4[0].plot(x,y)
axes4[1].plot(y,x)
plt.tight_layout()
fig4.savefig('my_picture.png', dpi = 200) # saves into local directory
plt.close(fig4)

fig5 = plt.figure()
axes5 = fig5.add_axes([0.1, 0.1, 0.8, 0.8])
axes5.plot(x, x ** 2, label = 'X Squared')
axes5.plot(x, x ** 3, label = 'X Cubed')
axes5.plot(x, x ** 4, label = 'X Quadrupled')
axes5.set_xlabel('X')
axes5.set_ylabel('Y')
axes5.set_title('Title')
axes5.legend(loc = 0)
#axes5.legend(loc = 'best')
#axes5.legend(0.1, 0.1) percentage of legend origin (bottom left) away from graph origin (0, 0)
plt.show()
