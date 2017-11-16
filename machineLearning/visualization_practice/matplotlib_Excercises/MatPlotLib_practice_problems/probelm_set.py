import matplotlib.pyplot as plt
import numpy as np


x = np.arange(0, 100)
y = x * 2
z = x ** 2

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(x, y)
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.set_title('title')
plt.close(fig)

fig2 = plt.figure()
axes = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = fig2.add_axes([0.2, 0.5, 0.2, 0.2])
axes.plot(x, y)
axes2.plot(x, y)
plt.close(fig2)

fig3 = plt.figure()
axes3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
axes4 = fig3.add_axes([0.2, 0.4, 0.3, 0.3])
axes3.plot(x, z)
axes4.plot(x, y)
axes3.set_xlabel('x')
axes4.set_xlabel('x')
axes3.set_ylabel('Z')
axes4.set_ylabel('y')
axes4.set_title('zoom')
axes4.set_xlim([20.0, 22.0])
axes4.set_ylim([30.0, 50.0])
plt.close(fig3)

newFig, newAx = plt.subplots(nrows = 1, ncols = 2)
newAx[0].plot(x, y, 'b--')
newAx[1].plot(x, z, 'r-')
plt.show()
