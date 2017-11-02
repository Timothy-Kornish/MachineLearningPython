import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 5, 11)
y = x ** 2

plt.figure(1)
plt.plot(x, y, 'r-')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.title("Exponents!!!")
plt.close(1) # makes plots non-visible

plt.figure(2)
plt.subplot(1, 2, 1)
plt.title("Subplot 1")
plt.plot(x, y, 'r-')
plt.subplot(1, 2, 2)
plt.title("Subplot 2")
plt.plot(y, x, 'b--') #'b*', b--, b+, b-
plt.close(2)

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(x, y)
axes.set_xlabel("X Label")
axes.set_ylabel('Y Label')
axes.set_title('Set Title')

plt.close(fig)

fig2 = plt.figure()
axes2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8]) # all values must be between 0 and 1
#point of focus in axes is at (0, 0) I will call the origin
#index 1: graph origin moves 10% right from left edge of window pane
#index 2: graph origin moves 10% up from bottom edge of window pane
#index 3: graph takes 80% width of window pane
#index 4: graph takes 80% height of window pane
axes3 = fig2.add_axes([0.2, 0.5, 0.4, 0.3])
#index 1: graph origin moves 20% right from left edge of window pane
#index 2: graph origin moves 50% up from bottom edge of window pane
#index 3: graph takes 40% width of window pane
#index 4: graph takes 30% height of window pane

#axes4 = fig2.add_axes([0.5, 0.1, 0.4, 0.3]) # in bottom right corner
axes2.plot(x, y, 'r')
axes3.plot(y, x, 'g')
axes2.set_title('Large Plot')
axes3.set_title('Small Plot')
plt.show(fig2)
