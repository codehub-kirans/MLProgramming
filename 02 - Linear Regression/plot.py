"""Matplot Chart Plotting"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.axis([0, 50, 0, 50])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Pizzas", fontsize=30)
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
plt.plot(X, Y, "bo")
plt.show()

# activate Seaborn
# scale axes (0 to 50) # set x axis ticks
# set y axis ticks
# set x axis label
# set y axis label
# load data
# plot data
# display chart
