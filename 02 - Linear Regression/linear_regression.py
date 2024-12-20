import numpy as np


def predict(X, w):
    return X * w


def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)


def train(X, Y, iterations, learning_rate):
    w = 0

    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("Iteration %d: Loss=> %.2f" % (i, current_loss))

        if loss(X, Y, w + learning_rate) < current_loss:
            w += learning_rate
        elif loss(X, Y, w - learning_rate) < current_loss:
            w -= learning_rate
        else:
            return w

    raise Exception("Couldn't converge within %d iterations" % i)


X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w = train(X, Y, iterations=10000, learning_rate=0.01)

print("Prediction: x=%d, y=>%.2f" % (20, predict(20, w)))

# Plot the chart
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.plot(X, Y, "bo")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Pizzas", fontsize=30)
x_edge, y_edge = 50, 50
plt.axis([0, x_edge, 0, y_edge])
plt.plot([0, x_edge], [0, predict(x_edge, w)], linewidth=1.0, color="g")
plt.show()
