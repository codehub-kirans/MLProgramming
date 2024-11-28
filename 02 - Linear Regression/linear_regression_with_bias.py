"""Prediction using linear regression"""
import numpy as np


# prediction
def predict(X, w, b):
    return X * w + b  # model function


# loss function
def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)  # mean squared error


# training phase
def train(
    X, Y, iterations, learning_rate
):  # hyperparameters  are iterations and learning_rate
    w = b = 0  # parameters of the model function

    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %d: Loss => %.2f" % (i, current_loss))

        if loss(X, Y, w + learning_rate, b) < current_loss:
            w += learning_rate
        elif loss(X, Y, w - learning_rate, b) < current_loss:
            w -= learning_rate
        elif loss(X, Y, w, b + learning_rate) < current_loss:
            b += learning_rate
        elif loss(X, Y, w, b - learning_rate) < current_loss:
            b -= learning_rate
        else:
            print("Found Model Parameters: w=>%.2f, b=>%.2f" % (w, b))
            return w, b

    raise Exception("Couldn't converge within %d iteration" % (iterations))


X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w, b = train(X, Y, iterations=100000, learning_rate=0.01)
print("Prediction: x= %d y=> %.2f" % (20, predict(20, w, b)))
# w, b = train(X, Y, iterations=10000000, learning_rate=0.00001)
# print("Prediction: x= %d y=> %.2f" % (20, predict(20, w, b)))


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
plt.plot([0, x_edge], [b, predict(x_edge, w, b)], linewidth=1.0, color="g")
plt.show()
