"""Prediction using linear regression"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# prediction
def predict(X, w, b):
    return X * w + b  # model function


# loss function (with gradient descent, gradient calculates loss step)
def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)  # mean squared error


def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average((predict(X, w, b) - Y))
    print("Gradients: w=>%.10f, b=>%.10f" % (w_gradient, b_gradient))
    return w_gradient, b_gradient

# training phase
# hyperparameters  are iterations and learning_rate


def train(X, Y, iterations, learning_rate):
    print("Hyperparameters are iterations: %d and learning rate %d" %
          (iterations, learning_rate))
    w = b = 0  # parameters of the model function

    for i in range(iterations):
        print("Iteration %d: Achieved Loss=> %.10f" % (i, loss(X, Y, w, b)))

        w_gradient, b_gradient = gradient(X, Y, w, b)
        w -= w_gradient * learning_rate
        b -= b_gradient * learning_rate

    return w, b


X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w, b = train(X, Y, iterations=10000, learning_rate=0.001)
print("Parameters are w=>%.10f and b=>%.10f" % (w, b))
print("Prediction: x= %d y=> %.10f" % (20, predict(20, w, b)))

# w, b = train(X, Y, iterations=10000000, learning_rate=0.00001)
# print("Prediction: x= %d y=> %.2f" % (20, predict(20, w, b)))


# Plot the chart

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
