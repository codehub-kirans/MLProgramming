import numpy as np


def predict(X, w):
    return np.matmul(X, w)


def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)


def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y))/X.shape[0]


def train(X, Y, iterations, learning_rate):
    w = np.zeros((X.shape[1], 1))

    for i in range(iterations):
        print("Iteration %d: Loss=> %.10f" % (i, loss(X, Y, w)))
        w_gradient = gradient(X, Y, w)
        w -= w_gradient * learning_rate

    return w


x1, x2, x3, y = np.loadtxt("pizza_3_vars.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.shape[0]), x1, x2, x3))
Y = y.reshape(-1, 1)

w = train(X, Y, iterations=100000, learning_rate=0.001)

print("\n A few predictions")
for i in range(5):
    print("X[%d] -> %.4f (label: %d)" % (i, predict(X[i], w), Y[i]))
