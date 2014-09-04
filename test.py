import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


from rbfn import RBFN, NormalizedRBFN, HyperplaneRBFN, GaussianRBF

## 1d to 1d
indim, bases, outdim, alpha = 1, 5, 1, 0.1

mu = np.linspace(0, 3.5, num=bases, endpoint=True).reshape((bases, indim))
sigma = np.ones((bases**indim, indim)) * 0.4
neurons = GaussianRBF(mu, sigma)
rbfn1 = RBFN(neurons, indim, bases, outdim, alpha)
rbfn2 = NormalizedRBFN(neurons, indim, bases, outdim, alpha)
rbfn3 = HyperplaneRBFN(neurons, indim, bases, outdim, alpha)

networks = [rbfn1, rbfn2, rbfn3]

f = lambda _x: 2*np.sin(_x) + np.cos(4*_x) + np.sqrt(_x)
x_train = 3.5 * np.random.random(1000)
y_train = f(x_train)

x_test = np.arange(0, 3.5, 0.01)
y_test = f(x_test)

for network in networks:
    network.train(x_train, y_train)

    p = np.zeros(y_test.shape)
    for i, x in enumerate(x_test):
        p[i] = network.evaluate(x)

    print '1d to 1d', r2_score(y_test, p)
    plt.figure()
    plt.plot(x_test, y_test)
    plt.plot(x_test, p)
    plt.title(network.__class__.__name__)

plt.show()