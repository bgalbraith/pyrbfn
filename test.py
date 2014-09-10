import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


from rbfn import RBFN, NormalizedRBFN, HyperplaneRBFN, GaussianRBF, \
    AdaptiveRBFN, AdaptiveHyperplaneRBFN

###############################################################################
## 1d to 1d
###############################################################################
indim, bases, outdim, alpha = 1, 5, 1, 0.5

mu = np.linspace(0, 3.5, num=bases, endpoint=True).reshape((bases, indim))
sigma = np.ones((bases**indim, indim)) * 0.4
neurons = GaussianRBF(mu, sigma)
rbfn1 = RBFN(neurons, indim, bases, outdim, alpha)
rbfn2 = NormalizedRBFN(neurons, indim, bases, outdim, alpha)
rbfn3 = HyperplaneRBFN(neurons, indim, bases, outdim, alpha)

bases = 3
mu = np.linspace(0, 3.5, num=bases, endpoint=True).reshape((bases, indim))
sigma = np.ones((bases**indim, indim)) * 0.4
neurons = GaussianRBF(mu, sigma)
rbfn4 = AdaptiveRBFN(neurons, indim, bases, outdim, alpha, 0.01)
rbfn5 = AdaptiveHyperplaneRBFN(neurons, indim, bases, outdim, alpha, 0.01)
networks = [rbfn1, rbfn2, rbfn3, rbfn4, rbfn5]

f = lambda _x: 2*np.sin(_x) + np.cos(4*_x) + np.sqrt(_x)
x_train = 3.5 * np.random.random(2000)
y_train = f(x_train)

x_test = np.arange(0, 3.5, 0.01)
y_test = f(x_test)

print '--- R1 -> R1 ---'
plt.figure()
plt.plot(x_test, y_test, 'k', linewidth=3)
for network in networks:
    network.train(x_train, y_train)

    p = np.zeros(y_test.shape)
    for i, x in enumerate(x_test):
        p[i] = network.evaluate(x)

    print '%.3f %s' % (r2_score(y_test, p), network.__class__.__name__)
    plt.plot(x_test, p)


###############################################################################
## 2d to 1d
###############################################################################
indim, bases, outdim, alpha = 2, 7, 1, 0.5
a = np.linspace(0, 3.5, num=bases, endpoint=True)
mu = np.array(np.meshgrid(*((a,)*indim))).reshape(indim, bases**indim).T
sigma = np.ones((bases**indim, indim)) * 0.2
neurons = GaussianRBF(mu, sigma)
rbfn1 = RBFN(neurons, indim, bases, outdim, alpha)
rbfn2 = NormalizedRBFN(neurons, indim, bases, outdim, alpha)
rbfn3 = HyperplaneRBFN(neurons, indim, bases, outdim, alpha)

bases = 6
a = np.linspace(0, 3.5, num=bases, endpoint=True)
mu = np.array(np.meshgrid(*((a,)*indim))).reshape(indim, bases**indim).T
sigma = np.ones((bases**indim, indim)) * 0.2
neurons = GaussianRBF(mu, sigma)
rbfn4 = AdaptiveRBFN(neurons, indim, bases, outdim, alpha, 0.001)
rbfn5 = AdaptiveHyperplaneRBFN(neurons, indim, bases, outdim, alpha, 0.001)
networks = [rbfn1, rbfn2, rbfn3, rbfn4, rbfn5]

f2 = lambda _x, _y: 3*np.cos((_x+2)*(_y+2)) - np.sin(6*_y) + _x
x_train = 3.5 * np.random.random((5000, 2))
y_train = f2(x_train[:, 0], x_train[:, 1])

x_test = np.mgrid[0:3.5:0.05, 0:3.5:0.05]
x_test = x_test.reshape((indim, x_test.size / indim)).T
y_test = f2(x_test[:, 0], x_test[:, 1])
idx = np.where(x_test[:, 1] == 2.0)[0]

print '--- R2 -> R1 ---'
plt.figure()
plt.plot(x_test[idx, 0], y_test[idx], 'k', linewidth=3)
for network in networks:
    network.train(x_train, y_train)

    p = np.zeros(y_test.shape)
    for i, x in enumerate(x_test):
        p[i] = network.evaluate(x)

    print '%0.3f %s' % (r2_score(y_test, p), network.__class__.__name__)
    plt.plot(x_test[idx, 0], p[idx])

plt.show()