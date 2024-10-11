import numpy as np
import scipy


def Laplace(x, theta):
    assert theta[1] > 0
    return scipy.stats.laplace.pdf(x, theta[0], theta[1])
    # return np.exp(-np.abs(x - theta[0]) / theta[1]) / (2 * theta[1])


def Beta(x, theta):
    assert 0 < x < 1
    return scipy.stats.beta.pdf(x, theta[0], theta[1])


f = Laplace
X = np.array([-3.244, 0.3532, 0.3016, 0.3512, 0.8663, 0.1368, 0.9057, 0.3646, -1.183, -0.2269, 0.7517, 0.3673])
Theta_0 = (0.4, 0.6)
Theta_1 = (0.5, 0.7)
alpha = 0.03
beta = 0.04

# f = Beta
# X = np.array([0.1135, 0.3147, 0.1506, 0.1194, 0.1839, 0.5928, 0.405, 0.9079, 0.01298, 0.5294, 0.406, 0.05982, 0.3305, 0.01509,
#      0.2026, 0.6879, 0.01231, 0.3831])
# Theta_0 = (0.4, 2.6)
# Theta_1 = (0.6, 1.4)
# alpha = 0.06
# beta = 0.04


if __name__ == '__main__':
    GH_0 = lambda x: x <= np.log(beta / (1 - alpha))
    GH_1 = lambda x: x >= np.log((1 - beta) / alpha)
    print(np.log(beta / (1 - alpha)), " ", np.log((1 - beta) / alpha))
    gamma = 0
    for i in range(len(X)):
        gamma += np.log(f(X[i], Theta_1) / f(X[i], Theta_0))
        print(i + 1, "\t%.14f" % gamma, end="\t")
        if GH_0(gamma):
            print("H_0")
            break
        if GH_1(gamma):
            print("H_1")
            break
        print("G*")
