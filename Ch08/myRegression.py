import numpy as np

from kernels import RBFKernel

"""
class Predictor:
    def __init__(self, w, kernel=None, *extras):
        self.weights = w
        self.kernel  = kernel
        self.extras  = extras

    def __call__(self, x):
        return self.kernel(self.extras[0], x).dot( if self.kernel else x.dot(self.w)
"""

def std_lin_regression(x, y):
    x = np.atleast_2d(x)
    w = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
#    return Predictor(w)
    predictor   = lambda x: x.dot(predictor.w)
    predictor.w = w
    return predictor


def gaussian_kernel(sigma=1):
    def kernel(y, x):
        n = x.shape[0]
        K = np.eye(n)
        d = y - x
        print("d", d.shape)
        print("d", d.dot(d.T).shape)
        K[np.arange(n), np.arange(n)] = np.exp(-0.5*d.dot(d.T)/sigma**2)
        return K
    return kernel


def lw_lin_regression(x, y, kernel):
    x = np.atleast_2d(x)
    def predictor(y):
        lw = kernel(y, x)
        print("w", lw.shape)
        print("x", y.shape)
        print("y", x.shape)
        return np.linalg.inv(x.T.dot(lw.dot(x))).dot(x.T.dot(lw.dot(y)))
    return predictor
