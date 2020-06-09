import abc
import functools
import numpy     as np

from utils import npmap

class Kernel:
    def __init__(self, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    @abc.abstractmethod
    def transform(self, ds, to):
        return 

    def __call__(self, ds, to=None):
        if to is None:
            to = ds
        elif len(np.shape(to)) == 1:
            to = to[np.newaxis]
        return npmap(functools.partial(self.transform, ds), to)


class LinearKernel(Kernel):
    def transform(self, ds, to):
        return ds.dot(to)


class RBFKernel(Kernel):
    def transform(self, ds, ref):
        d = ds - ref
        return np.exp(-2 * np.sum(d*d, axis=1)/self.sigma**2)


