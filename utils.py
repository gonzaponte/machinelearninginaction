from __future__ import print_function, division

import time
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as _colors

# colors
#1f77b4
#ff7f0e
#2ca02c
#d62728
#9467bd
#8c564b
#e377c2
#7f7f7f
#bcbd22
#17becf

def _get_mpl_colors(N=10):
    colors = []
    for i in range(N):
        a = plt.scatter(np.arange(2),
                        np.arange(2))
        colors += [_colors.to_hex(a.get_facecolor()[0])]
    return colors

colors = _get_mpl_colors()

def timethis(f):
    def timed_f(*args, **kwargs):
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        print("Time spent in {}: {:.5f} s".format(f.__name__,
                                                  t1 - t0   ))
        return result
    return timed_f


def listdir(path):
    return [os.path.join(os.path.abspath(path),
                         filename)
            for filename in os.listdir(path)]


def plot(data, labels, xcoord=0, ycoord=1, xlabel="", ylabel="", **kwargs):
    for i, label in enumerate(sorted(set(labels))):
        indices = labels == label
        plt.scatter(data[indices, xcoord],
                    data[indices, ycoord],
                    label     = label,
                    facecolor = colors[i],
                    **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    
def frequencies(items):
    return {item: np.count_nonzero(items == item)/items.size
            for item in set(items)}


def lmap(*args, **kwargs):
    return list(map(*args, **kwargs))


def nplist(*args, **kwargs):
    return np.array(list(*args, **kwargs))


def npmap(*args, **kwargs):
    return np.array(lmap(*args, **kwargs))


def load_data(filename):
    data = np.loadtxt(filename)
    return data[:, :-1], data[:, -1]


def confussion_matrix(predicted, truth):
    lbls = nplist(set(truth))
    cmat = np.zeros((len(lbls), len(lbls)), dtype=float)
    for i, lbl_i in enumerate(lbls):
        where     = truth == lbl_i
        true_lbls = predicted[where]
        for j, lbl_j in enumerate(lbls):
            cmat[i, j] = np.count_nonzero(true_lbls == lbl_j)/true_lbls.size
    return cmat, lbls


def precision_recall(cmat, p=0):
    n = int(not p)
    precision = 1/(1+cmat[n,p]/cmat[p,p]) if cmat[p,p] else 0
    recall    = 1/(1+cmat[p,n]/cmat[p,p]) if cmat[p,p] else 0
    return precision, recall


