import abc
import os
import numpy as np
import functools
import operator
import matplotlib.pyplot as plt

from collections import namedtuple
from utils       import timethis, colors, npmap, plot
from kernels     import LinearKernel, RBFKernel


def load_data_set(filename="testSet.txt"):
    data = np.loadtxt(filename)
    return data[:, :-1], data[:, -1]


def load_handwriting_data_set(folder = "./trainingDigits/"):
    data    = []
    labels  = []
    for filename in os.listdir(folder):
        number = -1 if int(filename.split("_")[0]) == 9 else 1
        vector = []
        for line in open(folder + filename):
            vector.extend(map(int, list(line.rstrip())))
        data.append(vector)
        labels.append(number)
    return np.array(data), np.array(labels)


def compute_error(dataset, labels, alphas, b, k):
    t1 = alphas * labels
    t2 = dataset.dot(dataset[k])
    return t1.dot(t2) + b - labels[k]


@timethis
def platts_simple_smo(dataset, labels, C, tol, max_iter):
    alphas  = np.zeros_like(labels, dtype=np.float)
    b       = 0
    error   = functools.partial(compute_error, dataset, labels)
    indices = np.arange(len(labels))
    
    it = 0
    while it < max_iter:
        has_any_alphas_pair_changed = False
        for i in indices:
            row_i   = dataset[i]
            lbl_i   = labels [i]
            alpha_i = alphas [i]
            error_i = error(alphas, b, i)
            if ((lbl_i*error_i < -tol and alpha_i < C) or
                (lbl_i*error_i >  tol and alpha_i > 0)):
                j = np.random.choice(np.delete(indices, i))
                
                row_j   = dataset[j]
                lbl_j   = labels [j]
                alpha_j = alphas [j]
                error_j = error(alphas, b, j)
                
                low  = max(0, alpha_j + alpha_i - C if lbl_i == lbl_j else alpha_j - alpha_i)
                high = min(C, alpha_j - alpha_i + C if lbl_i != lbl_j else alpha_j + alpha_i)
                if low == high:
                    continue
                
                eta = -np.sum((row_i - row_j)**2)
                if eta >= 0:
                    continue
                
                alphas[j] = np.clip(alpha_j - lbl_j * (error_i - error_j)/eta, low, high)
                d_alpha_j = alphas[j] - alpha_j
                if abs(d_alpha_j) < 1e-5:
                    continue
                
                alphas[i] -= lbl_i * lbl_j * d_alpha_j
                d_alpha_i  = alphas[i] - alpha_i
                
                b1  = b - error_i
                b1 -= lbl_i * d_alpha_i * np.sum(row_i**2)
                b1 -= lbl_j * d_alpha_j * np.sum(row_i*row_j)
                
                b2  = b - error_j
                b2 -= lbl_j * d_alpha_j * np.sum(row_j**2)
                b2 -= lbl_i * d_alpha_i * np.sum(row_i*row_j)
                
                if   0 < alphas[i] < C: b = b1
                elif 0 < alphas[j] < C: b = b2
                else:                   b = 0.5 * (b1 + b2)

                has_any_alphas_pair_changed = True
        it = 0 if has_any_alphas_pair_changed else it + 1
    return alphas, b


def plot_support_vectors(data, labels, radius = 1.):
    for lbl, color in zip(sorted(set(labels)), colors):
        for pos in data[labels==lbl]:
            plt.gca().add_artist(plt.Circle(pos, radius, color=color, fill=False))


class PlattsSMO:
    def __init__(self, dataset, labels, C=0.6, tol=1e-3, max_iter=40, kernel=None, dtype=np.float64):
        self.ds       = np.array(dataset, dtype=dtype)
        self.lbls     = np.array(labels , dtype=dtype)
        self.C        = dtype(C)
        self.tol      = dtype(tol)
        self.max_it   = int(max_iter)
        
        self.alphas   = np.zeros_like(labels, dtype=dtype)
        self.b        = dtype(0.0)
        
        self.indices  = np.arange(labels.size)
        self.errors   = np.zeros_like(self.alphas, dtype=dtype)
        self.is_valid = np.zeros_like(self.errors, dtype=bool)

        self.kernel = kernel
        if kernel:
            self.ks = self.kernel(self.ds)

        self.train()
        self.compute_weights()
        
    def compute_error(self, k):
        if self.kernel:
            t1 = self.alphas * self.lbls
            t2 = self.ks[:, k]
            return t1.dot(t2) + self.b - self.lbls[k]
        else:
            return compute_error(self.ds, self.lbls, self.alphas, self.b, k)

    def select_alpha(self, i):
        if np.count_nonzero(self.is_valid) <= 1:
            index = 5#np.random.choice(np.delete(self.indices, i))
            return index, self.compute_error(index)

        indices = self.indices[self.is_valid]
        indices = indices[indices != i]
        errors  = npmap(self.compute_error, indices)
        largest = np.argmax(np.abs(errors - self.errors[i]))

        return indices[largest], errors[largest]
    
    def update_error(self, i, error_i):
        self.is_valid[i] = True
        self.errors  [i] = error_i

    def valid_alphas(self):
        return np.nonzero((self.alphas > 0) & (self.alphas < self.C))[0]
    
    def try_to_optimize(self, i):
        row_i   = self.ds    [i]
        lbl_i   = self.lbls  [i]
        alpha_i = self.alphas[i]
        error_i = self.compute_error(i)
#        print("Ei", error_i)
        if ((lbl_i*error_i < -self.tol and alpha_i < self.C) or
            (lbl_i*error_i >  self.tol and alpha_i > 0)):
#            print("ERRORS", self.errors)
            self.update_error(i, error_i)
            
            j, error_j = self.select_alpha(i)
#            print("SELECTED ALPHA", j, error_j)

            row_j   = self.ds    [j]
            lbl_j   = self.lbls  [j]
            alpha_j = self.alphas[j]

            low  = max(     0, alpha_j + alpha_i - self.C if lbl_i == lbl_j else alpha_j - alpha_i)
            high = min(self.C, alpha_j - alpha_i + self.C if lbl_i != lbl_j else alpha_j + alpha_i)
            if low == high:
                return 0

            eta = 2 * self.ks[i,j] - self.ks[i,i] - self.ks[j,j] if self.kernel else -np.sum((row_i - row_j)**2)
            if eta >= 0:
                return 0

            self.alphas[j] = np.clip(alpha_j - lbl_j * (error_i - error_j)/eta, low, high)
            self.update_error(j, self.compute_error(j))
            
            d_alpha_j = self.alphas[j] - alpha_j
            if abs(d_alpha_j) < 1e-5:
                return 0

            self.alphas[i] -= lbl_i * lbl_j * d_alpha_j
            self.update_error(i, self.compute_error(i))
            
            d_alpha_i = self.alphas[i] - alpha_i

            b1  = self.b - error_i
            b1 -= lbl_i * d_alpha_i * (self.ks[i,i] if self.kernel else np.sum(row_i**2))
            b1 -= lbl_j * d_alpha_j * (self.ks[i,j] if self.kernel else np.sum(row_i*row_j))

            b2  = self.b - error_j
            b2 -= lbl_j * d_alpha_j * (self.ks[j,j] if self.kernel else np.sum(row_j**2))
            b2 -= lbl_i * d_alpha_i * (self.ks[i,j] if self.kernel else np.sum(row_i*row_j))

            if   0 < self.alphas[i] < self.C: self.b = b1
            elif 0 < self.alphas[j] < self.C: self.b = b2
            else                            : self.b = 0.5 * (b1 + b2)
            return 1
        return 0
    
    @timethis
    def train(self):
        full_loop   = True
        
        it = 0
        n_optimized=0
        while it < self.max_it and (full_loop or n_optimized):
#            print("TRAIN", full_loop, it, n_optimized)
            n_optimized = 0

            indices = self.indices if full_loop else self.valid_alphas()
            for i in indices:
                n_optimized += self.try_to_optimize(i)
#                print("N_OPTIMIZED", n_optimized)
            it += 1
            
            if       full_loop  : full_loop = False
            elif not n_optimized: full_loop = True

    def compute_weights(self):
        print("Number of support vectors:", len(self.valid_alphas()))
        if self.kernel:
            valid        = self.valid_alphas()
            self.SVs     = self.ds  [valid]
            self.weights = self.lbls[valid] * self.alphas[valid]
        else:
            self.weights = np.zeros_like(self.ds[0])
            for i in self.valid_alphas():
                self.weights += self.alphas[i] * self.lbls[i] * self.ds[i]

    def classify(self, data):
#        print(self.weights.shape, self.SVs.shape, data.shape, self.kernel.transform(self.SVs, data).shape)
        return self.weights.dot(self.kernel.transform(self.SVs, data) if self.kernel else data) + self.b

    def classify_all(self, data):
        return npmap(self.classify, data)

    def plot_support_vectors(self, plot_data=True):
        if plot_data:
            plot(self.ds, self.lbls, 0, 1)
        r = 0.1*np.sum(np.std(self.ds, axis=0)**2)**0.5
        plot_support_vectors(self.ds[self.alphas>0], self.lbls[self.alphas>0], r)


def test_rbf_kernel(sigma=1.8, print_result=False):
    dataset, labels = load_data_set("testSetRBF.txt")
    smo             = PlattsSMO(dataset, labels, 200, 1e-4, 1e4, RBFkernel(sigma=sigma))
    right_guesses   = np.count_nonzero(np.sign(smo.classify_all(dataset)) == np.sign(labels))
    train_er        = (1 - right_guesses/labels.size)*100
    if print_result:
        print("Traning error rate: {} %".format(train_er))

    dataset, labels = load_data_set("testSetRBF2.txt")
    right_guesses   = np.count_nonzero(np.sign(smo.classify_all(dataset)) == np.sign(labels))
    test_er         = (1 - right_guesses/labels.size)*100
    if print_result:
        print("Testing error rate: {} %".format(test_er))
    return train_er, test_er


def test_handwriting(sigma=14.2, print_result=False):
    dataset, labels = load_handwriting_data_set("../Ch02/trainingDigits/")
    smo             = PlattsSMO(dataset, labels, 200, 1e-4, 1e4, RBFkernel(sigma=sigma))
    right_guesses   = np.count_nonzero(np.sign(smo.classify_all(dataset)) == np.sign(labels))
    train_er        = (1 - right_guesses/labels.size)*100
    if print_result:
        print("Traning error rate: {} %".format(train_er))

    dataset, labels = load_handwriting_data_set("../Ch02/testDigits/")
    right_guesses   = np.count_nonzero(np.sign(smo.classify_all(dataset)) == np.sign(labels))
    test_er         = (1 - right_guesses/labels.size)*100
    if print_result:
        print("Testing error rate: {} %".format(test_er))
    return train_er, test_er