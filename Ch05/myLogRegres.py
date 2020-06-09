import numpy as np
import functools
import matplotlib.pyplot as plt
from utils import npmap

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def load_data_set():
    data = np.loadtxt("testSet.txt")
    return np.concatenate([np.ones((len(data),1)), data[:,:-1]], axis=1), data[:, -1].astype(int)


def load_horse_colic_data_set():
    def get_data(filename):
        data = np.loadtxt(filename)
        return data[:, :-1], data[:, -1]
    
    return get_data("horseColicTraining.txt") + get_data("horseColicTest.txt")


def gradient_ascent(dataset, labels, step=1e-3, max_iter=1000):
    labels  = labels[:, np.newaxis]
    weights = np.ones((dataset.shape[1], 1))
    for i in range(max_iter):
        weighted = dataset.dot(weights)
        errors   = labels - sigmoid(weighted)
        weights += step * dataset.T.dot(errors)
    return weights


def stochastic_gradient_ascent(dataset, labels, step=1e-2):
    weights = np.ones(dataset.shape[1])
    for row, label in zip(dataset, labels):
        weights += step * row * (label - sigmoid(np.sum(row * weights)))
    return weights


def stochastic_gradient_ascent_upgrade(dataset, labels, n_iter=150, min_step=1e-2):
    weights = np.ones(dataset.shape[1])
    indices = np.arange(dataset.shape[0])
    for i in range(n_iter):
        np.random.shuffle(indices)
        for j, index in enumerate(indices):
            step     = 4/(1+i+j) + min_step
            row      = dataset[index]
            weights += step * row * (labels[index] - sigmoid(np.sum(row * weights)))
    return weights


def plot_best_fit(weights, xmin, xmax, npoints=1000):
    x = np.linspace(xmin, xmax, npoints)
    # Sigmoid function = sum(wT · x) = 0
    y = -(weights[0] + weights[1]*x) / weights[2]
    plt.plot(x, y, "k")


def classify(weights, data):
    return round(sigmoid(np.sum(weights * data)))


def test_horse_colic(n_iter=250):
    train_dataset, train_labels, test_dataset, test_labels = load_horse_colic_data_set()
    
    weights = stochastic_gradient_ascent_upgrade(train_dataset, train_labels, n_iter)
    classifier_output = npmap(functools.partial(classify, weights), test_dataset)
    error_rate = (1 - np.count_nonzero(classifier_output == test_labels)/test_labels.size)*100
    return error_rate