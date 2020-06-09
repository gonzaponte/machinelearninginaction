"""
Classify using the k-Nearest neighbors algorithm.
"""
from __future__ import division

import os
import numpy as np
import operator as op

from utils import timethis

def create_data_set():
    group  = [[1.0, 1.1],
              [1.0, 1.0],
              [0.0, 0.0],
              [0.0, 0.1]]
    labels = ["A", "A", "B", "B"]
    return np.array(group), np.array(labels)


def load_dating_data_set():
    filename = "./datingTestSet.txt"
    a = np.loadtxt(filename,
                   dtype=np.float32,
                   usecols = (0, 1, 2))
    b = np.loadtxt(filename,
                   dtype=str,
                   usecols = (3,))
    return a, b


def load_handwriting_data_set(folder = "./trainingDigits/"):
    data    = []
    labels  = []
    for filename in os.listdir(folder):
        number = int(filename.split("_")[0])
        vector = []
        for line in open(folder + filename):
            vector.extend(map(int, list(line.rstrip())))
        data.append(vector)
        labels.append(number)
    return np.array(data), np.array(labels)


def normalize_dataset(dataset):
    means  = np.apply_along_axis(np.mean, 0, dataset)
    stdevs = np.apply_along_axis(np.std , 0, dataset)
    return (dataset-means) / stdevs

"""
def normalize_dataset(dataset):
    mins = np.min(dataset, axis=0)
    maxs = np.max(dataset, axis=0)
    return (dataset-mins) / (maxs-mins)
"""

def classify_evt(evt, dataset, labels, k, norm=False):
    """
    Classify evt in some label based on the
    properties of (dataset, labels) using k neighbors.
    """
    if norm:
        dataset = normalize_dataset(dataset)
    d = np.apply_along_axis(np.sum, 1, (evt - dataset)**2)
    closest_points = np.argsort(d)[:k]
    closest_labels = labels[closest_points].tolist()
    classification = {lbl: closest_labels.count(lbl) for lbl in set(labels)}
    return max(classification.items(),
               key = op.itemgetter(1))[0]


#@timethis
def classify_data(evts, dataset, labels, k):
    classify = lambda x: classify_evt(x, dataset, labels, k)
    return np.apply_along_axis(classify, 1, evts)


@timethis
def test_dating(k, test_ratio = 0.1, debug=False):
    props, labels = load_dating_data_set()
    props = normalize_dataset(props)
    ntest = int(test_ratio * props.shape[0])
    classifier_result = classify_data(props [:ntest],
                                      props [ntest:],
                                      labels[ntest:],
                                      k)
    matches = np.count_nonzero(classifier_result == labels[:ntest])
    if debug:
        for result, real in zip(classifier_result, labels[:ntest]):
            print("Result is {0} when real is {1}".format(result, real))
    return (1 - matches/ntest) * 100


@timethis
def test_handwriting(k, debug=False):
    train_props, train_labels = load_handwriting_data_set("trainingDigits/")
    test_props , test_labels  = load_handwriting_data_set("testDigits/")
    classifier_result = classify_data(  test_props,
                                       train_props,
                                      train_labels,
                                      k)
    matches = np.count_nonzero(classifier_result == test_labels)
    if debug:
        for result, real in zip(classifier_result, test_labels):
            print("Result is {0} when real is {1}".format(result, real))
    return (1 - matches/test_labels.size) * 100

