import functools
import collections
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import npmap, frequencies


Tree = collections.namedtuple("Tree", "feature index branches")

_decision_node = dict(boxstyle="sawtooth", fc="0.8")
_leaf_node     = dict(boxstyle="round4", fc="0.8")
_arrow_args    = dict(arrowstyle="<-")

def create_data_set():
    data_set   = [[1, 1],
                  [1, 1],
                  [1, 0],
                  [0, 1],
                  [0, 1]]
    labels     = ["yes"] * 2 + ["no"] * 3
    features   = ['no surfacing', 'flippers']
    return np.array(data_set), np.array(labels, dtype=str), np.array(features, dtype=str)


def load_lenses_data_set():
    data     = np.loadtxt("lenses.txt", dtype=str, delimiter="\t")
    labels   = data[:,-1]
    features = ["age", "prescript", "astigmatic", "tear rate"]
    return data[:, :-1], labels, np.array(features)


def compute_entropy(labels):
    probs   = np.array(list(frequencies(labels).values()))
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def split_dataset(dataset, labels, feature, value):
    rows = dataset[:, feature] == value
    return np.delete(dataset[rows], feature, axis=1), labels[rows]


def find_best_feature(dataset, labels):
    best_feature  = None
    best_gain     = -np.inf
    base_entropy  = compute_entropy(labels)
    split_feature = functools.partial(split_dataset, dataset, labels)
    for feature, values in enumerate(dataset.T):
        split     = functools.partial(split_feature, feature)
        sublabels = list(zip(*map(split, set(values))))[1]
        probs     = npmap(len, sublabels) / len(labels)
        entrs     = npmap(compute_entropy, sublabels)
        entropy   = np.sum(probs * entrs)
        gain      = base_entropy - entropy
        
        if gain > best_gain:
            best_feature = feature
            best_gain    = gain
    return best_feature


"""
def split_dataset(feature, dataset, labels):
    feature_values = dataset[:, feature]
    for value in set(feature_values):
        selection = feature_values == value
        yield value, dataset[:, selection], labels[selection]
""" 

def create_branch(dataset, labels, features):
    if len(set(labels)) == 1:
        return labels[0]
    if not dataset.size:
        probs = frequencies(labels)
        return max(probs.items(), key=op.itemgetter(1))[0]

    ifeature  = find_best_feature(dataset, labels)
    feature   = features[ifeature]
    split     = functools.partial(split_dataset, dataset,
                                  labels, ifeature)

    tree = Tree(feature, ifeature, {})
    for value in set(dataset[:, ifeature]):
        subdataset, sublabels = split(value)
        tree.branches[value] = create_branch(subdataset, sublabels,
                                             np.delete(features, ifeature))
    return tree


def create_tree(dataset, labels, features=None):
    if features is None:
        features = npmap(str, range(dataset.shape[1]))
    assert dataset.shape[0] == labels.size
    assert dataset.shape[1] == features.size
    
    return create_branch(dataset, labels, features)


def classify(tree, data):
    if not hasattr(tree, "feature"):
        return tree
    value = data[tree.index]
    return classify(tree.branches[value],
                    np.delete(data, tree.index))


def print_tree(tree, *prepends):
    prepends = list(prepends)
    feature, index, branches = tree
    for value, branch in branches.items():
        prompt = prepends + [(feature, value)]
        if hasattr(branch, "feature"):
            print_tree(branch, *prompt)
        else:
            print("{} => {}".format(", ".join(map(str, prompt)),
                                    branch))

            
def compute_leafs(tree):
    if not hasattr(tree, "branches"):
        return 1
    return sum(compute_leafs(branch)
               for branch in tree.branches.values())


"""
def compute_depth(tree):
    depths = []
    for branch in tree.branches.values():
        if hasattr(branch, "branches"):
            depths.append(1 + compute_depth(branch))
        else:
            depths.append(1)
    return max(depths)
"""

def compute_depth(tree):
    if not hasattr(tree, "branches"):
        return 0
    return max(1 + compute_depth(branch) for branch in tree.branches.values()) 

def plot_branch(node, x0, y0):
    if hasattr(node, "branches"):
        pass
    else:
        plt.annotate(node.feature,
                     xy=(x0, y0),
#                     xycoords   = "axes fraction",
#                     xytext     = centerPt,
#                     textcoords = "axes fraction",
                     va="center", ha="center",
                     bbox       = _decision_node,
                     arrowprops = _arrow_args )


def plot_tree(tree):
    nleafs = compute_leafs(tree)
    depth  = compute_depth(tree)
    fig    = plt.figure()
    plt.xlim(-nleafs, nleafs)
    plt.ylim(-depth, 0)
    plt.xticks([])
    plt.yticks([])
    plot_branch(tree, 0, 0)


def store_tree(tree, filename="tree.pckl"):
    with open(filename, "wb") as f:
        pickle.dump(tree, f)


def read_tree(filename="tree.pckl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

