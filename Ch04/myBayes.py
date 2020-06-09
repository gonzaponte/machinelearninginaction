import numpy as np
import operator as op
import functools
import collections
import feedparser
import shelve
import re
import os

from utils import nplist, lmap, npmap, frequencies, listdir, timethis

def load_data_set():
    words = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
             ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
             ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
             ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
             ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
             ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    labels = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
    labels = ["non abusive", "abusive"] * 3
    return words, np.array(labels)


def tokenize_text(text):
    regex    = re.compile("\\W*")
    return lmap(str.lower,
                filter(lambda x: len(x) > 2,
                       regex.split(text)))


def load_mail_data_set():
    read_from_file = lambda filename: open(filename,
                                           errors = "ignore").read()
    spam_data = lmap(tokenize_text,
                     map(read_from_file,
                         listdir("email/spam/")))
    ham_data  = lmap(tokenize_text,
                     map(read_from_file,
                         listdir("email/ham/" )))

    dataset = spam_data + ham_data
    labels  = ["spam"] * len(spam_data) + ["ham"] * len(ham_data)
    return dataset, np.array(labels)


def download_feeds(filename="feeds", overwrite=False):
    if os.path.exists(filename + ".db") and not overwrite:
        return
    with shelve.open(filename) as outfile:
        outfile["NY"] = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
        outfile["SF"] = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')


def load_feeds(filename="feeds"):
    with shelve.open(filename) as infile:
        return infile["NY"], infile["SF"]


def vocabulary(dataset):
    return nplist(sorted(functools.reduce(set.union, map(set, dataset))))


def to_vector(vocabulary, data):
    data = np.array(data)
    return nplist(np.count_nonzero(data == w) for w in vocabulary)


def dataset_to_vector(vocabulary, dataset):
    return npmap(functools.partial(to_vector, vocabulary), dataset)


def train(dataset, labels):
    label_probs = {lbl: np.log(p) for lbl, p in frequencies(labels).items()}
    class_probs = {}
    for label in set(labels):
        v = dataset[labels == label].sum(axis=0)
        class_probs[label] = np.log((1+v) / (2+np.sum(v)))
    return class_probs, label_probs


def classify(class_probs, label_probs, data):
    probs = {lbl: np.sum(data * class_probs[lbl]) + lbl_prob
             for lbl, lbl_prob in label_probs.items()}
    return max(probs.items(),
               key = op.itemgetter(1))[0]


def test_spam(train_ratio = 0.8):    
    mails, labels = load_mail_data_set()
    vocab = vocabulary(mails)
    dataset = dataset_to_vector(vocab, mails)
    
    ntrain  = int(train_ratio * len(dataset))
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    train_dataset = dataset[indices[:ntrain]]
    train_labels  = labels [indices[:ntrain]]
    test_dataset  = dataset[indices[ntrain:]]
    test_labels   = labels [indices[ntrain:]]
    
    class_probs, label_probs = train(train_dataset, train_labels)
    classifier_output = npmap(functools.partial(classify, class_probs, label_probs),
                              test_dataset)
    return (1 - np.count_nonzero(test_labels == classifier_output) / test_labels.size) * 100


def test_feeds(train_ratio=0.8, top_remove=30):
    download_feeds()
    nyfeed, sffeed = load_feeds()

    entries  = min(map(len, (x["entries"] for x in [nyfeed, sffeed])))
    nywords  = lmap(tokenize_text,
                    [nyfeed["entries"][i]['summary'] for i in range(entries)])
    sfwords  = lmap(tokenize_text,
                    [sffeed["entries"][i]['summary'] for i in range(entries)])

    words    = nywords + sfwords
    labels   = np.array(["NY"] * entries + ["SF"] * entries)

    vocab    = vocabulary(words)
    freqs    = frequencies(np.concatenate(words))

    top_used = sorted(freqs.items(), key=op.itemgetter(1))[-top_remove:]
    vocab    = nplist(filter(lambda x: x not in top_used, vocab))
    dataset  = dataset_to_vector(vocab, words)

    ntrain  = int(train_ratio * 2 * entries)
    indices = np.arange(2 * entries)
    np.random.shuffle(indices)

    train_dataset = dataset[indices[:ntrain]]
    train_labels  = labels [indices[:ntrain]]
    test_dataset  = dataset[indices[ntrain:]]
    test_labels   = labels [indices[ntrain:]]

    class_probs, label_probs = train(train_dataset, train_labels)
    classifier_output = npmap(functools.partial(classify, class_probs, label_probs),
                              test_dataset)
    error_rate = (1 - np.count_nonzero(test_labels == classifier_output) / test_labels.size) * 100
    
    test_info = collections.namedtuple("test_info", "error_rate class_probs label_probs vocabulary")
    output = test_info(error_rate, class_probs, label_probs, vocab)
    return output
