from bayes import *
from Multivariate import NaiveBayes

data, classes = loadDataSet()
vocabulary = createVocabList(data)
traindata = [ setOfWords2Vec(vocabulary,x) for x in data ]

testingNB()

test_non = setOfWords2Vec(vocabulary,['love', 'my', 'dalmation'])
test_yes = setOfWords2Vec(vocabulary,['stupid', 'garbage'])

myNB = NaiveBayes(traindata,classes)
print 'test non classified as', myNB.Classify(test_non)
print 'test yes classified as', myNB.Classify(test_yes)
