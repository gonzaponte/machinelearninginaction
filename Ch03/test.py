from Multivariate import *
from trees import *

def PlotTree( tree, previous_branches = [] ):
    for key, value in tree.items():
        if isinstance( value, dict ):
            PlotTree( value, previous_branches + [key] )
        else:
            print '{0} => {1}'.format( ' + '.join(map(str,previous_branches + [key])), value )

#################################
##### First check
#################################

data, properties = createDataSet()
labels = [ d.pop() for d in data ]

t = DiscreteDecissionTree( data, labels, properties )

PlotTree( t.tree )

#################################
##### Second check
#################################

lens_data = [ line.rstrip().split('\t') for line in open('lenses.txt') ]
labels = zip(*lens_data)[-1]
data = zip(*zip(*lens_data)[:-1])
properties = ['age','prescript','astigmatic','tearRate']

t = DiscreteDecissionTree( data, labels, properties )
PlotTree( t.tree )

#################################
##### Third check
#################################

data, properties = createDataSet()

testdata  = data.pop(0)
testlabel = testdata[-1]
testdata  = testdata[:-1]

labels = [ d.pop() for d in data ]

dt = DiscreteDecissionTree( data, labels, properties )

print testlabel, dt.Classify( testdata )
