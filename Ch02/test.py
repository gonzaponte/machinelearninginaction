from kNN import *
from Multivariate import *
from RandomNumbers import LCG


def dummy():
  rng = LCG()
  train_evts, train_labels = createDataSet()
  test_evts = [ [r.Uniform(-1.,1.) for i in range(2)] for j in range(100) ]
  
  ds  = Dataset( train_evts.tolist() )
  knn = kNN( ds, train_labels, default_k = 2 )
  
  result = knn.ClassifyEvents( e )
  for i in zip(test_evts,result): print i
  k.Plot2D(test_evts)
  raw_input('dummy')

def LoadDatingData( f = '' ):
    filename = 'datingTestSet' + str(f) + '.txt'
    ifile = open( filename )
    evts, labels = [], []
    for line in ifile:
        line = line.rstrip().split('\t')
        evts.append( map( float, line[:-1] ) )
        labels.append( line[-1] )
    ifile.close()
    return evts, labels

def test_dating( f = '', ptest = 0.1 ):
    evts, labels = LoadDatingData( f )
    
    ntest = int( ptest * len(evts) )
    
    train_evts   = evts[ntest:]
    train_labels = labels[ntest:]

    test_evts = evts[:ntest]
    test_labels = labels[:ntest]

    ds  = Dataset( train_evts, compute_statistics = False )
    knn = kNN( ds, train_labels, default_k = 3 )
    test_result = knn.ClassifyEvents( test_evts )[0]

    failure_rate = 0.
    for i in range(ntest):
#        print 'classifier: {0}. real: {1}'.format( test_result[i], test_labels[i])
        if test_result[i] != test_labels[i]:
            failure_rate += 1.
    return failure_rate / ntest

def LoadNumbersData( foldername ):
    data, labels = [], []
    for filename in listdir(foldername):
        data.append( reduce( lambda x,y: x+y, [ map( float, line.rstrip() ) for line in open('/'.join([foldername,filename]) ) ] ) )
        labels.append( filename.split('_')[0] )
    
    return data, labels

def test_handwriting():
    train_data, train_labels = LoadNumbersData( 'trainingDigits' )
    test_data , test_labels  = LoadNumbersData( 'testDigits' )
    print 'Data loaded'
    ds = Dataset( train_data, compute_statistics = False )
    knn = kNN( ds, train_labels, 3, False )
    
#    for i in range(10): knn.ClassifyEvent( test_data[i] )
#    return
    test_result = knn.ClassifyEvents( test_data )
    failure_rate = sum( result != true for result, true in zip( test_result, test_labels ) ) / float( len(test_data) )
    
    print failure_rate
    return failure_rate

#test_dating()
test_handwriting()

