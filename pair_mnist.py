import numpy, joblib, random, sys
from sklearn.datasets import fetch_mldata
SCALE = False
N_SAMPLES = 10
mnist = fetch_mldata('MNIST original')
X = numpy.asarray(mnist.data, dtype='uint8')
if SCALE:
    X = numpy.asarray(X, dtype='float32')
    #X = preprocessing.scale(X)
    X /= 255.
y = numpy.asarray(mnist.target, dtype='uint8')
X_train = X[:60000]
X_test = X[60000:]
y_train = y[:60000]
y_test = y[60000:]


for X, y, name in ((X_train, y_train, "train"), (X_test, y_test, "test")):
    assert(X.shape[0] == y.shape[0])
    print name
    pairs = []
    i_start = 0
    n_start = 0
    nc = {}
    for n in range(10):
        nc[n] = numpy.sum(y==n)
    for i, n in enumerate(y):
        sys.stdout.write('%s\r' % str(i))
        sys.stdout.flush()
        if n != n_start:
            n_start = n
            i_start = i
        else:
            for _ in xrange(N_SAMPLES):  #
                j = random.randint(i_start, i_start+nc[n]-1)  #
                while j == i:  #
                    j = random.randint(i_start, i_start+nc[n]-1)  #
            #for j in xrange(i_start, i):  # If we do not sample we should be doing that
                pairs.append((X[i], X[j], 1))
    nonpairs = []
    m = X.shape[0]-1
    print "pairs of same:", len(pairs)
    print ""
    for i in xrange(len(pairs)):
        if (i % 1000) == 0:
            sys.stdout.write('%s\r' % str(i))
            sys.stdout.flush()
        j = random.randint(0, m)
        ii = i % len(y)
        while y[j] == y[ii]:
            j = random.randint(0, m)
        nonpairs.append((X[ii], X[j], 0))
    both = [None] * (len(pairs)+len(nonpairs))
    both[::2] = pairs
    both[1::2] = nonpairs
    print ""
    print "pairs of same:", len(pairs)
    print "pairs of different:", len(nonpairs)
    print ""
    print "saving pairs"
    print ""
    joblib.dump(both, "MNIST_" + name + ".joblib", compress=3, cache_size=2048)


