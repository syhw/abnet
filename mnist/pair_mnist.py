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
    pairs_1 = numpy.ndarray((X.shape[0] * N_SAMPLES, X.shape[1]), dtype='uint8')
    pairs_2 = numpy.ndarray((X.shape[0] * N_SAMPLES, X.shape[1]), dtype='uint8')
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
            for k in xrange(N_SAMPLES):  #
                j = random.randint(i_start, i_start+nc[n]-1)  #
                while j == i:  #
                    j = random.randint(i_start, i_start+nc[n]-1)  #
            #for j in xrange(i_start, i):  # If we do not sample we should be doing that
                pairs_1[i*N_SAMPLES+k] = X[i]
                pairs_2[i*N_SAMPLES+k] = X[j]
    nonpairs_1 = numpy.ndarray((pairs_1.shape[0], X.shape[1]), dtype='uint8')
    nonpairs_2 = numpy.ndarray((pairs_1.shape[0], X.shape[1]), dtype='uint8')
    m = X.shape[0]-1
    print "pairs of same:", len(pairs_1)
    print ""
    for i in xrange(nonpairs_1.shape[0]):
        if (i % 1000) == 0:
            sys.stdout.write('%s\r' % str(i))
            sys.stdout.flush()
        j = random.randint(0, m)
        ii = i % len(y)
        while y[j] == y[ii]:
            j = random.randint(0, m)
        nonpairs_1[i] = X[ii]
        nonpairs_2[i] = X[j]
    both_1 = numpy.ndarray((pairs_1.shape[0]+nonpairs_1.shape[0], pairs_1.shape[1]), dtype='uint8')
    both_2 = numpy.ndarray((pairs_1.shape[0]+nonpairs_1.shape[0], pairs_1.shape[1]), dtype='uint8')
    labels = numpy.zeros(pairs_1.shape[0]+nonpairs_1.shape[0], dtype='uint8')
    labels[::2] = 1
    both_1[::2] = pairs_1
    both_1[1::2] = nonpairs_1
    both_2[::2] = pairs_2
    both_2[1::2] = nonpairs_2
    print ""
    print "pairs of same:", len(pairs_1)
    print "pairs of different:", len(nonpairs_1)
    print ""
    print "saving pairs"
    print ""
    joblib.dump((both_1, both_2, labels),
            "MNIST_" + name + ".joblib", compress=3, cache_size=2048)


