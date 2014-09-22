import cPickle
import numpy as np
from sklearn.datasets import fetch_mldata


with open('deep_cos_cos2_MNIST_ab_net_adadelta_emb_100.pickle') as f:
    nnet = cPickle.load(f) 
mnist = fetch_mldata('MNIST original')
#X = np.asarray(mnist.data, dtype='uint8')
X = np.asarray(mnist.data, dtype='float32')
X /= 255.
print X.mean()
print X.max()
print X.min()
y = np.asarray(mnist.target, dtype='uint8')
X_train = X[:60000]
X_test = X[60000:]
y_train = y[:60000]
y_test = y[60000:]
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape

transform = nnet.transform_x1()
embedded_X_train = transform(X_train)
embedded_X_test = transform(X_test)

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#clf = SGDClassifier()
#clf = SGDClassifier(loss='log')
#clf = LogisticRegression()
clf = SVC(kernel="rbf", C=2.8, gamma=.0073)
#clf = SVC()

clf.fit(X_train, y_train)
print "acc on input features:", clf.score(X_test, y_test)

clf.fit(embedded_X_train, y_train)
print "acc on embedded features:", clf.score(embedded_X_test, y_test)
