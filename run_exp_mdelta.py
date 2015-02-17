import os, time, cPickle, sys
import numpy as np

SPACES = [15, 20, 25, 30]
NFRAMES = 7
MIN_N_FRAMES = 2*max(SPACES) + NFRAMES
BATCH_SIZE = 500
DIM_EMBEDDING = 100

output_file_name = "mdelta_test"

x1_same = []
x2_same = []
x2_diff = []

bdir = "/fhgfs/bootphon/scratch/gsynnaeve/BUCKEYE/buckeye_modified_split_devtest/"
for dset in ['test', 'dev']:
#for dset in ['dev']:
    for bd, _, files in os.walk(bdir + 'wav/' + dset + '/'):
        for fname in files:
            if fname[-4:] == '.npy':
                npyfname = bd + fname
                with open(npyfname) as rf:
                    tmp = np.load(rf)
                    if tmp.shape[0] < MIN_N_FRAMES:
                        continue

                    stacked = np.ndarray((tmp.shape[0] - NFRAMES, tmp.shape[1]*NFRAMES))
                    for i in xrange(stacked.shape[0]):
                        stacked[i] = tmp[i:i+NFRAMES].flatten()

                    x1_same.append(stacked[:-1])
                    x2_same.append(stacked[1:])
                    for sp in SPACES:
                        x2_diff.append(stacked[sp:stacked.shape[0]/2+sp])
                        x2_diff.append(stacked[stacked.shape[0]/2-sp:-sp-1])

x1_tmp = np.concatenate(x1_same, axis=0)
nby = (len(SPACES)+1)
x1 = np.ndarray((nby*x1_tmp.shape[0], x1_tmp.shape[1]), dtype='float32')
x1[::nby] = x1_tmp
x1[1::nby] = x1_tmp
x1[2::nby] = x1_tmp
x1[3::nby] = x1_tmp
x1[4::nby] = x1_tmp
x2_same_tmp = np.concatenate(x2_same, axis=0)
x2_same_tmp = x2_same_tmp.reshape((x2_same_tmp.shape[0], 1, x2_same_tmp.shape[1]))
x2_diff_tmp = np.concatenate(x2_diff, axis=0)
x2_diff_tmp = x2_diff_tmp.reshape((x2_diff_tmp.shape[0]/len(SPACES), len(SPACES), x2_diff_tmp.shape[1]))
x2 = np.asarray(np.concatenate([x2_same_tmp, x2_diff_tmp], axis=1), dtype='float32')
x2 = x2.reshape((nby*x2_same_tmp.shape[0], x2_same_tmp.shape[2]))
y = np.zeros(x1.shape[0], dtype='int32')
y[::5] = 1

print x1.shape
print x2.shape

from sklearn.preprocessing import scale
x1 = scale(x1)
x2 = scale(x2)

from dataset_iterators import DatasetABIterator

from sklearn import cross_validation
l = x1.shape[0]
stop = 0.9*l
x1_test = x1[stop:]
x2_test = x2[stop:]
y_test = y[stop:]

x1_train, x1_dev, x2_train, x2_dev, y_train, y_dev = cross_validation.train_test_split(
        x1[:stop], x2[:stop], y[:stop])

train_it = DatasetABIterator(x1_train, x2_train, y_train, batch_size=BATCH_SIZE)
dev_it = DatasetABIterator(x1_dev, x2_dev, y_dev, batch_size=BATCH_SIZE)
test_it = DatasetABIterator(x1_test, x2_test, y_test, batch_size=BATCH_SIZE)
numpy_rng = np.random.RandomState(123)
from nnet_archs import ABNeuralNet
from layers import ReLU, SigmoidLayer, Linear
nnet = ABNeuralNet(numpy_rng=numpy_rng, 
        n_ins=x1_train.shape[1],
        layers_types=[SigmoidLayer, SigmoidLayer, SigmoidLayer],
        layers_sizes=[500, 500],
        #layers_types=[Linear],
        #layers_sizes=[],
        n_outs=DIM_EMBEDDING,
        loss='cos_cos2',
        rho=0.90,
        eps=1.E-5,
        max_norm=4.,
        debugprint=1)
print nnet

train_fn = nnet.get_adadelta_trainer(debug=True)
#train_fn = nnet.get_SGD_trainer(debug=True)
train_scoref = nnet.score_classif_same_diff_separated(train_it)
valid_scoref = nnet.score_classif_same_diff_separated(dev_it)
test_scoref = nnet.score_classif_same_diff_separated(test_it)

best_validation_loss = np.inf
test_score = 0.
max_epochs = 100
epoch = 0
data_iterator = train_it
start_time = time.clock()

tmp_train = zip(*train_scoref())
print('  epoch %i, training sim same %f, diff %f' % \
      (epoch, np.mean(tmp_train[0]), np.mean(tmp_train[1])))
validation_losses = zip(*valid_scoref())
print('  epoch %i, valid sim same %f, diff %f' % \
      (epoch, np.mean(validation_losses[0]), np.mean(validation_losses[1])))
test_losses = zip(*test_scoref())
print('  epoch %i, test sim same %f, diff %f' % \
      (epoch, np.mean(test_losses[0]), np.mean(test_losses[1])))

print "Starting the training..."

while (epoch < max_epochs):
    epoch = epoch + 1
    avg_costs = []
    timer = time.time()
    for iteration, (x, y) in enumerate(data_iterator):
        #print "x[0][0]", x[0][0]
        #print "x[1][0]", x[1][0]
        #print "y[0][0]", y[0][0]
        #print "y[1][0]", y[1][0]
        avg_cost = 0.
        avg_cost = train_fn(x[0], x[1], y)
        #avg_cost = train_fn(x[0], x[1], y, lr=0.0001)
        if type(avg_cost) == list:
            avg_costs.append(avg_cost[0])
        else:
            avg_costs.append(avg_cost)
    print('  epoch %i took %f seconds' % (epoch, time.time() - timer))
    avg_cost = np.mean(avg_costs)
    if np.isnan(avg_cost):
        print("avg costs is NaN so we're stopping here!")
        break
    print('  epoch %i, avg costs %f' % \
          (epoch, avg_cost))
    tmp_train = zip(*train_scoref())
    print('  epoch %i, training sim same %f, diff %f' % \
          (epoch, np.mean(tmp_train[0]), np.mean(tmp_train[1])))

    # we check the validation loss on every epoch
    validation_losses = zip(*valid_scoref())
    this_validation_loss = 0.5*(1.-np.mean(validation_losses[0])) +\
            0.5*np.mean(validation_losses[1])

    print('  epoch %i, valid sim same %f, diff %f' % \
          (epoch, np.mean(validation_losses[0]), np.mean(validation_losses[1])))
    # if we got the best validation score until now
    if this_validation_loss < best_validation_loss:
        with open(output_file_name + '.pickle', 'wb') as f:
            cPickle.dump(nnet, f, protocol=-1)
        # save best validation score and iteration number
        best_validation_loss = this_validation_loss
        # test it on the test set
        test_losses = zip(*test_scoref())
        print('  epoch %i, test sim same %f, diff %f' % \
              (epoch, np.mean(test_losses[0]), np.mean(test_losses[1])))

end_time = time.clock()
print(('Optimization complete with best validation score of %f, '
       'with test performance %f') %
             (best_validation_loss, test_score))
print >> sys.stderr, ('The fine tuning code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time)
                                          / 60.))
with open(output_file_name + '_final.pickle', 'wb') as f:
    cPickle.dump(nnet, f, protocol=-1)
