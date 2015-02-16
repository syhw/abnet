import numpy as np
import os, cPickle, time, sys
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler, LabelEncoder
sys.path.insert(1, '/home/gsynnaeve/abnet')
from dataset_iterators import DatasetMiniBatchIterator, pad
from layers import ReLU 
from classifiers import LogisticRegression
from nnet_archs import NeuralNet, DropoutNet
from collections import defaultdict

bdir = "/fhgfs/bootphon/scratch/gsynnaeve/BUCKEYE/buckeye_modified_split_devtest/" # wav/test/ or wav/dev/
fatrain = "/fhgfs/bootphon/scratch/gsynnaeve/BUCKEYE/forcedAlign.rec"
fatest = "/fhgfs/bootphon/scratch/gsynnaeve/BUCKEYE/forcedAlign_dev.rec" 

FORCEALIGNED = False
TWOCLASSES = False
PLOTALL = True

dataset_name = "buckeye"
if not FORCEALIGNED:
    dataset_name += "_phn_gold"
features = 'fbanks'
nframes = 11
network_type = "dropout_net"
#network_type = "simple_net"
trainer_type = "adadelta"
layers_types = [ReLU, ReLU, ReLU, ReLU, LogisticRegression]
layers_sizes = [2400, 2400, 2400, 2400]
#layers_sizes = [1000, 1000, 1000, 1000]
#layers_types = [LogisticRegression]
#layers_sizes = []
dropout_rates = [0.2, 0.5, 0.5, 0.5, 0.5]
init_lr = 0.01
max_epochs = 500
if "dropout" in network_type:
    max_epochs = 2000
iterator_type = DatasetMiniBatchIterator
batch_size = 1000
debug_print = 1
debug_time = 1
debug_plot = 0

output_file_name = "buckeye_supervised"
output_file_name += "_" + features + str(nframes)
output_file_name += "_" + network_type + "_" + trainer_type
print "output file name:", output_file_name

n_ins = None
n_outs = None
print "loading dataset from", bdir, fatrain, fatest


def parse(fname):
    """ Parses a forced aligned file into a dict of dicts. """
    d = {}
    with open(fname) as f:
        curr = ""
        buf = {}
        for line in f:
            if "#!MLF!#" in line:
                continue
            if ".rec" in line:
                if curr:
                    d[curr] = buf
                curr = line.strip('\n"').split('/')[-1].split('.')[0]
                buf = {}
            else:
                l = line.strip('\n"').split()
                if len(l) < 2:
                    continue
                s = int(round(float(l[0])/100000))
                e = int(round(float(l[1])/100000))
                if len(l) > 4:
                    p = l[-1]
                    if p == '<s>':
                        p = '!ENTER'
                    elif p == '</s>':
                        p = '!EXIT'
                st = l[2][1:]
                pst = p + '[' + st + ']'
                buf[(s, e)] = pst
    return d


if FORCEALIGNED:
    # Parse forced aligned files
    train_ys = parse(fatrain)
    test_ys = parse(fatest)
    #print train_ys
    #print test_ys
else:
    tmp = defaultdict(lambda: {})
    for dset in ['test', 'dev']:
        for bd, _, files in os.walk(bdir + 'phn/' + dset + '/'):
            for fname in files:
                if ".phn" in fname:
                    curfname = fname.split('.')[0]
                    with open(bd + fname) as rf:
                        for line in rf:
                            s, e, phn = line.rstrip('\n').split()
                            s = int(round(100*float(s)))
                            e = int(round(100*float(e)))
                            tmp[curfname][(s,e)] = phn
        if dset == 'test':
            train_ys = tmp
        else:
            test_ys = tmp
        tmp = defaultdict(lambda: {})


# Load filterbanks of files
fbanks = {}
for dset in ['test', 'dev']:
    for bd, _, files in os.walk(bdir + 'wav/' + dset + '/'):
        for fname in files:
            if ".npy" in fname:
                fbanks[fname.split('.')[0]] = np.load(bd + fname)

#print len(fbanks)
#print len(train_ys)
#print len(test_ys)

# Pad filterbanks
for fn, fb in fbanks.iteritems():
    fbanks[fn] = pad(fb, nframes)

# Duplicate annotations
train_set_y = []
test_set_y = []

def align_transcriptions(fbanks, ys):
    to_ext = []
    for fn in ys.iterkeys():
        fb = fbanks[fn]
        t = ys[fn]
        tmp = np.array(['NONE' for _ in xrange(fb.shape[0])])
        for (s, e), phn in t.iteritems():
            tmp[s:e] = phn
        if tmp[-1] == 'NONE':
            tmp[-1] = tmp[-2]
        if tmp[-2] == 'NONE':
            print >> sys.stderr, "annotation and fbanks differ by more than 1 frame for", fn
        to_ext.extend(tmp)
    return to_ext

train_set_y = align_transcriptions(fbanks, train_ys)
test_set_y = align_transcriptions(fbanks, test_ys)


#train_set_x = np.concatenate([fbanks[k] for k in train_ys.iterkeys()], axis=0)
train_set_x = np.vstack([fbanks[k] for k in train_ys.iterkeys()])
train_set_x = StandardScaler().fit_transform(train_set_x)  # TODO put back
train_set_x = np.array(train_set_x, dtype='float32')
train_set_y = np.array(train_set_y)
print train_set_x.shape

if TWOCLASSES:
    #train_set_x = train_set_x[(train_set_y == "ah[3]") + (train_set_y == "ih[3]")]
    #train_set_y = train_set_y[(train_set_y == "ah[3]") + (train_set_y == "ih[3]")]
    train_set_x = train_set_x[(train_set_y == "ah[3]") + (train_set_y == "s[3]")]
    train_set_y = train_set_y[(train_set_y == "ah[3]") + (train_set_y == "s[3]")]
    #train_set_x = train_set_x[(train_set_y == "!ENTER[3]") + (train_set_y == "!EXIT[3]")]
    #train_set_y = train_set_y[(train_set_y == "!ENTER[3]") + (train_set_y == "!EXIT[3]")]
    print train_set_x.shape

le = LabelEncoder()
train_set_y = le.fit_transform(train_set_y)
print le
train_set_y = np.array(train_set_y, dtype='int32')
print train_set_y.shape

#test_set_x = np.concatenate([fbanks[k] for k in test_ys.iterkeys()], axis=0)
test_set_x = np.vstack([fbanks[k] for k in test_ys.iterkeys()])
#print "means, stds before scaling:", test_set_x.mean(axis=0), test_set_x.std(axis=0)
test_set_x = StandardScaler().fit_transform(test_set_x)  # TODO put back
#print "means, stds after scaling:", test_set_x.mean(axis=0), test_set_x.std(axis=0)
test_set_x = np.array(test_set_x, dtype='float32')
test_set_y = np.array(test_set_y)
print test_set_x.shape

import pylab as pl
import random
if TWOCLASSES:
    #test_set_x = test_set_x[(test_set_y == "ah[3]") + (test_set_y == "ih[3]")]
    #test_set_y = test_set_y[(test_set_y == "ah[3]") + (test_set_y == "ih[3]")]
    test_set_x = test_set_x[(test_set_y == "ah[3]") + (test_set_y == "s[3]")]
    test_set_y = test_set_y[(test_set_y == "ah[3]") + (test_set_y == "s[3]")]
    #test_set_x = test_set_x[(test_set_y == "!ENTER[3]") + (test_set_y == "!EXIT[3]")]
    #test_set_y = test_set_y[(test_set_y == "!ENTER[3]") + (test_set_y == "!EXIT[3]")]
    print test_set_x.shape

if PLOTALL:
    phonests = set(test_set_y)
    for phnst in phonests:
        pl.figure()
        print phnst
        print test_set_x[test_set_y == phnst].shape
        print test_set_x[test_set_y == phnst].mean(axis=0).shape
        pl.imshow(test_set_x[test_set_y == phnst].mean(axis=0).reshape((nframes, 40)).transpose(), interpolation='nearest')
        pl.savefig("mean_" + phnst + ".png")
        for _ in xrange(10):
            pl.figure()
            ii = random.randint(0, test_set_x[test_set_y == phnst].shape[0]-1)
            pl.imshow(test_set_x[test_set_y == phnst][ii].reshape((nframes, 40)).transpose(), interpolation='nearest')
            pl.savefig("sampled_" + str(ii) + "_" + phnst + ".png")

test_set_y = le.transform(test_set_y)
test_set_y = np.array(test_set_y, dtype='int32')
print test_set_y.shape

from sklearn.linear_model import SGDClassifier
s = SGDClassifier()
s.fit(test_set_x, test_set_y)
print "with simple SGD on test/test:", s.score(test_set_x, test_set_y)
s.fit(train_set_x, train_set_y)
print "with simple SGD on train/test:", s.score(test_set_x, test_set_y)
pred = s.predict(test_set_x)
n_outs = len(set(test_set_y))
gold_counts = np.bincount(test_set_y)
print gold_counts
pl.figure(figsize=(24,18))
pl.bar(np.arange(n_outs), gold_counts, width=1.)
pl.xticks(np.arange(n_outs), le.inverse_transform(np.arange(n_outs)), rotation=90, fontsize=9)
pl.savefig("hist_gold.png")
print "predicted for #", len(set(pred))
pred_counts = np.bincount(pred, minlength=n_outs)
print pred_counts
pl.figure(figsize=(24,18))
pl.bar(np.arange(n_outs), pred_counts, width=1.)
pl.xticks(np.arange(n_outs), le.inverse_transform(np.arange(n_outs)), rotation=90, fontsize=9)
pl.savefig("hist_preds.png")

#train_set_x, valid_set_x, train_set_y, valid_set_y = cross_validation\
#        .train_test_split(train_set_x, train_set_y, test_size=0.15,
#                random_state=0)
tmp_set_x = train_set_x
tmp_set_y = train_set_y
split = 0.9*tmp_set_x.shape[0]
train_set_x = tmp_set_x[:split]
train_set_y = tmp_set_y[:split]
valid_set_x = tmp_set_x[split:]
valid_set_y = tmp_set_y[split:]
from sklearn.utils import shuffle
#train_set_x, test_set_y = shuffle(train_set_x, train_set_y)
#valid_set_x, test_set_y = shuffle(valid_set_x, valid_set_y)

assert train_set_x.shape[1] == valid_set_x.shape[1]
assert test_set_x.shape[1] == valid_set_x.shape[1]

print "dataset loaded!"
print "train set size", train_set_x.shape[0]
print "validation set size", valid_set_x.shape[0]
print "test set size", test_set_x.shape[0]
print "phones in train", len(set(train_set_y))
print "phones in valid", len(set(valid_set_y))
print "phones in test", len(set(test_set_y))
n_outs = len(set(train_set_y))

with open(dataset_name + '_LabelEncoder.pickle', 'wb') as f:
    cPickle.dump(le, f, -1)

#with open("buckeye_supervised_train.npz", 'wb') as f:
#    np.savez(f, x=train_set_x, y=train_set_y)  # TODO too big
with open("buckeye_supervised_valid.npz", 'wb') as f:
    np.savez(f, x=valid_set_x, y=valid_set_y)
with open("buckeye_supervised_test.npz", 'wb') as f:
    np.savez(f, x=test_set_x, y=test_set_y)

print "nframes:", nframes
train_set_iterator = iterator_type(train_set_x, train_set_y,
        batch_size=batch_size)
valid_set_iterator = iterator_type(valid_set_x, valid_set_y,
        batch_size=batch_size)
test_set_iterator = iterator_type(test_set_x, test_set_y,
        batch_size=batch_size)
n_ins = test_set_x.shape[1]

assert n_ins != None
assert n_outs != None

# numpy random generator
numpy_rng = np.random.RandomState(123)
print '... building the model'

# TODO the proper network type other than just dropout or not
nnet = None
if "dropout" in network_type:
    nnet = DropoutNet(numpy_rng=numpy_rng, 
            n_ins=n_ins,
            layers_types=layers_types,
            layers_sizes=layers_sizes,
            dropout_rates=dropout_rates,
            n_outs=n_outs,
            max_norm=4.,
            debugprint=debug_print)
else:
    nnet = NeuralNet(numpy_rng=numpy_rng, 
            n_ins=n_ins,
            layers_types=layers_types,
            layers_sizes=layers_sizes,
            n_outs=n_outs,
            debugprint=debug_print)
print "Created a neural net as:",
print str(nnet)

# get the training, validation and testing function for the model
print '... getting the training functions'
print trainer_type
train_fn = None
if debug_plot or debug_print:
    if trainer_type == "adadelta":
        train_fn = nnet.get_adadelta_trainer(debug=True)
    elif trainer_type == "adagrad":
        train_fn = nnet.get_adagrad_trainer(debug=True)
    else:
        train_fn = nnet.get_SGD_trainer(debug=True)
else:
    if trainer_type == "adadelta":
        train_fn = nnet.get_adadelta_trainer()
    elif trainer_type == "adagrad":
        train_fn = nnet.get_adagrad_trainer()
    else:
        train_fn = nnet.get_SGD_trainer()

train_scoref = nnet.score_classif(train_set_iterator)
valid_scoref = nnet.score_classif(valid_set_iterator)
test_scoref = nnet.score_classif(test_set_iterator)
data_iterator = train_set_iterator

print '... training the model'
# early-stopping parameters
patience = 1000  # look as this many examples regardless TODO
patience_increase = 2.  # wait this much longer when a new best is
                        # found
improvement_threshold = 0.995  # a relative improvement of this much is
                               # considered significant

best_validation_loss = np.inf
test_score = 0.
start_time = time.clock()

done_looping = False
epoch = 0
lr = init_lr
timer = None
if debug_plot:
    print_mean_weights_biases(nnet.params)
with open(output_file_name + 'epoch_0.pickle', 'wb') as f:
    cPickle.dump(nnet, f, -1)

while (epoch < max_epochs) and (not done_looping):
    epoch = epoch + 1
    avg_costs = []
    avg_params_gradients_updates = []
    if debug_time:
        timer = time.time()
    for iteration, (x, y) in enumerate(data_iterator):
        avg_cost = 0.
        if "delta" in trainer_type:  # TODO remove need for this if
            avg_cost = train_fn(x, y)
        else:
            avg_cost = train_fn(x, y, lr)
        if type(avg_cost) == list:
            avg_costs.append(avg_cost[0])
        else:
            avg_costs.append(avg_cost)
    if debug_print >= 2:
        print_mean_weights_biases(nnet.params)
    if debug_plot >= 2:
        plot_params_gradients_updates(epoch, avg_params_gradients_updates)
    if debug_time:
        print('  epoch %i took %f seconds' % (epoch, time.time() - timer))
    print('  epoch %i, avg costs %f' % \
          (epoch, np.mean(avg_costs)))
    print('  epoch %i, training error %f' % \
          (epoch, np.mean(train_scoref())))
    # TODO update lr(t) = lr(0) / (1 + lr(0) * lambda * t)
    # or another scheme for learning rate decay

    # we check the validation loss on every epoch
    validation_losses = valid_scoref()
    this_validation_loss = np.mean(validation_losses)  # TODO this is a mean of means (with different lengths)
    print('  epoch %i, validation error %f' % \
          (epoch, this_validation_loss))
    # if we got the best validation score until now
    if this_validation_loss < best_validation_loss:
        with open(output_file_name + '.pickle', 'wb') as f:
            cPickle.dump(nnet, f, -1)
        # improve patience if loss improvement is good enough
        if (this_validation_loss < best_validation_loss *
            improvement_threshold):
            patience = max(patience, iteration * patience_increase)
        # save best validation score and iteration number
        best_validation_loss = this_validation_loss
        # test it on the test set
        test_losses = test_scoref()
        test_score = np.mean(test_losses)  # TODO this is a mean of means (with different lengths)
        print(('  epoch %i, test error of best model %f') %
              (epoch, test_score))

        pred_counts = np.bincount(np.concatenate([nnet.predict(x) for x, _ in test_set_iterator]), minlength=n_outs)
        print pred_counts
        pl.figure(figsize=(24,18))
        pl.bar(np.arange(n_outs), pred_counts, width=1.)
        pl.xticks(np.arange(n_outs), le.inverse_transform(np.arange(n_outs)), rotation=90, fontsize=9)
        pl.savefig("hist_preds_" + str(epoch) + ".png")

    if patience <= iteration:  # TODO correct that
        done_looping = True
        break

end_time = time.clock()
print(('Optimization complete with best validation score of %f, '
       'with test performance %f') %
             (best_validation_loss, test_score))
print >> sys.stderr, ('The fine tuning code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time)
                                          / 60.))
with open(output_file_name + '_final.pickle', 'wb') as f:
    cPickle.dump(nnet, f, -1)
